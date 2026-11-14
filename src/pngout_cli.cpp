// Command line front end for the reconstructed PNGOUT.
//
// The reference binary accepts an input file, an optional output file and an
// option list. Options may appear before or after the file names; the first
// non-option argument is the input and the second the output. When the output
// is omitted the input file is optimised in place (PNGOUT replaces the
// extension with .png, which is a no-op for a .png input).
//
// Everything the reference writes, including errors and warnings, goes to
// stdout. The only exception is the overwrite prompt's answer, which is read
// from stdin.
#include "png_image.hpp"

#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

const char kUsage[] =
   "PNGOUT [In:{PNG,JPG,GIF,TGA,PCX,BMP}] (Out:PNG) (options...)        Jan 15 "
   "2020\n"
   "by Ken Silverman (http://advsys.net/ken)\n"
   "Linux port by Jonathon Fowler (http://www.jonof.id.au/pngout)\n"
   "PNGOUT optimizes PNG size losslessly using my own deflate algorithm (not "
   "Zlib)\n"
   "With the right options, it can often beat other programs by 5-10%. "
   "Options:\n"
   "   -c# PNG output color: 0=Gray, 2=RGB, 3=Pal, 4=Gray+Alpha, 6=RGB+Alpha\n"
   "   -f# PNG output filter: 0=none, 1=\x7fx, 2=\x7fy, 3=\x7fx&y, 4=Paeth, "
   "5=mixed, 6=reuse\n"
   "   -d# Override default bitdepth: 0(min),1,2,4,8 (valid only in -c0,-c3 "
   "modes)\n"
   "   -s# Select strategy. 0:Xtreme(default), 1:Intense, 2:Longest Match,\n"
   "       3:Huffman Only, 4:Uncompressed\n"
   "   -b# Block split threshold (lower=more blocks, 0=1 block/file, "
   "default=256)\n"
   "       Use trial&error! Suggested values to try are: "
   "0,128,192,256,512,1024,..\n"
   "   -n# Exact number of Huffman blocks (overrides -b#)\n"
   "   -r  Randomized initial tables (good for many trials with same "
   "options)\n"
   "   -k? 0=Remove optional chunks (default), 1=Keep all\n"
   "       p=Keep palette indices, s=Keep settings for -c,f,d,b\n"
   "   -k(Chunk,Chunk,..) Preserve only named chunk(s). Example: "
   "-kgAMA,bKGD,tEXt\n"
   "   -v,q,l Verbose,Quiet,List mode (use '-' as filename to specify "
   "stdin/stdout)\n"
   "   -y  Assume Yes at the 'overwrite file?' prompt\n"
   "   -force  Write file even if bigger.\n"
   "   -mincodes#  Workaround for buggy decoders. 1:Zlib 1.2.1 bug, 2:buggy "
   "mobiles\n"
   "The 1st filename is the input. If you omit the output\n"
   "filename, PNGOUT will use the same filename with a .PNG extension. "
   "Examples:\n"
   "   $ pngout inlarge.bmp outsmall.png           <- writes outsmall.png\n"
   "   $ pngout duke3d.png -c2 -f3 -b128 -kbKGD -v <- writes duke3d.png if "
   "smaller\n";

bool g_quiet = false;

// printf that honours -q. The reference routes every diagnostic through the
// same buffered writer, so a single choke point keeps the streams identical.
void say(const char* format, ...) {
   if (g_quiet) return;
   std::va_list args;
   va_start(args, format);
   std::vprintf(format, args);
   va_end(args);
}

bool read_file(const char* path, std::vector<std::uint8_t>& data) {
   std::FILE* file = std::fopen(path, "rb");
   if (file == nullptr) return false;
   std::fseek(file, 0, SEEK_END);
   const long size = std::ftell(file);
   std::fseek(file, 0, SEEK_SET);
   if (size < 0) {
      std::fclose(file);
      return false;
   }
   data.resize(static_cast<std::size_t>(size));
   const std::size_t read =
      data.empty() ? 0 : std::fread(data.data(), 1, data.size(), file);
   std::fclose(file);
   return read == data.size();
}

bool write_file(const char* path, const std::vector<std::uint8_t>& data) {
   std::FILE* file = std::fopen(path, "wb");
   if (file == nullptr) return false;
   const std::size_t written =
      data.empty() ? 0 : std::fwrite(data.data(), 1, data.size(), file);
   std::fclose(file);
   return written == data.size();
}

bool file_exists(const char* path) {
   std::FILE* file = std::fopen(path, "rb");
   if (file == nullptr) return false;
   std::fclose(file);
   return true;
}

int parse_int(const char* text) {
   return static_cast<int>(std::strtol(text, nullptr, 10));
}

bool equals_ignore_case(const char* a, const char* b) {
   while (*a != '\0' && *b != '\0') {
      unsigned char ca = static_cast<unsigned char>(*a++);
      unsigned char cb = static_cast<unsigned char>(*b++);
      if (ca >= 'a' && ca <= 'z') ca = static_cast<unsigned char>(ca - 'a' + 'A');
      if (cb >= 'a' && cb <= 'z') cb = static_cast<unsigned char>(cb - 'a' + 'A');
      if (ca != cb) return false;
   }
   return *a == '\0' && *b == '\0';
}

// The filter value shown on the "In:" line. The reference looks at the row
// filter bytes it decoded: a single value when every row agrees, otherwise 5
// (mixed), with the special case of "Paeth first, Sub elsewhere" reported as 3.
int source_filter_display(const pngout::Image& image) {
   if (image.filters.empty()) return -1;
   const int first = image.filters[0];
   if (image.height <= 1) return first;
   std::uint32_t mask = 0;
   for (std::size_t i = 1; i < image.filters.size(); ++i) {
      mask |= 1u << (image.filters[i] & 31u);
   }
   if ((~(1u << (first & 31u)) & mask) == 0) return first;
   if (first == 1 && (mask & ~8u) == 0) return 3;
   return 5;
}

bool chunk_required(std::uint32_t type, int requested_color_type) {
   constexpr std::uint32_t ihdr = 0x49484452u;
   constexpr std::uint32_t plte = 0x504c5445u;
   constexpr std::uint32_t idat = 0x49444154u;
   constexpr std::uint32_t iend = 0x49454e44u;
   constexpr std::uint32_t trns = 0x74524e53u;
   if (type == ihdr || type == plte || type == idat || type == iend) return true;
   // A colour-key tRNS chunk is structural unless the user explicitly asked
   // for an alpha-capable output, which stores the same information in the
   // alpha samples.
   if (type == trns) {
      return requested_color_type != 4 && requested_color_type != 6;
   }
   return false;
}

// Verbose chunk table, printed before the "In:" line.
void print_chunk_table(const pngout::Image& image,
                       const pngout::EncodeOptions& options) {
   say("                Name  Offset   Length     CRC\n");
   for (const pngout::PngChunk& chunk : image.chunks) {
      if (chunk_required(chunk.type, options.color_type)) {
         say("Required chunk: ");
      } else if (pngout::chunk_is_kept(chunk.type, options)) {
         say("    Keep chunk: ");
      } else {
         say("  Remove chunk: ");
      }
      say("%c%c%c%c %08x %08x %08x\n",
          static_cast<char>((chunk.type >> 24) & 0xffu),
          static_cast<char>((chunk.type >> 16) & 0xffu),
          static_cast<char>((chunk.type >> 8) & 0xffu),
          static_cast<char>(chunk.type & 0xffu),
          static_cast<unsigned>(chunk.offset),
          static_cast<unsigned>(chunk.data.size()),
          static_cast<unsigned>(chunk.crc));
   }
   say("\n");
}

// One phase of the row-by-row progress display. The reference prints the
// percentage only when it changes, so repeated values are skipped.
void print_progress(std::uint32_t height) {
   if (height == 0) return;
   // Each phase starts on a fresh line position; the reference emits this
   // carriage return before the first percentage of every pass.
   say("\r");
   int last = -1;
   for (std::uint32_t y = 0; y < height; ++y) {
      const int percent =
         static_cast<int>((static_cast<long long>(y) * 100) / height);
      if (percent > last) {
         say("%2d%% \r", percent);
         last = percent;
      }
   }
}

// One "Out:" size line. In normal mode the line begins with a carriage return
// and has no terminator; verbose mode uses a plain newline instead.
void print_out_line(std::size_t total, long long delta, bool verbose) {
   if (!verbose) say("\r");
   say("Out:%8d", static_cast<int>(total));
   if (delta >= 1) {
      const long long whole = delta >> 3;
      const long long fraction = ((delta & 7) * 125) / 100;
      if (whole == 0) {
         say("     -0");
      } else {
         say("%7d", static_cast<int>(-whole));
      }
      const long long bits = static_cast<long long>(total) * 8 - delta;
      say(".%1d=%7d.%1d", static_cast<int>(fraction),
          static_cast<int>(bits >> 3),
          static_cast<int>(((bits & 7) * 125) / 100));
   } else {
      say(" bytes               ");
   }
   if (verbose) say("\n");
}

// Ask the overwrite question. Only the first character matters and case is
// ignored; 'y' accepts, 'n' or end of input cancels, anything else repeats.
bool confirm_overwrite(const char* path) {
   say("%s already exists. Overwrite? (y/n) ", path);
   char buffer[0x80];
   for (;;) {
      if (std::fgets(buffer, sizeof buffer, stdin) == nullptr) break;
      const unsigned char first =
         static_cast<unsigned char>(buffer[0]) & 0xdfu;
      if (first == 'N') break;
      if (first == 'Y') return true;
      say("Unrecognised response. Overwrite? (y/n) ");
   }
   say("Operation cancelled.\n");
   return false;
}

} // namespace

int main(int argc, char** argv) {
   pngout::EncodeOptions options;
   options.filter = -1;
   options.color_type = -1;
   options.bit_depth = -1;

   bool assume_yes = false;  // -y
   bool force_write = false; // -force
   bool usage_requested = false;
   std::vector<std::string> files;

   for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      if (argument.empty() || argument[0] != '-') {
         files.push_back(argument);
         continue;
      }
      if (equals_ignore_case(argument.c_str() + 1, "FORCE")) {
         force_write = true;
         continue;
      }
      if (equals_ignore_case(argument.c_str() + 1, "MINCODES") ||
          equals_ignore_case(argument.c_str() + 1, "ZL121")) {
         continue;
      }
      char letter = argument.size() > 1 ? argument[1] : '\0';
      if (letter >= 'a' && letter <= 'z') {
         letter = static_cast<char>(letter - 'a' + 'A');
      }
      switch (letter) {
      case 'B':
         options.block_threshold = std::strtod(argument.c_str() + 2, nullptr);
         if (options.block_threshold < 0.0) {
            options.block_threshold = -options.block_threshold;
         }
         break;
      case 'C': {
         int value = 0;
         if (argument.size() > 2) {
            const char* text = argument.c_str() + 2;
            if (*text < '0' || *text > '9') {
               say("Invalid coltype\n");
               usage_requested = true;
               break;
            }
            value = parse_int(text);
            if (value >= 7 || (value & 3) == 1) {
               say("Invalid coltype\n");
               usage_requested = true;
               break;
            }
         }
         options.color_type = value;
         break;
      }
      case 'D': {
         const int value = parse_int(argument.c_str() + 2);
         options.bit_depth = -1;
         if (value <= 8 &&
             ((1 << value) & 0x117) != 0) { // 1, 2, 4, 8 (and 0 = minimum)
            options.bit_depth = value;
         }
         break;
      }
      case 'F': {
         if (argument.size() == 2) {
            options.filter = 0;
            break;
         }
         const char* text = argument.c_str() + 2;
         if (*text < '0' || *text > '9') {
            say("Invalid filter\n");
            usage_requested = true;
            break;
         }
         const int value = parse_int(text);
         if (value >= 7) {
            say("Invalid filter\n");
            usage_requested = true;
            break;
         }
         options.filter = value;
         break;
      }
      case 'K': {
         const char* spec = argument.c_str() + 2;
         if (spec[0] == '\0') break;
         const char upper =
            static_cast<char>((spec[0] >= 'a' && spec[0] <= 'z')
                                 ? spec[0] - 'a' + 'A'
                                 : spec[0]);
         if (spec[1] == '\0' && upper == 'P') break; // keep palette indices
         if (spec[1] == '\0' && upper == 'S') break; // keep settings
         options.keep_chunks = true;
         options.keep_chunk_list = spec;
         break;
      }
      case 'L':
         break; // list mode
      case 'N':
         options.block_count =
            static_cast<std::size_t>(parse_int(argument.c_str() + 2));
         break;
      case 'Q':
         options.quiet = true;
         g_quiet = true;
         break;
      case 'R':
         break; // randomised initial tables
      case 'S': {
         int value = parse_int(argument.c_str() + 2);
         if (value > 3) value = 4;
         if (value < 1) value = 0;
         options.strategy = value;
         break;
      }
      case 'V':
         options.verbose = true;
         break;
      case 'Y':
         if (argument.size() == 2) {
            assume_yes = true;
         } else {
            say("Warning: unrecognized option: %s\n", argument.c_str());
         }
         break;
      case '?':
         usage_requested = true;
         break;
      default:
         say("Warning: unrecognized option: %s\n", argument.c_str());
         break;
      }
   }

   if (usage_requested) {
      if (!g_quiet) std::fputs(kUsage, stdout);
      return 1;
   }
   if (files.empty()) {
      say("Must specify a source or target image\n");
      if (!g_quiet) std::fputs(kUsage, stdout);
      return 1;
   }
   if (files.size() > 2) {
      say("Too many filenames specified on cmdline\n");
      if (!g_quiet) std::fputs(kUsage, stdout);
      return 1;
   }

   std::string input_path = files[0];
   std::string output_path;
   bool in_place = false;
   if (files.size() == 2) {
      output_path = files[1];
   } else {
      // Derive the output name: replace the extension with .png.
      output_path = input_path;
      const std::size_t dot = output_path.find_last_of('.');
      const std::size_t slash = output_path.find_last_of('/');
      if (dot != std::string::npos &&
          (slash == std::string::npos || dot > slash)) {
         output_path.erase(dot);
      }
      output_path += ".png";
      in_place = true;
   }

   std::vector<std::uint8_t> input;
   if (!read_file(input_path.c_str(), input)) {
      say("%s not found\n", input_path.c_str());
      return 1;
   }

   const bool is_png =
      input.size() >= 8 && input[0] == 0x89 && input[1] == 0x50 &&
      input[2] == 0x4e && input[3] == 0x47 && input[4] == 0x0d &&
      input[5] == 0x0a && input[6] == 0x1a && input[7] == 0x0a;

   pngout::Image image;
   std::string error;
   if (!pngout::decode_image(input, image, error)) {
      say("%s unsupported format\n", input_path.c_str());
      return 1;
   }

   if (options.verbose && is_png) print_chunk_table(image, options);

   if (!in_place && !assume_yes && file_exists(output_path.c_str())) {
      if (!confirm_overwrite(output_path.c_str())) return 1;
   }

   const int source_color_type = image.source_color_type;
   const int source_depth = image.bit_depth;
   const int source_filter = source_filter_display(image);

   say(" In:                             %s -c%d", input_path.c_str(),
       source_color_type);
   if (source_filter < 0) {
      say("    ");
   } else {
      say(" -f%d", source_filter);
   }
   if (source_color_type == 0 || source_color_type == 3) {
      say(" -d%d", source_depth);
   }
   say("\r In:%8d bytes\n", static_cast<int>(input.size()));

   say("Out:                             %s", output_path.c_str());

   const auto start = std::chrono::steady_clock::now();
   std::vector<std::uint8_t> output;
   pngout::EncodeStats stats;
   stats.input_size = input.size();
   const bool encoded =
      pngout::encode_png(image, options, output, stats, error);
   const auto finish = std::chrono::steady_clock::now();
   const double elapsed =
      std::chrono::duration<double>(finish - start).count();

   if (!encoded) {
      say("\r%s", error.c_str());
      say("\n");
      return 3;
   }

   say(" -c%d -f%d", stats.color_type, stats.filter);
   if (stats.color_type == 0) {
      say(" -d%d", stats.bit_depth);
   } else if (stats.color_type == 3) {
      say(" -d%d, %d colors", stats.bit_depth,
          static_cast<int>(stats.palette_colors));
   }

   // Block splitting runs a throwaway statistics pass before the real ones.
   // With -b0 (and no -n) there is no throwaway pass at all; verbose mode
   // instead prints a blank line right after the output options.
   const bool split_enabled =
      options.block_count > 0 || options.block_threshold > 0.0;
   if (split_enabled) {
      print_progress(image.height);
      if (options.verbose) {
         say("Out:\nBlock boundaries: ");
         if (stats.block_ends.size() < 2) {
            say("none");
         } else {
            for (std::size_t i = 0; i + 1 < stats.block_ends.size(); ++i) {
               if (i != 0) say(",");
               say("%x", static_cast<unsigned>(stats.block_ends[i]));
            }
         }
         say(" (%d block", static_cast<int>(stats.block_ends.size()));
         if (stats.block_ends.size() > 1) say("s");
         say(")\n\n");
      }
   } else if (options.verbose) {
      say("\n");
   }

   const std::size_t overhead = stats.png_overhead;
   for (std::size_t i = 0; i < stats.pass_sizes.size(); ++i) {
      print_progress(image.height);
      const std::size_t total = overhead + stats.pass_sizes[i];
      print_out_line(total, stats.pass_deltas[i], options.verbose);
   }

   // The stored fallback emits a final size line after the optimisation passes.
   if (stats.stored) {
      if (!options.verbose) say("\r");
      say("Out:%8d bytes (stored)", static_cast<int>(stats.output_size));
      if (options.verbose) say("\n");
   }

   if (options.verbose) {
      say("Took %.3f sec.\n", elapsed);
   }
   say("\n");

   // Without -force the reference refuses to replace a file when the result is
   // not smaller. For a separate output it copies the original bytes instead.
   if (!force_write && is_png && stats.output_size >= input.size()) {
      if (!in_place && output_path != input_path && !input.empty()) {
         if (!write_file(output_path.c_str(), input)) {
            say("ERROR: Unable to open %s for writing!\n", output_path.c_str());
            return 1;
         }
         say("Unable to compress further: copying original file\n");
         return 2;
      }
      say("Unable to compress further\n");
      return 2;
   }

   if (!write_file(output_path.c_str(), output)) {
      say("ERROR: Unable to open %s for writing!\n", output_path.c_str());
      return 1;
   }

   if (!input.empty()) {
      const long long change = static_cast<long long>(stats.output_size) -
                               static_cast<long long>(input.size());
      const long long magnitude = change < 0 ? -change : change;
      const char plural = magnitude == 1 ? ' ' : 's';
      const int percent = static_cast<int>(
         (static_cast<double>(stats.output_size) * 100.0) /
         static_cast<double>(input.size()));
      say("Chg:%+8d byte%c (%3d%% of original)\n", static_cast<int>(change),
          plural, percent);
   }

   return 0;
}
