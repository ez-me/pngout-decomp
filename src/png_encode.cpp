// PNG encoder reconstructed from the reference binary at 0x407900.
//
// The target normalises every input image into a 32-bit 0xAARRGGBB pixel
// array, derives the output colour type and bit depth from a small set of
// whole-image flags, builds an internal palette (or a grayscale ramp), filters
// the samples row by row and finally runs the KZIP strategy-0 optimisation
// loop over the filtered stream.
//
// Palette output is special: while the optimisation loop converges it also
// permutes the palette (and every block's literal/length table) so that the
// most frequent palette entries receive the shortest codes. The permutation is
// only performed for -f0 and when the image actually owns its palette.
#include "png_image.hpp"

#include "block_splitter.hpp"
#include "strategy0_encoder.hpp"
#include "strategy12_parser.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace pngout {
namespace {

using kzip::kTokenSlots;
using LengthTable = std::array<std::uint8_t, kTokenSlots>;

constexpr std::uint32_t kOpaque = 0xff000000u;
constexpr std::uint32_t kOpaqueMax = 0xfeffffffu;

// The target hashes the 32-bit ARGB value with a 2048-entry open hash table
// (`(v >> 22 ^ v >> 11) - v & 0x7ff`). Only the low 11 bits survive, so an
// unsigned shift is equivalent to the signed shifts in the binary.
std::uint32_t palette_hash(std::uint32_t value) {
   return (((value >> 22) ^ (value >> 11)) - value) & 0x7ffu;
}

// A pixel is grayscale when R, G and B are equal; alpha is ignored.
bool is_grayscale(std::uint32_t value) {
   return ((value << 8 ^ value) & 0x00ffff00u) == 0;
}

std::uint32_t gray_argb(std::uint8_t gray) {
   return kOpaque | (static_cast<std::uint32_t>(gray) * 0x00010101u);
}

// Name the reference prints when suggesting a different output colour type.
// The index is a PNG colour type (0, 2, 3, 4 or 6).
const char* target_name(int color_type) {
   switch (color_type) {
   case 0:
      return "/c0 (Gray)";
   case 2:
      return "/c2 (RGB)";
   case 3:
      return "/c3 (Pal)";
   case 4:
      return "/c4 (Gray+Alpha)";
   case 6:
      return "/c6 (RGB+Alpha)";
   default:
      return "/c0 (Gray)";
   }
}

// Four-character chunk type as a big-endian 32-bit value.
constexpr std::uint32_t type_of(const char (&name)[5]) {
   return (static_cast<std::uint32_t>(static_cast<std::uint8_t>(name[0])) << 24) |
          (static_cast<std::uint32_t>(static_cast<std::uint8_t>(name[1])) << 16) |
          (static_cast<std::uint32_t>(static_cast<std::uint8_t>(name[2])) << 8) |
          static_cast<std::uint32_t>(static_cast<std::uint8_t>(name[3]));
}

bool ascii_equal_ignore_case(const char* a, const char* b) {
   while (*a != '\0' && *b != '\0') {
      unsigned char ca = static_cast<unsigned char>(*a++);
      unsigned char cb = static_cast<unsigned char>(*b++);
      if (ca >= 'a' && ca <= 'z') ca = static_cast<unsigned char>(ca - 'a' + 'A');
      if (cb >= 'a' && cb <= 'z') cb = static_cast<unsigned char>(cb - 'a' + 'A');
      if (ca != cb) return false;
   }
   return *a == '\0' && *b == '\0';
}

std::vector<std::uint32_t> make_argb_pixels(const Image& image) {
   const std::size_t count =
      static_cast<std::size_t>(image.width) * image.height;
   std::vector<std::uint32_t> pixels(count);
   if (image.palette) {
      for (std::size_t i = 0; i < count; ++i) {
         const std::size_t index = image.samples[i];
         pixels[i] = index < image.palette_rgba.size()
                        ? image.palette_rgba[index]
                        : image.palette_fill;
      }
      return pixels;
   }

   const int channels = image.channels;
   for (std::size_t i = 0; i < count; ++i) {
      const std::uint8_t* sample = &image.samples[i * channels];
      switch (channels) {
      case 1:
         pixels[i] = gray_argb(sample[0]);
         break;
      case 2:
         pixels[i] = (static_cast<std::uint32_t>(sample[1]) << 24) |
                     (gray_argb(sample[0]) & 0x00ffffffu);
         break;
      case 3:
         pixels[i] = kOpaque | (static_cast<std::uint32_t>(sample[0]) << 16) |
                     (static_cast<std::uint32_t>(sample[1]) << 8) | sample[2];
         break;
      default:
         pixels[i] = (static_cast<std::uint32_t>(sample[3]) << 24) |
                     (static_cast<std::uint32_t>(sample[0]) << 16) |
                     (static_cast<std::uint32_t>(sample[1]) << 8) | sample[2];
         break;
      }
      if (image.has_transparent_key &&
          (pixels[i] & 0x00ffffffu) == image.transparent_key) {
         pixels[i] &= 0x00ffffffu;
      }
   }
   return pixels;
}

struct ColorInfo {
   int flags{0}; // bit0 alpha, bit1 not grayscale, bit2 >256 colours
   std::size_t distinct{0};
   std::vector<std::uint32_t> palette; // ARGB, final (sorted) order
   std::vector<std::uint32_t> frequency;
};

// Shell sort the palette into descending frequency order. The target uses
// gaps n/2, n/4, ..., 1 and a strict `>` comparison, so equal frequencies keep
// their relative order.
void shell_sort_palette(std::vector<std::uint32_t>& palette,
                        std::vector<std::uint32_t>& frequency) {
   const std::size_t n = palette.size();
   for (std::size_t gap = n / 2; gap > 0; gap /= 2) {
      for (std::size_t i = gap; i < n; ++i) {
         const std::uint32_t value = palette[i];
         const std::uint32_t count = frequency[i];
         std::size_t j = i;
         while (j >= gap && frequency[j - gap] < count) {
            palette[j] = palette[j - gap];
            frequency[j] = frequency[j - gap];
            j -= gap;
         }
         palette[j] = value;
         frequency[j] = count;
      }
   }
}

// Move fully transparent entries in front of the opaque ones. This is a
// separate pass from the frequency sort and runs for both truecolor-derived
// and palette-input palettes. The swap also carries the frequency entries
// because the optimisation loop uses them indirectly through the palette
// order.
void bubble_transparent_entries(std::vector<std::uint32_t>& palette,
                                std::vector<std::uint32_t>& frequency) {
   const std::size_t n = palette.size();
   for (std::size_t i = 1; i < n; ++i) {
      for (std::size_t j = 0; j < i; ++j) {
         if (palette[j] > kOpaqueMax && palette[i] < kOpaque) {
            std::swap(palette[j], palette[i]);
            std::swap(frequency[j], frequency[i]);
         }
      }
   }
}

// Decide whether the image needs a real alpha channel. The target works from
// the smallest ARGB value rather than from any single non-opaque pixel:
//   min >= 0xff000000        every pixel is opaque
//   0x01000000 <= min        some pixel is only partially transparent
//   min <  0x01000000        fully transparent pixels exist; they form a usable
//                            colour key unless one of them has a different
//                            colour or an opaque pixel shares the key's RGB
// This lets a palette or tRNS colour key survive without an alpha channel.
int alpha_flag(const std::vector<std::uint32_t>& pixels) {
   if (pixels.empty()) return 0;
   std::uint32_t minimum = pixels[0];
   for (const std::uint32_t value : pixels) {
      if (value < minimum) minimum = value;
   }
   if (minimum > 0xfeffffffu) return 0;
   if (minimum > 0x00ffffffu) return 1;
   for (const std::uint32_t value : pixels) {
      if ((value & 0x00ffffffu) == minimum &&
          (value & 0xff000000u) != 0) {
         return 1;
      }
      if (value != minimum && value < 0xff000000u) return 1;
   }
   return 0;
}

// Truecolor input: palette entries appear in first-appearance order, are
// frequency sorted, then the transparent bubble runs.
ColorInfo analyse_truecolor(const std::vector<std::uint32_t>& pixels) {
   ColorInfo info;
   std::array<int, 2048> heads;
   heads.fill(-1);
   std::vector<int> chain;
   std::vector<std::uint32_t> palette;
   std::vector<std::uint32_t> frequency;

   info.flags |= alpha_flag(pixels);
   for (const std::uint32_t value : pixels) {
      if ((info.flags & 2) == 0 && !is_grayscale(value)) info.flags |= 2;
      if ((info.flags & 4) == 0) {
         const std::uint32_t hash = palette_hash(value);
         int index = heads[hash];
         while (index >= 0 && palette[index] != value) index = chain[index];
         if (index < 0) {
            if (palette.size() < 256) {
               chain.push_back(heads[hash]);
               heads[hash] = static_cast<int>(palette.size());
               palette.push_back(value);
               frequency.push_back(0);
            } else {
               info.flags |= 4;
            }
         }
      }
   }

   // The target rebuilds the palette and counts frequencies in a second pass,
   // which preserves the first-appearance order established above.
   for (const std::uint32_t value : pixels) {
      for (std::size_t i = 0; i < palette.size(); ++i) {
         if (palette[i] == value) {
            ++frequency[i];
            break;
         }
      }
   }

   info.distinct = palette.size();
   shell_sort_palette(palette, frequency);
   bubble_transparent_entries(palette, frequency);
   info.palette = std::move(palette);
   info.frequency = std::move(frequency);
   return info;
}

// Palette input: keep the used entries in their original PLTE order, compact
// out unused slots, and never frequency sort.
ColorInfo analyse_palette_input(const Image& image,
                                const std::vector<std::uint32_t>& pixels) {
   ColorInfo info;
   const std::size_t entries = image.palette_rgba.size();
   std::vector<std::uint32_t> frequency(entries, 0);

   // Duplicate ARGB entries resolve to their lowest index, matching the
   // reverse-built hash chain in the target.
   std::vector<int> first(entries, -1);
   std::array<int, 2048> heads;
   heads.fill(-1);
   for (std::size_t i = 0; i < entries; ++i) {
      const std::uint32_t hash = palette_hash(image.palette_rgba[i]);
      first[i] = heads[hash];
      heads[hash] = static_cast<int>(i);
   }
   auto find_index = [&](std::uint32_t value) -> int {
      int index = heads[palette_hash(value)];
      while (index >= 0) {
         if (image.palette_rgba[index] == value) return index;
         index = first[index];
      }
      return -1;
   };
   bool missing = false;
   for (const std::uint32_t value : pixels) {
      const int index = find_index(value);
      if (index >= 0) {
         ++frequency[index];
      } else {
         missing = true;
      }
   }

   // The reference stores decoded pixels as ARGB values and looks them up in
   // the palette table. A value that is absent (an out-of-range index filled
   // with the decoder's padding colour) makes it abandon the palette path and
   // rebuild the palette from first appearance, exactly like truecolor input.
   if (missing) return analyse_truecolor(pixels);

   std::vector<std::uint32_t> palette;
   std::vector<std::uint32_t> kept;
   for (std::size_t i = 0; i < entries; ++i) {
      if (frequency[i] != 0) {
         palette.push_back(image.palette_rgba[i]);
         kept.push_back(frequency[i]);
      }
   }

   for (const std::uint32_t value : pixels) {
      if ((info.flags & 2) == 0 && !is_grayscale(value)) info.flags |= 2;
   }
   info.flags |= alpha_flag(pixels);
   if (palette.size() > 256) info.flags |= 4;
   info.distinct = palette.size();
   bubble_transparent_entries(palette, kept);
   info.palette = std::move(palette);
   info.frequency = std::move(kept);
   return info;
}

// Grayscale input (color type 0): the palette is the set of used gray levels
// in ascending order. There is no frequency sort; a tRNS colour key still
// produces a fully transparent entry, which the shared transparency pass then
// bubbles to the front.
ColorInfo analyse_grayscale_input(const std::vector<std::uint32_t>& pixels) {
   ColorInfo info;
   std::array<std::uint32_t, 256> frequency{};
   std::array<std::uint32_t, 256> value{};
   std::array<bool, 256> used{};
   for (const std::uint32_t pixel : pixels) {
      const std::uint32_t gray = (pixel >> 16) & 0xffu;
      if (!used[gray]) {
         used[gray] = true;
         value[gray] = pixel;
      }
      ++frequency[gray];
      if ((pixel >> 24) != 0xff) info.flags |= 1;
   }
   for (std::uint32_t v = 0; v < 256; ++v) {
      if (used[v]) {
         info.palette.push_back(value[v]);
         info.frequency.push_back(frequency[v]);
      }
   }
   info.distinct = info.palette.size();
   bubble_transparent_entries(info.palette, info.frequency);
   return info;
}

int palette_bit_depth(std::size_t count) {
   if (count <= 2) return 1;
   if (count <= 4) return 2;
   if (count <= 16) return 4;
   return 8;
}

// The target honours an explicit depth only when it is at least the minimum
// the image needs; otherwise it fails. Any other requested depth (including
// the automatic 0) falls back to the minimum.
bool apply_requested_depth(int requested, int minimum, int& depth) {
   depth = minimum;
   if (requested == 1 || requested == 2 || requested == 4 || requested == 8) {
      if (requested < minimum) return false;
      depth = requested;
   }
   return true;
}

int grayscale_bit_depth(const std::vector<std::uint8_t>& used) {
   for (const int bits : {1, 2, 4, 8}) {
      const int step = 255 / ((1 << bits) - 1);
      bool fits = true;
      for (std::uint8_t value : used) {
         if (value % step != 0) {
            fits = false;
            break;
         }
      }
      if (fits) return bits;
   }
   return 8;
}

int grayscale_step(int bits) {
   switch (bits) {
   case 1:
      return 0xff;
   case 2:
      return 0x55;
   case 4:
      return 0x11;
   default:
      return 0x01;
   }
}

// Pack one sample row at the requested bit depth. 8-bit rows are unchanged;
// lower depths pack most-significant sample first.
void pack_row(const std::vector<std::uint8_t>& samples, int bits,
              std::vector<std::uint8_t>& out) {
   if (bits == 8) {
      out = samples;
      return;
   }
   const std::size_t per_byte = static_cast<std::size_t>(8 / bits);
   out.assign((samples.size() + per_byte - 1) / per_byte, 0);
   const int mask = (1 << bits) - 1;
   for (std::size_t x = 0; x < samples.size(); ++x) {
      const int shift =
         8 - bits * (static_cast<int>(x % per_byte) + 1);
      out[x / per_byte] |= static_cast<std::uint8_t>(
         (samples[x] & mask) << shift);
   }
}

std::vector<std::uint8_t> filter_row(const std::uint8_t* row,
                                     const std::uint8_t* previous,
                                     std::size_t count, std::size_t bpp,
                                     int filter, int y) {
   std::vector<std::uint8_t> out(count);
   for (std::size_t x = 0; x < count; ++x) {
      const int a = row[x];
      const int b = x >= bpp ? row[x - bpp] : 0;
      const int c = previous[x];
      const int d = x >= bpp ? previous[x - bpp] : 0;
      int value = 0;
      switch (filter) {
      case 0:
         value = a;
         break;
      case 1:
         value = a - b;
         break;
      case 2:
         value = a - c;
         break;
      case 3:
         value = y == 0 ? a - b : a - ((b + c) >> 1);
         break;
      default: {
         const int p = b + c - d;
         const int pa = std::abs(p - b);
         const int pb = std::abs(p - c);
         const int pc = std::abs(p - d);
         const int predictor =
            (pa <= pb && pa <= pc) ? b : (pb <= pc ? c : d);
         value = a - predictor;
         break;
      }
      }
      out[x] = static_cast<std::uint8_t>(value);
   }
   return out;
}

// Entropy estimate used by -f5. The target builds a 256-bin histogram of the
// filtered row (excluding the filter byte) and scores it as
//   0x7ffffffe - sum_bins (int)(hist[b] * log(hist[b]) * d)
// with d = 2147483647 / (N * log(N)). The smallest cost wins; ties keep the
// earlier filter because the comparison is strict.
int entropy_cost(const std::vector<std::uint8_t>& row) {
   const int n = static_cast<int>(row.size());
   if (n <= 1) return 0x7ffffffe;
   int histogram[256] = {};
   for (const std::uint8_t value : row) ++histogram[value];
   const double scale =
      2147483647.0 / (static_cast<double>(n) * std::log(static_cast<double>(n)));
   std::vector<int> lut(static_cast<std::size_t>(n) + 1, 0);
   for (int k = 1; k <= n; ++k) {
      lut[static_cast<std::size_t>(k)] =
         static_cast<int>(static_cast<double>(k) *
                          std::log(static_cast<double>(k)) * scale);
   }
   long long sum = 0;
   for (int b = 0; b < 256; ++b) sum += lut[static_cast<std::size_t>(histogram[b])];
   return static_cast<int>(0x7ffffffe - sum);
}

// Build the filtered byte stream. `rows` holds the unfiltered, already packed
// sample rows. `source_filters` is only consulted for -f6.
std::vector<std::uint8_t> build_filtered_stream(
   const std::vector<std::vector<std::uint8_t>>& rows, int filter,
   const std::vector<std::uint8_t>& source_filters, std::size_t bpp) {
   std::vector<std::uint8_t> stream;
   std::size_t total = 0;
   for (const auto& row : rows) total += row.size() + 1;
   stream.reserve(total);
   std::vector<std::uint8_t> zeros(rows.empty() ? 0 : rows.front().size(), 0);
   for (std::size_t y = 0; y < rows.size(); ++y) {
      const std::vector<std::uint8_t>& row = rows[y];
      const std::uint8_t* previous = y == 0 ? zeros.data() : rows[y - 1].data();
      int chosen = filter;
      if (filter == 6) {
         chosen = y < source_filters.size() ? source_filters[y] : 0;
      } else if (filter == 5) {
         int best = 0;
         int best_cost = std::numeric_limits<int>::max();
         for (int candidate = 0; candidate < 5; ++candidate) {
            const std::vector<std::uint8_t> filtered =
               filter_row(row.data(), previous, row.size(), bpp, candidate,
                          static_cast<int>(y));
            const int cost = entropy_cost(filtered);
            if (cost < best_cost) {
               best_cost = cost;
               best = candidate;
            }
         }
         chosen = best;
      }
      int emitted = chosen;
      if (chosen == 3 && y == 0) emitted = 1;
      stream.push_back(static_cast<std::uint8_t>(emitted));
      if (chosen == 0) {
         stream.insert(stream.end(), row.begin(), row.end());
      } else {
         const std::vector<std::uint8_t> filtered =
            filter_row(row.data(), previous, row.size(), bpp, chosen,
                       static_cast<int>(y));
         stream.insert(stream.end(), filtered.begin(), filtered.end());
      }
   }
   return stream;
}

std::uint32_t adler32(std::span<const std::uint8_t> data) {
   std::uint32_t a = 1;
   std::uint32_t b = 0;
   for (const std::uint8_t value : data) {
      a = (a + value) % 65521;
      b = (b + a) % 65521;
   }
   return (b << 16) | a;
}

void append_be32(std::vector<std::uint8_t>& out, std::uint32_t value) {
   out.push_back(static_cast<std::uint8_t>(value >> 24));
   out.push_back(static_cast<std::uint8_t>(value >> 16));
   out.push_back(static_cast<std::uint8_t>(value >> 8));
   out.push_back(static_cast<std::uint8_t>(value));
}

std::uint32_t crc32(std::span<const std::uint8_t> data) {
   std::uint32_t crc = 0xffffffffu;
   for (const std::uint8_t value : data) {
      crc ^= value;
      for (int bit = 0; bit < 8; ++bit) {
         crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
      }
   }
   return ~crc;
}

void append_chunk(std::vector<std::uint8_t>& out, const char type[5],
                  std::span<const std::uint8_t> data) {
   append_be32(out, static_cast<std::uint32_t>(data.size()));
   const std::size_t start = out.size();
   out.insert(out.end(), type, type + 4);
   out.insert(out.end(), data.begin(), data.end());
   const std::uint32_t checksum =
      crc32(std::span<const std::uint8_t>(out.data() + start,
                                          out.size() - start));
   append_be32(out, checksum);
}

// Copy a chunk exactly as it appeared in the input, including its stored CRC.
void append_raw_chunk(std::vector<std::uint8_t>& out, const PngChunk& chunk) {
   append_be32(out, static_cast<std::uint32_t>(chunk.data.size()));
   append_be32(out, chunk.type);
   out.insert(out.end(), chunk.data.begin(), chunk.data.end());
   append_be32(out, chunk.crc);
}

// Re-emit a preserved chunk. sBIT describes the significance of the source
// samples, so the target rewrites it for the output colour type instead of
// copying it verbatim. The internal channel values are:
//   * grayscale and gray+alpha input: every channel takes the first byte (the
//     alpha significance of a gray+alpha source is deliberately ignored);
//   * RGB and palette input: the three colour bytes, with alpha mirroring red;
//   * RGBA input: all four bytes.
// Output then selects the channels that its colour type can carry.
void append_kept_chunk(std::vector<std::uint8_t>& out, const PngChunk& chunk,
                       const Image& image, int color_type) {
   if (chunk.type == type_of("sBIT")) {
      const std::vector<std::uint8_t>& data = chunk.data;
      const auto at = [&](std::size_t index) -> std::uint8_t {
         return index < data.size() ? data[index] : 0;
      };
      std::uint8_t r = 0;
      std::uint8_t g = 0;
      std::uint8_t b = 0;
      std::uint8_t a = 0;
      switch (image.source_color_type) {
      case 0:
      case 4:
         r = g = b = a = at(0);
         break;
      case 2:
      case 3:
         r = at(0);
         g = at(1);
         b = at(2);
         a = r;
         break;
      case 6:
         r = at(0);
         g = at(1);
         b = at(2);
         a = at(3);
         break;
      default:
         append_raw_chunk(out, chunk);
         return;
      }
      std::vector<std::uint8_t> rewritten;
      switch (color_type) {
      case 0:
         rewritten.push_back(r);
         break;
      case 2:
      case 3:
         rewritten = {r, g, b};
         break;
      case 4:
         rewritten = {r, a};
         break;
      case 6:
         rewritten = {r, g, b, a};
         break;
      default:
         append_raw_chunk(out, chunk);
         return;
      }
      append_chunk(out, "sBIT", rewritten);
      return;
   }
   append_raw_chunk(out, chunk);
}

// Build the IDAT payload for the reference's "stored" fallback: a zlib stream
// made of one uncompressed deflate block per image row (filter byte 0 followed
// by the packed samples). The last block sets BFINAL.
std::vector<std::uint8_t> build_stored_idat(
   const std::vector<std::vector<std::uint8_t>>& rows) {
   std::vector<std::uint8_t> idat;
   idat.push_back(0x78); // CMF: deflate, 32K window
   idat.push_back(0x01); // FLG: no dictionary, fastest
   std::vector<std::uint8_t> payload;
   std::size_t total = 0;
   for (const std::vector<std::uint8_t>& row : rows) total += row.size() + 1;
   payload.reserve(total);
   for (std::size_t y = 0; y < rows.size(); ++y) {
      const std::vector<std::uint8_t>& row = rows[y];
      const std::uint16_t length =
         static_cast<std::uint16_t>(row.size() + 1);
      const std::uint16_t inverse = static_cast<std::uint16_t>(~length);
      idat.push_back(y + 1 == rows.size() ? 0x01 : 0x00); // BFINAL, stored
      idat.push_back(static_cast<std::uint8_t>(length));
      idat.push_back(static_cast<std::uint8_t>(length >> 8));
      idat.push_back(static_cast<std::uint8_t>(inverse));
      idat.push_back(static_cast<std::uint8_t>(inverse >> 8));
      idat.push_back(0x00); // filter type 0
      idat.insert(idat.end(), row.begin(), row.end());
      payload.push_back(0x00);
      payload.insert(payload.end(), row.begin(), row.end());
   }
   append_be32(idat, adler32(payload));
   return idat;
}

std::uint32_t read_be32(const std::uint8_t* data) {
   return (static_cast<std::uint32_t>(data[0]) << 24) |
          (static_cast<std::uint32_t>(data[1]) << 16) |
          (static_cast<std::uint32_t>(data[2]) << 8) |
          static_cast<std::uint32_t>(data[3]);
}

// Swap the generated IDAT chunk of `png` for `idat`, leaving every other chunk
// untouched.
std::vector<std::uint8_t> replace_idat(const std::vector<std::uint8_t>& png,
                                       std::span<const std::uint8_t> idat) {
   std::size_t position = 8;
   while (position + 12 <= png.size()) {
      const std::uint32_t length = read_be32(png.data() + position);
      const std::uint32_t type = read_be32(png.data() + position + 4);
      if (type == type_of("IDAT")) {
         std::vector<std::uint8_t> result;
         result.insert(result.end(), png.begin(),
                       png.begin() + static_cast<std::ptrdiff_t>(position));
         append_chunk(result, "IDAT", idat);
         result.insert(result.end(),
                       png.begin() + static_cast<std::ptrdiff_t>(
                                         position + 12 + length),
                       png.end());
         return result;
      }
      position += 12 + length;
   }
   return png;
}

// Conservative size the reference expects an uncompressed PNG to occupy. When
// the compressed result is larger than this, the stored representation is
// guaranteed to be smaller and is used instead.
std::size_t stored_size_bound(const Image& image, int color_type,
                              int bit_depth, std::size_t palette_count,
                              std::span<const PngChunk* const> kept_chunks,
                              bool has_key) {
   std::size_t bound = 0x83f;
   for (const PngChunk* chunk : kept_chunks) {
      bound += 12 + chunk->data.size();
   }
   const std::size_t width = image.width;
   const std::size_t height = image.height;
   const std::size_t row_bytes = (width * static_cast<std::size_t>(bit_depth) + 7) / 8;
   switch (color_type) {
   case 0:
      bound += (row_bytes + 6) * height;
      if (has_key) bound += 0xe;
      break;
   case 2:
      bound += (width * 3 + 6) * height;
      if (has_key) bound += 0x12;
      break;
   case 3:
      bound += (row_bytes + 6) * height + 0x18 + palette_count * 4;
      break;
   case 4:
      bound += (width * 2 + 6) * height;
      break;
   case 6:
      bound += (width * 4 + 6) * height;
      break;
   default:
      break;
   }
   return bound;
}

// Apply the stored fallback when the compressed PNG exceeds the conservative
// bound. Returns true when the output was replaced.
bool apply_stored_fallback(const Image& image, int color_type, int bit_depth,
                           std::size_t palette_count,
                           std::span<const PngChunk* const> kept_chunks,
                           bool has_key,
                           const std::vector<std::vector<std::uint8_t>>& rows,
                           std::vector<std::uint8_t>& png,
                           EncodeStats& stats) {
   const std::size_t bound = stored_size_bound(
      image, color_type, bit_depth, palette_count, kept_chunks, has_key);
   if (png.size() <= bound) return false;
   const std::vector<std::uint8_t> stored = build_stored_idat(rows);
   png = replace_idat(png, stored);
   stats.stored = true;
   return true;
}

struct OptimiseResult {
   std::vector<std::uint8_t> deflate;
   std::vector<std::uint8_t> stream;
   std::vector<std::size_t> block_ends;
   std::vector<std::size_t> pass_sizes;
   std::vector<std::int64_t> pass_deltas;
};

// Palette state hooks used by the optimisation loop. The reference saves the
// palette that produced the best pass and restores it for the final re-emit
// pass, and folds the palette bytes into the convergence fingerprint for
// color type 3.
struct PaletteHooks {
   std::function<bool(std::vector<LengthTable>&)> reorder;
   std::function<void()> save_best;
   std::function<void()> restore_best;
   std::function<std::uint32_t(std::uint32_t)> hash;
};

// Fold the current palette into a convergence fingerprint. The target runs a
// second CRC pass over its 256-entry 0xAARRGGBB palette buffer, consuming bytes
// from the end; unused entries are zero at the start of the process.
std::uint32_t fold_palette_fingerprint(
   std::uint32_t crc, const std::vector<std::uint32_t>& palette) {
   static const std::array<std::uint32_t, 256> table = [] {
      std::array<std::uint32_t, 256> generated{};
      for (std::uint32_t i = 0; i < 256; ++i) {
         std::uint32_t value = i;
         for (int bit = 0; bit < 8; ++bit) {
            value = (value >> 1) ^ (0xedb88320u & (0u - (value & 1u)));
         }
         generated[i] = value;
      }
      return generated;
   }();
   for (std::size_t k = 1024; k-- > 0;) {
      const std::size_t entry = k / 4;
      const unsigned shift = static_cast<unsigned>((k % 4) * 8);
      const std::uint32_t value = entry < palette.size() ? palette[entry] : 0;
      const std::uint8_t byte = static_cast<std::uint8_t>(value >> shift);
      crc = (crc >> 8) ^ table[(crc ^ byte) & 0xffu];
   }
   return crc;
}

// Run the PNGOUT optimisation loop. `build` recreates the filtered stream
// (palette reordering changes it) and `palette` permutes the palette from the
// previous pass's literal/length table.
//
// The loop mirrors the target's post-pass decision tree:
//   * a pass that improved the Huffman code lengths is recorded together with
//     a fingerprint of its start tables; if the same (fingerprint, size) state
//     appears twice the loop has cycled and stops;
//   * a pass that did not improve but produced a smaller stream stops;
//   * a non-improving pass that grew the stream restores the best start tables
//     (and palette) and re-emits once before stopping.
OptimiseResult optimise_stream(
   const std::function<std::vector<std::uint8_t>()>& build,
   PaletteHooks* palette, const EncodeOptions& options) {
   kzip::SplitOptions split;
   split.threshold = options.block_threshold;
   split.block_count = options.block_count;

   kzip::EncoderOptions encoder;
   encoder.strategy = options.strategy;

   OptimiseResult result;
   std::vector<std::uint8_t> stream = build();
   std::vector<std::size_t> ends = kzip::choose_block_ends(stream, split);
   kzip::Strategy12Parser parser(stream, encoder.strategy, encoder.run_cap,
                                 encoder.arena_base_limit);
   std::vector<LengthTable> tables(ends.size(), kzip::initial_code_lengths());

   // Best pass seen so far (start tables, stream and palette state).
   std::size_t best_size = std::numeric_limits<std::size_t>::max();
   std::vector<LengthTable> best_tables;
   std::vector<std::uint8_t> best_stream;
   std::vector<std::size_t> best_ends;

   std::vector<std::uint32_t> fingerprints;
   std::vector<std::size_t> recorded_sizes;
   bool stop_flag = false;
   // Our first pass corresponds to the target's second pass (the target's
   // first pass is the single-block throwaway pass whose progress line is
   // printed by the CLI), so the attempt index starts at one.
   std::size_t attempt = 1;

   for (;;) {
      kzip::EncoderPassResult pass =
         kzip::run_strategy0_pass(stream, ends, tables, encoder, &parser);
      result.pass_sizes.push_back(pass.size);
      result.pass_deltas.push_back(pass.delta);

      if (stop_flag) {
         // The restored re-emit pass always reports a plain size; the target
         // suppresses the delta once it has decided to stop.
         result.pass_deltas.back() = 0;
         result.deflate = std::move(pass.bytes);
         break;
      }

      if (pass.delta < 1) {
         if (pass.size <= best_size) {
            result.deflate = std::move(pass.bytes);
            break;
         }
         // The pass grew the stream: re-emit with the best tables/palette.
         tables = best_tables;
         stream = best_stream;
         ends = best_ends;
         if (palette != nullptr && palette->restore_best) {
            palette->restore_best();
         }
         stop_flag = true;
         continue;
      }

      // Improving pass: the target stops once the 1,024-entry state table is
      // full.
      if (attempt > 1023) {
         result.deflate = std::move(pass.bytes);
         break;
      }
      if (recorded_sizes.size() <= attempt) {
         recorded_sizes.resize(attempt + 1, 0);
         fingerprints.resize(attempt + 1, 0);
      }
      recorded_sizes[attempt] = pass.size;
      if (pass.size < best_size) {
         best_size = pass.size;
         best_tables = tables;
         best_stream = stream;
         best_ends = ends;
         if (palette != nullptr && palette->save_best) palette->save_best();
      }

      std::uint32_t fingerprint = kzip::length_fingerprint(tables);
      if (palette != nullptr && palette->hash) {
         fingerprint = palette->hash(fingerprint);
      }
      bool repeated = false;
      for (std::size_t j = attempt; j-- > 0;) {
         if (fingerprints[j] == fingerprint &&
             recorded_sizes[j] == pass.size) {
            repeated = true;
            break;
         }
      }
      fingerprints[attempt] = fingerprint;
      if (repeated) {
         if (pass.size <= best_size) {
            result.deflate = std::move(pass.bytes);
            break;
         }
         tables = best_tables;
         stream = best_stream;
         ends = best_ends;
         if (palette != nullptr && palette->restore_best) {
            palette->restore_best();
         }
         stop_flag = true;
         continue;
      }

      // Prepare the next pass: permute the palette using the new tables and
      // rebuild the filtered stream.
      std::vector<LengthTable> next = pass.new_lengths;
      if (palette != nullptr && palette->reorder) palette->reorder(next);
      stream = build();
      ends = kzip::choose_block_ends(stream, split);
      parser.set_input(stream);

      // The target keeps the block boundaries fixed between attempts. The
      // palette permutation does not change the stream length, so the same
      // boundaries remain valid; when the splitter nevertheless returns a
      // different count, reuse the closest available table.
      if (next.size() != ends.size()) {
         std::vector<LengthTable> adjusted(ends.size(),
                                           kzip::initial_code_lengths());
         for (std::size_t i = 0; i < ends.size(); ++i) {
            adjusted[i] = next[i < next.size() ? i : next.size() - 1];
         }
         next.swap(adjusted);
      }
      tables = std::move(next);
      ++attempt;
   }

   result.stream = stream;
   result.block_ends = ends;
   return result;
}

} // namespace

bool chunk_is_required(std::uint32_t type, int requested_color_type) {
   if (type == type_of("IHDR") || type == type_of("PLTE") ||
       type == type_of("IDAT") || type == type_of("IEND")) {
      return true;
   }
   if (type == type_of("tRNS")) {
      // Colour-key transparency is regenerated for the colour types that can
      // carry it; an explicit gray+alpha or RGBA target stores the same
      // information in its alpha samples instead.
      return requested_color_type != 4 && requested_color_type != 6;
   }
   return false;
}

bool chunk_is_kept(std::uint32_t type, const EncodeOptions& options) {
   const std::string& spec = options.keep_chunk_list;
   if (spec.empty()) return false;
   if (spec == "1") return true;
   if (ascii_equal_ignore_case(spec.c_str(), "ALL")) return true;
   std::size_t position = 0;
   while (position <= spec.size()) {
      const std::size_t comma = spec.find(',', position);
      const std::size_t length =
         (comma == std::string::npos ? spec.size() : comma) - position;
      if (length == 4) {
         const char* name = spec.data() + position;
         const std::uint32_t candidate =
            (static_cast<std::uint32_t>(
                static_cast<std::uint8_t>(name[0])) << 24) |
            (static_cast<std::uint32_t>(
                static_cast<std::uint8_t>(name[1])) << 16) |
            (static_cast<std::uint32_t>(
                static_cast<std::uint8_t>(name[2])) << 8) |
            static_cast<std::uint32_t>(static_cast<std::uint8_t>(name[3]));
         if (candidate == type) return true;
      }
      if (comma == std::string::npos) break;
      position = comma + 1;
   }
   return false;
}

std::uint32_t input_bkgd_color(const Image& image) {
   for (const PngChunk& chunk : image.chunks) {
      if (chunk.type != type_of("bKGD")) continue;
      const std::vector<std::uint8_t>& data = chunk.data;
      if (image.source_color_type == 3) {
         if (data.empty() || data[0] >= image.palette_rgba.size()) return 0;
         return image.palette_rgba[data[0]] & 0x00ffffffu;
      }
      if (image.source_color_type == 0 || image.source_color_type == 4) {
         if (data.size() < 2) return 0;
         const int levels = (1 << image.bit_depth) - 1;
         const int raw = (data[0] << 8) | data[1];
         const std::uint32_t gray =
            static_cast<std::uint32_t>((raw * 0xff) / levels);
         return gray * 0x00010101u;
      }
      if (image.bit_depth == 8) {
         if (data.size() < 6) return 0;
         return (static_cast<std::uint32_t>(data[1]) << 16) |
                (static_cast<std::uint32_t>(data[3]) << 8) |
                static_cast<std::uint32_t>(data[5]);
      }
      if (data.size() < 6) return 0;
      std::uint32_t value = 0;
      for (int i = 0; i < 3; ++i) {
         const std::uint32_t sample =
            (static_cast<std::uint32_t>(data[i * 2]) << 8) | data[i * 2 + 1];
         value = (value << 8) | (sample / 255u);
      }
      return value;
   }
   return 0;
}

bool encode_png(const Image& image, const EncodeOptions& options,
                std::vector<std::uint8_t>& output, EncodeStats& stats,
                std::string& error) {
   if (image.width == 0 || image.height == 0) {
      error = "empty image";
      return false;
   }

   const std::vector<std::uint32_t> pixels = make_argb_pixels(image);
   // An automatically selected palette output reuses the input PNG's palette
   // order (compacted to the used entries). Forcing -c3 goes through the
   // generic truecolor conversion instead, which sorts the palette by
   // descending frequency.
   ColorInfo colors;
   if (options.color_type >= 0) {
      colors = analyse_truecolor(pixels);
   } else if (image.palette) {
      colors = analyse_palette_input(image, pixels);
   } else if (image.channels == 1 && !image.filters.empty()) {
      // Only a PNG grayscale input keeps the ascending ramp palette. Grayscale
      // samples from other decoders (TGA, JPEG) go through the truecolor
      // frequency sort, exactly like the reference.
      colors = analyse_grayscale_input(pixels);
   } else {
      colors = analyse_truecolor(pixels);
   }

   int auto_type = 3;
   switch (colors.flags) {
   case 0:
      auto_type = colors.distinct <= 0x7f ? 3 : 0;
      break;
   case 1:
   case 2:
   case 3:
      auto_type = 3;
      break;
   case 5:
      auto_type = 4;
      break;
   case 6:
      auto_type = 2;
      break;
   case 7:
      auto_type = 6;
      break;
   default:
      auto_type = 3;
      break;
   }

   const int color_type =
      options.color_type >= 0 ? options.color_type : auto_type;

   EncodeOptions resolved = options;
   if (resolved.filter < 0) resolved.filter = color_type == 3 ? 0 : 5;
   // -f6 asks for the input's own filter bytes. Formats other than PNG have
   // none, and the reference falls back to its adaptive choice in that case.
   if (resolved.filter == 6 && image.filters.empty()) resolved.filter = 5;

   // Split the input's preservable chunks by where they appeared relative to
   // the generated PLTE and IDAT chunks. The reference writes them back in
   // those same three groups.
   std::vector<const PngChunk*> kept_before_plte;
   std::vector<const PngChunk*> kept_after_plte;
   std::vector<const PngChunk*> kept_after_idat;
   const PngChunk* kept_bkgd = nullptr;
   int region = 1;
   for (const PngChunk& chunk : image.chunks) {
      if (chunk.type == type_of("IHDR")) {
         region = 1;
         continue;
      }
      if (chunk.type == type_of("PLTE")) {
         region = 2;
         continue;
      }
      if (chunk.type == type_of("IDAT")) {
         region = 3;
         continue;
      }
      if (chunk.type == type_of("IEND")) continue;
      if (chunk_is_required(chunk.type, options.color_type)) continue;
      if (!chunk_is_kept(chunk.type, options)) continue;
      if (chunk.type == type_of("bKGD")) {
         kept_bkgd = &chunk;
         continue;
      }
      if (region == 1) {
         kept_before_plte.push_back(&chunk);
      } else if (region == 2) {
         kept_after_plte.push_back(&chunk);
      } else {
         kept_after_idat.push_back(&chunk);
      }
   }
   const std::uint32_t bkgd_color =
      kept_bkgd != nullptr ? input_bkgd_color(image) : 0;
   const bool keep_bkgd = kept_bkgd != nullptr && bkgd_color != 0;

   // A tRNS colour key can be carried through color types 0 and 2 (the target
   // re-emits it as a tRNS chunk); real per-pixel alpha cannot. Palette input
   // expresses a fully transparent entry the same way: its pixels keep the
   // entry's RGB with a zero alpha, and the smallest such value becomes the
   // key when the alpha test above found no real transparency.
   std::uint32_t transparent_key = image.transparent_key;
   bool has_transparent_key = image.has_transparent_key;
   if (!has_transparent_key && image.palette && (colors.flags & 1) == 0) {
      std::uint32_t minimum = 0xffffffffu;
      for (const std::uint32_t value : pixels) {
         if (value < minimum) minimum = value;
      }
      if (minimum < 0x01000000u) {
         has_transparent_key = true;
         transparent_key = minimum & 0x00ffffffu;
      }
   }
   const bool key_alpha = has_transparent_key;

   // Colour type the reference would have chosen for this image. Its failure
   // messages point the user at this type, not at the requested one.
   const char* const suggested = target_name(auto_type);
   const bool needs_alpha = (colors.flags & 1) != 0 && !key_alpha;
   // A tRNS colour key does not force an alpha channel, so it must not add the
   // "requires full alpha" clause to the color-type-0 diagnostic.
   const int effective_flags =
      key_alpha ? (colors.flags & ~1) : colors.flags;

   switch (color_type) {
   case 0:
      if (effective_flags != 0) {
         std::string message = "Image ";
         if ((effective_flags & 1) != 0) message += "requires full alpha; ";
         if ((effective_flags & 2) != 0) message += "is not grayscale; ";
         if ((effective_flags & 4) != 0) {
            message += "doesn't fit into 256 color entries; ";
         }
         message += "try ";
         message += suggested;
         message += " instead";
         error = std::move(message);
         return false;
      }
      break;
   case 2:
      if (needs_alpha) {
         error = std::string("Image requires full alpha; try ") + suggested +
                 " instead";
         return false;
      }
      break;
   case 3:
      if ((colors.flags & 4) != 0) {
         error = std::string("Image doesn't fit into 256 color entries; try ") +
                 suggested + " instead";
         return false;
      }
      break;
   case 4:
      if ((colors.flags & 2) != 0) {
         error = std::string("Image is not grayscale; try ") + suggested +
                 " instead";
         return false;
      }
      break;
   case 6:
      break;
   default:
      error = "unsupported color type";
      return false;
   }

   int bit_depth = 8;
   std::vector<std::uint32_t> palette;
   std::vector<std::uint8_t> gray_levels;

   if (color_type == 3) {
      palette = colors.palette;
      if (palette.empty()) {
         error = "empty palette";
         return false;
      }
      // A preserved bKGD colour has to be addressable in the palette. The
      // reference appends it before choosing the bit depth.
      if (keep_bkgd) {
         bool found = false;
         for (const std::uint32_t value : palette) {
            if (((value ^ bkgd_color) & 0x00ffffffu) == 0) {
               found = true;
               break;
            }
         }
         if (!found) {
            const int limit =
               options.bit_depth >= 0 ? (1 << options.bit_depth) : 256;
            if (static_cast<int>(palette.size()) >= limit) {
               error =
                  "bKGD chunk can't fit palette index. Try higher bitdepth, "
                  "/c2 or /c6 instead";
               return false;
            }
            palette.push_back(bkgd_color | kOpaque);
         }
      }
      const int minimum = palette_bit_depth(palette.size());
      if (!apply_requested_depth(options.bit_depth, minimum, bit_depth)) {
         error = "Image doesn't fit in selected bitdepth";
         return false;
      }
   } else if (color_type == 0) {
      std::array<bool, 256> used{};
      for (const std::uint32_t value : pixels) {
         used[(value >> 16) & 0xff] = true;
      }
      for (int v = 0; v < 256; ++v) {
         if (used[v]) gray_levels.push_back(static_cast<std::uint8_t>(v));
      }
      const int minimum = grayscale_bit_depth(gray_levels);
      if (!apply_requested_depth(options.bit_depth, minimum, bit_depth)) {
         error = "Image doesn't fit in selected bitdepth";
         return false;
      }
   } else {
      bit_depth = 8;
      // The reference only honours -d1/-d2/-d4 for grayscale and palette
      // output; truecolor targets reject those requests outright.
      if (options.bit_depth == 1 || options.bit_depth == 2 ||
          options.bit_depth == 4) {
         error = "Image doesn't fit in selected bitdepth";
         return false;
      }
      if (options.bit_depth == 16) {
         error = "16-bit output is not supported yet";
         return false;
      }
   }

   // Build the sample value for every pixel.
   if (color_type == 3) {
      std::vector<int> position(palette.size(), -1);
      for (std::size_t i = 0; i < palette.size(); ++i) {
         position[i] = static_cast<int>(i);
      }
      std::vector<int> index_of(pixels.size(), 0);
      for (std::size_t p = 0; p < pixels.size(); ++p) {
         for (std::size_t i = 0; i < palette.size(); ++i) {
            if (palette[i] == pixels[p]) {
               index_of[p] = static_cast<int>(i);
               break;
            }
         }
      }
      // `orig_index` identifies the palette slot independent of later
      // permutations; `pos`/`inv` translate it to the current slot.
      std::vector<int> pos(palette.size());
      std::vector<int> inv(palette.size());
      for (std::size_t i = 0; i < palette.size(); ++i) {
         pos[i] = static_cast<int>(i);
         inv[i] = static_cast<int>(i);
      }
      std::vector<int> orig_index = index_of;

      auto make_rows = [&]() {
         std::vector<std::vector<std::uint8_t>> rows(image.height);
         for (std::uint32_t y = 0; y < image.height; ++y) {
            std::vector<std::uint8_t> samples(image.width);
            for (std::uint32_t x = 0; x < image.width; ++x) {
               const std::size_t p = static_cast<std::size_t>(y) * image.width + x;
               samples[x] = static_cast<std::uint8_t>(
                  pos[static_cast<std::size_t>(orig_index[p])]);
            }
            pack_row(samples, bit_depth, rows[y]);
         }
         return rows;
      };

      std::vector<std::vector<std::uint8_t>> rows = make_rows();
      auto build = [&]() {
         return build_filtered_stream(rows, resolved.filter, image.filters, 1);
      };
      auto reorder = [&](std::vector<LengthTable>& tables) {
         if (resolved.filter != 0 || bit_depth != 8) return false;
         if (palette.size() <= 2) return false;
         if (tables.empty()) return false;
         bool changed = false;
         const std::size_t count = palette.size();
         for (std::size_t i = 1; i + 1 < count; ++i) {
            for (std::size_t j = 1; j <= i; ++j) {
               const std::size_t a = j;
               const std::size_t b = i + 1;
               if (((palette[a] ^ palette[b]) & 0xff000000u) != 0) continue;
               if (tables.front()[a] <= tables.front()[b]) continue;
               for (auto& block : tables) {
                  std::swap(block[a], block[b]);
               }
               std::swap(palette[a], palette[b]);
               std::swap(inv[a], inv[b]);
               pos[static_cast<std::size_t>(inv[a])] = static_cast<int>(a);
               pos[static_cast<std::size_t>(inv[b])] = static_cast<int>(b);
               changed = true;
            }
         }
         if (changed) rows = make_rows();
         return changed;
      };

      // Snapshot the palette/row state that produced the best pass so the
      // final re-emit pass can reproduce it exactly.
      std::vector<std::uint32_t> saved_palette;
      std::vector<int> saved_pos;
      std::vector<int> saved_inv;
      std::vector<std::vector<std::uint8_t>> saved_rows;
      PaletteHooks hooks;
      hooks.reorder = reorder;
      hooks.save_best = [&]() {
         saved_palette = palette;
         saved_pos = pos;
         saved_inv = inv;
         saved_rows = rows;
      };
      hooks.restore_best = [&]() {
         if (saved_palette.empty()) return;
         palette = saved_palette;
         pos = saved_pos;
         inv = saved_inv;
         rows = saved_rows;
      };
      hooks.hash = [&](std::uint32_t crc) {
         return fold_palette_fingerprint(crc, palette);
      };

      OptimiseResult optimised = optimise_stream(build, &hooks, options);
      stats.color_type = 3;
      stats.bit_depth = bit_depth;
      stats.filter = resolved.filter;
      stats.palette_colors = palette.size();
      stats.block_ends = optimised.block_ends;
      stats.pass_sizes = optimised.pass_sizes;
      stats.pass_deltas = optimised.pass_deltas;

      std::vector<std::uint8_t> png;
      static constexpr std::uint8_t signature[8] = {
         0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a};
      png.insert(png.end(), signature, signature + 8);

      std::vector<std::uint8_t> ihdr;
      append_be32(ihdr, image.width);
      append_be32(ihdr, image.height);
      ihdr.push_back(static_cast<std::uint8_t>(bit_depth));
      ihdr.push_back(3);
      ihdr.push_back(0);
      ihdr.push_back(0);
      ihdr.push_back(0);
      append_chunk(png, "IHDR", ihdr);
      for (const PngChunk* chunk : kept_before_plte) {
         append_kept_chunk(png, *chunk, image, color_type);
      }

      std::vector<std::uint8_t> plte;
      plte.reserve(palette.size() * 3);
      for (const std::uint32_t value : palette) {
         plte.push_back(static_cast<std::uint8_t>(value >> 16));
         plte.push_back(static_cast<std::uint8_t>(value >> 8));
         plte.push_back(static_cast<std::uint8_t>(value));
      }
      append_chunk(png, "PLTE", plte);

      std::size_t last_transparent = 0;
      for (std::size_t i = 0; i < palette.size(); ++i) {
         if ((palette[i] >> 24) != 0xff) last_transparent = i + 1;
      }
      if (last_transparent > 0) {
         std::vector<std::uint8_t> trns(last_transparent);
         for (std::size_t i = 0; i < last_transparent; ++i) {
            trns[i] = static_cast<std::uint8_t>(palette[i] >> 24);
         }
         append_chunk(png, "tRNS", trns);
      }

      if (keep_bkgd) {
         std::uint8_t index = 0;
         for (std::size_t i = 0; i < palette.size(); ++i) {
            if (((palette[i] ^ bkgd_color) & 0x00ffffffu) == 0) {
               index = static_cast<std::uint8_t>(i);
               break;
            }
         }
         append_chunk(png, "bKGD",
                      std::span<const std::uint8_t>(&index, 1));
      }
      for (const PngChunk* chunk : kept_after_plte) {
         append_kept_chunk(png, *chunk, image, color_type);
      }

      std::vector<std::uint8_t> idat;
      idat.push_back(0x78);
      idat.push_back(0x5e);
      idat.insert(idat.end(), optimised.deflate.begin(),
                  optimised.deflate.end());
      append_be32(idat, adler32(optimised.stream));
      append_chunk(png, "IDAT", idat);
      for (const PngChunk* chunk : kept_after_idat) {
         append_kept_chunk(png, *chunk, image, color_type);
      }
      append_chunk(png, "IEND", {});
      stats.deflate_size = optimised.deflate.size();
      stats.png_overhead = png.size() - stats.deflate_size;
      stats.output_size = png.size();
      std::vector<const PngChunk*> kept;
      kept.insert(kept.end(), kept_before_plte.begin(),
                  kept_before_plte.end());
      kept.insert(kept.end(), kept_after_plte.begin(),
                  kept_after_plte.end());
      kept.insert(kept.end(), kept_after_idat.begin(),
                  kept_after_idat.end());
      if (keep_bkgd) kept.push_back(kept_bkgd);
      apply_stored_fallback(image, 3, bit_depth, palette.size(), kept, false,
                            rows, png, stats);
      stats.output_size = png.size();
      output = std::move(png);
      return true;
   }

   // Non-palette paths build byte rows directly from the ARGB pixels.
   int channels = 1;
   switch (color_type) {
   case 0:
      channels = 1;
      break;
   case 2:
      channels = 3;
      break;
   case 4:
      channels = 2;
      break;
   default:
      channels = 4;
      break;
   }
   const std::size_t bpp = static_cast<std::size_t>(channels);

   std::vector<std::vector<std::uint8_t>> rows(image.height);
   for (std::uint32_t y = 0; y < image.height; ++y) {
      std::vector<std::uint8_t> samples;
      if (color_type == 0) {
         samples.resize(image.width);
         for (std::uint32_t x = 0; x < image.width; ++x) {
            const std::size_t p = static_cast<std::size_t>(y) * image.width + x;
            const std::uint8_t gray =
               static_cast<std::uint8_t>(pixels[p] >> 16);
            samples[x] = static_cast<std::uint8_t>(gray / grayscale_step(bit_depth));
         }
         pack_row(samples, bit_depth, rows[y]);
      } else {
         samples.reserve(static_cast<std::size_t>(image.width) * channels);
         for (std::uint32_t x = 0; x < image.width; ++x) {
            const std::size_t p = static_cast<std::size_t>(y) * image.width + x;
            const std::uint32_t value = pixels[p];
            switch (color_type) {
            case 2:
               samples.push_back(static_cast<std::uint8_t>(value >> 16));
               samples.push_back(static_cast<std::uint8_t>(value >> 8));
               samples.push_back(static_cast<std::uint8_t>(value));
               break;
            case 4:
               samples.push_back(static_cast<std::uint8_t>(value >> 16));
               samples.push_back(static_cast<std::uint8_t>(value >> 24));
               break;
            default:
               samples.push_back(static_cast<std::uint8_t>(value >> 16));
               samples.push_back(static_cast<std::uint8_t>(value >> 8));
               samples.push_back(static_cast<std::uint8_t>(value));
               samples.push_back(static_cast<std::uint8_t>(value >> 24));
               break;
            }
         }
         rows[y] = std::move(samples);
      }
   }

   auto build = [&]() {
      return build_filtered_stream(rows, resolved.filter, image.filters, bpp);
   };
   OptimiseResult optimised = optimise_stream(build, nullptr, options);

   stats.color_type = color_type;
   stats.bit_depth = bit_depth;
   stats.filter = resolved.filter;
   stats.palette_colors = 0;
   stats.block_ends = optimised.block_ends;
   stats.pass_sizes = optimised.pass_sizes;
   stats.pass_deltas = optimised.pass_deltas;

   std::vector<std::uint8_t> png;
   static constexpr std::uint8_t signature[8] = {
      0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a};
   png.insert(png.end(), signature, signature + 8);

   std::vector<std::uint8_t> ihdr;
   append_be32(ihdr, image.width);
   append_be32(ihdr, image.height);
   ihdr.push_back(static_cast<std::uint8_t>(bit_depth));
   ihdr.push_back(static_cast<std::uint8_t>(color_type));
   ihdr.push_back(0);
   ihdr.push_back(0);
   ihdr.push_back(0);
   append_chunk(png, "IHDR", ihdr);
   for (const PngChunk* chunk : kept_before_plte) {
      append_kept_chunk(png, *chunk, image, color_type);
   }

   // A tRNS colour key survives into color types 0 and 2; alpha-capable types
   // store the same information in their alpha samples instead.
   if (has_transparent_key && (color_type == 0 || color_type == 2)) {
      std::vector<std::uint8_t> trns;
      if (color_type == 0) {
         const std::uint32_t gray = (transparent_key >> 16) & 0xffu;
         const std::uint32_t levels = (1u << bit_depth) - 1u;
         const std::uint32_t value = levels * gray / 255u;
         trns.push_back(0);
         trns.push_back(static_cast<std::uint8_t>(value));
      } else {
         for (int shift = 16; shift >= 0; shift -= 8) {
            trns.push_back(0);
            trns.push_back(static_cast<std::uint8_t>(
               (transparent_key >> shift) & 0xffu));
         }
      }
      append_chunk(png, "tRNS", trns);
   }

   if (keep_bkgd) {
      std::vector<std::uint8_t> bkgd;
      const std::uint32_t levels = (1u << bit_depth) - 1u;
      if (color_type == 0 || color_type == 4) {
         const std::uint32_t gray =
            levels * ((bkgd_color >> 16) & 0xffu) / 255u;
         bkgd.push_back(0);
         bkgd.push_back(static_cast<std::uint8_t>(gray));
      } else {
         for (int shift = 16; shift >= 0; shift -= 8) {
            const std::uint32_t sample =
               levels * ((bkgd_color >> shift) & 0xffu) / 255u;
            bkgd.push_back(0);
            bkgd.push_back(static_cast<std::uint8_t>(sample));
         }
      }
      append_chunk(png, "bKGD", bkgd);
   }
   for (const PngChunk* chunk : kept_after_plte) {
      append_kept_chunk(png, *chunk, image, color_type);
   }

   std::vector<std::uint8_t> idat;
   idat.push_back(0x78);
   idat.push_back(0x5e);
   idat.insert(idat.end(), optimised.deflate.begin(), optimised.deflate.end());
   append_be32(idat, adler32(optimised.stream));
   append_chunk(png, "IDAT", idat);
   for (const PngChunk* chunk : kept_after_idat) {
      append_kept_chunk(png, *chunk, image, color_type);
   }
   append_chunk(png, "IEND", {});
   stats.deflate_size = optimised.deflate.size();
   stats.png_overhead = png.size() - stats.deflate_size;
   stats.output_size = png.size();
   std::vector<const PngChunk*> kept;
   kept.insert(kept.end(), kept_before_plte.begin(), kept_before_plte.end());
   kept.insert(kept.end(), kept_after_plte.begin(), kept_after_plte.end());
   kept.insert(kept.end(), kept_after_idat.begin(), kept_after_idat.end());
   if (keep_bkgd) kept.push_back(kept_bkgd);
   const bool key_emitted =
      has_transparent_key && (color_type == 0 || color_type == 2);
   apply_stored_fallback(image, color_type, bit_depth, 0, kept, key_emitted,
                         rows, png, stats);
   stats.output_size = png.size();
   output = std::move(png);
   return true;
}

} // namespace pngout
