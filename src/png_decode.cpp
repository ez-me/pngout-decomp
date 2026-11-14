#include "png_image.hpp"
#include "inflate.hpp"

#include <array>
#include <cstring>

namespace pngout {
namespace {

std::uint32_t read_be32(const std::uint8_t* p) {
   return (static_cast<std::uint32_t>(p[0]) << 24) |
          (static_cast<std::uint32_t>(p[1]) << 16) |
          (static_cast<std::uint32_t>(p[2]) << 8) |
          static_cast<std::uint32_t>(p[3]);
}

std::uint32_t chunk_type(const char (&name)[5]) {
   return (static_cast<std::uint32_t>(static_cast<std::uint8_t>(name[0])) << 24) |
          (static_cast<std::uint32_t>(static_cast<std::uint8_t>(name[1])) << 16) |
          (static_cast<std::uint32_t>(static_cast<std::uint8_t>(name[2])) << 8) |
          static_cast<std::uint32_t>(static_cast<std::uint8_t>(name[3]));
}

bool valid_png_signature(const std::vector<std::uint8_t>& file) {
   static constexpr std::array<std::uint8_t, 8> kSignature = {
      0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a};
   return file.size() >= kSignature.size() &&
          std::memcmp(file.data(), kSignature.data(), kSignature.size()) == 0;
}

// Reverse the PNG per-row filters in place. `row` is `stride` bytes including
// the filter byte at index 0; the result overwrites `row` starting at index 0
// with the unfiltered samples.
void unfilter_row(std::uint8_t filter, std::uint8_t* row, std::size_t count,
                  const std::uint8_t* previous, std::size_t bpp) {
   for (std::size_t i = 0; i < count; ++i) {
      const int a = (i >= bpp) ? row[i - bpp] : 0;
      const int b = previous ? previous[i] : 0;
      const int c = (previous && i >= bpp) ? previous[i - bpp] : 0;
      int value = row[i];
      switch (filter) {
      case 0:
         break;
      case 1:
         value += a;
         break;
      case 2:
         value += b;
         break;
      case 3:
         value += (a + b) >> 1;
         break;
      case 4: {
         const int p = a + b - c;
         const int pa = p > a ? p - a : a - p;
         const int pb = p > b ? p - b : b - p;
         const int pc = p > c ? p - c : c - p;
         int predictor = 0;
         if (pa <= pb && pa <= pc) {
            predictor = a;
         } else if (pb <= pc) {
            predictor = b;
         } else {
            predictor = c;
         }
         value += predictor;
         break;
      }
      default:
         break;
      }
      row[i] = static_cast<std::uint8_t>(value);
   }
}

} // namespace

bool decode_png(const std::vector<std::uint8_t>& file, Image& image,
                std::string& error) {
   if (!valid_png_signature(file)) {
      error = "not a PNG file";
      return false;
   }

   std::vector<std::uint8_t> idat;
   std::vector<std::uint8_t> plte;
   std::vector<std::uint8_t> trns;
   std::uint32_t width = 0;
   std::uint32_t height = 0;
   int bit_depth = 8;
   int color_type = 0;
   int interlace = 0;
   bool saw_header = false;
   bool saw_end = false;

   std::size_t offset = 8;
   while (offset + 12 <= file.size()) {
      const std::uint32_t length = read_be32(&file[offset]);
      if (offset + 12 + static_cast<std::size_t>(length) > file.size()) {
         error = "truncated chunk";
         return false;
      }
      const std::uint32_t type = read_be32(&file[offset + 4]);
      const std::uint8_t* data = &file[offset + 8];
      PngChunk chunk;
      chunk.type = type;
      chunk.data.assign(data, data + length);
      chunk.offset = static_cast<std::uint32_t>(offset + 8);
      chunk.crc = read_be32(&file[offset + 8 + length]);
      image.chunks.push_back(std::move(chunk));

      if (type == chunk_type("IHDR")) {
         if (length != 13) {
            error = "bad IHDR";
            return false;
         }
         width = read_be32(data);
         height = read_be32(data + 4);
         bit_depth = data[8];
         color_type = data[9];
         interlace = data[12];
         saw_header = true;
      } else if (type == chunk_type("PLTE")) {
         plte.assign(data, data + length);
      } else if (type == chunk_type("tRNS")) {
         trns.assign(data, data + length);
      } else if (type == chunk_type("IDAT")) {
         idat.insert(idat.end(), data, data + length);
      } else if (type == chunk_type("IEND")) {
         saw_end = true;
         break;
      }
      offset += 12 + static_cast<std::size_t>(length);
   }

   if (!saw_header || !saw_end) {
      error = "missing IHDR or IEND";
      return false;
   }
   if (interlace != 0 && interlace != 1) {
      error = "unsupported format";
      return false;
   }
   // The target refuses 16-bit PNG input outright, and also rejects bit
   // depths that are invalid for the color type.
   bool depth_ok = false;
   switch (color_type) {
   case 0:
      depth_ok = bit_depth == 1 || bit_depth == 2 || bit_depth == 4 ||
                 bit_depth == 8;
      break;
   case 2:
   case 4:
   case 6:
      depth_ok = bit_depth == 8;
      break;
   case 3:
      depth_ok = bit_depth == 1 || bit_depth == 2 || bit_depth == 4 ||
                 bit_depth == 8;
      break;
   default:
      break;
   }
   if (!depth_ok) {
      error = "unsupported format";
      return false;
   }
   if (width == 0 || height == 0) {
      error = "empty image";
      return false;
   }

   int channels = 0;
   switch (color_type) {
   case 0:
      channels = 1;
      break;
   case 2:
      channels = 3;
      break;
   case 3:
      channels = 1;
      break;
   case 4:
      channels = 2;
      break;
   case 6:
      channels = 4;
      break;
   default:
      error = "unsupported color type";
      return false;
   }

   std::vector<std::uint8_t> raw;
   if (!inflate_zlib(idat, raw)) {
      error = "IDAT inflate failed";
      return false;
   }

   image.width = width;
   image.height = height;
   image.bit_depth = bit_depth;
   image.source_color_type = color_type;
   image.channels = channels;
   image.palette = (color_type == 3);
   image.filters.assign(height, 0);
   image.samples.assign(static_cast<std::size_t>(width) * height *
                           static_cast<std::size_t>(channels),
                        0);

   if (color_type == 3) {
      image.palette_rgba.resize(plte.size() / 3);
      for (std::size_t i = 0; i < image.palette_rgba.size(); ++i) {
         std::uint8_t alpha = 255;
         if (i < trns.size()) alpha = trns[i];
         image.palette_rgba[i] =
            (static_cast<std::uint32_t>(alpha) << 24) |
            (static_cast<std::uint32_t>(plte[i * 3]) << 16) |
            (static_cast<std::uint32_t>(plte[i * 3 + 1]) << 8) |
            static_cast<std::uint32_t>(plte[i * 3 + 2]);
      }
   }

   // A tRNS chunk on grayscale or truecolor input names one fully transparent
   // sample value. Palette tRNS is handled above.
   if (color_type == 0 && trns.size() >= 2) {
      const std::uint32_t raw = (static_cast<std::uint32_t>(trns[0]) << 8) |
                                trns[1];
      const std::uint32_t scale =
         bit_depth < 8 ? 255u / ((1u << bit_depth) - 1) : 1u;
      const std::uint32_t gray = (raw * scale) & 0xffu;
      image.has_transparent_key = true;
      image.transparent_key = gray * 0x00010101u;
   } else if (color_type == 2 && trns.size() >= 6) {
      const std::uint32_t r = (static_cast<std::uint32_t>(trns[0]) << 8) | trns[1];
      const std::uint32_t g = (static_cast<std::uint32_t>(trns[2]) << 8) | trns[3];
      const std::uint32_t b = (static_cast<std::uint32_t>(trns[4]) << 8) | trns[5];
      image.has_transparent_key = true;
      image.transparent_key = ((r & 0xffu) << 16) | ((g & 0xffu) << 8) |
                              (b & 0xffu);
   }

   const std::size_t bits_per_pixel =
      static_cast<std::size_t>(channels) * static_cast<std::size_t>(bit_depth);
   const std::size_t bpp = (bits_per_pixel + 7) / 8;

   // Adam7 pass origins and steps. A non-interlaced file is a single 1x1 pass.
   static constexpr int kAdam7[7][4] = {
      {0, 0, 8, 8}, {4, 0, 8, 8}, {0, 4, 4, 8}, {2, 0, 4, 4},
      {0, 2, 2, 4}, {1, 0, 2, 2}, {0, 1, 1, 2}};
   const int pass_count = interlace ? 7 : 1;

   std::size_t raw_offset = 0;
   for (int pass = 0; pass < pass_count; ++pass) {
      const int x_start = interlace ? kAdam7[pass][0] : 0;
      const int y_start = interlace ? kAdam7[pass][1] : 0;
      const int x_step = interlace ? kAdam7[pass][2] : 1;
      const int y_step = interlace ? kAdam7[pass][3] : 1;
      const std::size_t pass_width =
         width > static_cast<std::uint32_t>(x_start)
            ? (width - x_start + x_step - 1) / x_step
            : 0;
      const std::size_t pass_height =
         height > static_cast<std::uint32_t>(y_start)
            ? (height - y_start + y_step - 1) / y_step
            : 0;
      if (pass_width == 0 || pass_height == 0) continue;

      const std::size_t row_bytes = (pass_width * bits_per_pixel + 7) / 8;
      if (raw_offset + (row_bytes + 1) * pass_height > raw.size()) {
         error = "IDAT too short";
         return false;
      }
      std::vector<std::uint8_t> previous(row_bytes, 0);
      std::vector<std::uint8_t> current(row_bytes, 0);

      for (std::size_t py = 0; py < pass_height; ++py) {
         const std::uint8_t* row = &raw[raw_offset];
         raw_offset += row_bytes + 1;
         const std::uint8_t filter = row[0];
         std::memcpy(current.data(), row + 1, row_bytes);
         unfilter_row(filter, current.data(), row_bytes,
                      py ? previous.data() : nullptr, bpp);
         const std::size_t y = static_cast<std::size_t>(y_start) + py * y_step;
         // -f6 reuses these bytes. For interlaced input a row can be written
         // by several passes; the last pass to touch it wins.
         image.filters[y] = filter;

         if (color_type == 3) {
            for (std::size_t px = 0; px < pass_width; ++px) {
               const std::size_t x =
                  static_cast<std::size_t>(x_start) + px * x_step;
               if (bit_depth == 8) {
                  image.samples[y * width + x] = current[px];
               } else {
                  const int per_byte = 8 / bit_depth;
                  const std::uint32_t mask = (1u << bit_depth) - 1;
                  const std::uint8_t byte = current[px / per_byte];
                  const int shift =
                     8 - bit_depth * (static_cast<int>(px % per_byte) + 1);
                  image.samples[y * width + x] =
                     static_cast<std::uint8_t>((byte >> shift) & mask);
               }
            }
         } else if (color_type == 0 || color_type == 4) {
            for (std::size_t px = 0; px < pass_width; ++px) {
               const std::size_t x =
                  static_cast<std::size_t>(x_start) + px * x_step;
               std::uint8_t* out =
                  &image.samples[(y * width + x) *
                                 static_cast<std::size_t>(channels)];
               if (bit_depth == 8) {
                  std::memcpy(out,
                              &current[px * static_cast<std::size_t>(channels)],
                              static_cast<std::size_t>(channels));
               } else {
                  const int per_byte = 8 / bit_depth;
                  const std::uint32_t mask = (1u << bit_depth) - 1;
                  // Sub-8-bit grayscale samples expand to the full 0..255
                  // range, matching the target's normalisation step.
                  const int scale = 255 / ((1 << bit_depth) - 1);
                  const std::uint8_t byte = current[px / per_byte];
                  const int shift =
                     8 - bit_depth * (static_cast<int>(px % per_byte) + 1);
                  out[0] = static_cast<std::uint8_t>(
                     ((byte >> shift) & mask) * scale);
               }
            }
         } else {
            for (std::size_t px = 0; px < pass_width; ++px) {
               const std::size_t x =
                  static_cast<std::size_t>(x_start) + px * x_step;
               std::uint8_t* out =
                  &image.samples[(y * width + x) *
                                 static_cast<std::size_t>(channels)];
               std::memcpy(out,
                           &current[px * static_cast<std::size_t>(channels)],
                           static_cast<std::size_t>(channels));
            }
         }
         previous.swap(current);
      }
   }

   return true;
}

} // namespace pngout
