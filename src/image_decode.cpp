#include "png_image.hpp"

#include <array>
#include <cstring>
#include <vector>

namespace pngout {
namespace {

// Small little-endian readers with bounds checks. The reference decoder trusts
// the input file, but returning zero past the end keeps malformed files from
// reading outside our buffer.
std::uint8_t byte_at(const std::vector<std::uint8_t>& file, std::size_t offset) {
   return offset < file.size() ? file[offset] : 0;
}

std::uint32_t le16_at(const std::vector<std::uint8_t>& file,
                      std::size_t offset) {
   return static_cast<std::uint32_t>(byte_at(file, offset)) |
          (static_cast<std::uint32_t>(byte_at(file, offset + 1)) << 8);
}

std::uint32_t le32_at(const std::vector<std::uint8_t>& file,
                      std::size_t offset) {
   return static_cast<std::uint32_t>(byte_at(file, offset)) |
          (static_cast<std::uint32_t>(byte_at(file, offset + 1)) << 8) |
          (static_cast<std::uint32_t>(byte_at(file, offset + 2)) << 16) |
          (static_cast<std::uint32_t>(byte_at(file, offset + 3)) << 24);
}

void reset_palette_image(Image& image) {
   image.palette = true;
   image.palette_rgba.clear();
   image.palette_fill = 0;
   image.samples.assign(static_cast<std::size_t>(image.width) * image.height, 0);
}

// ---------------------------------------------------------------------------
// BMP
//
// The reference accepts the 12-byte core header and any later info header.
// 1/4/8bpp inputs become palette images (depth equals the source bpp); 16/24/32
// become truecolor. A 32-bit file always has its alpha byte forced opaque.
// 16-bit defaults to RGB555; a BI_BITFIELDS header supplies explicit masks.
// ---------------------------------------------------------------------------

struct ChannelBits {
   unsigned shift{0};
   std::uint32_t mask{0};
};

// Split a bit mask into the shift and output mask used by the reference:
//   k = lowest set bit, n = number of consecutive set bits starting at k
//   shift = (base - (k + n)) mod 32
//   mask  = ((uint32)-1 << (base - n)) & (0xff << (base - 8))
ChannelBits make_channel(std::uint32_t bitmask, int base) {
   ChannelBits result;
   if (bitmask == 0) return result;
   unsigned k = 0;
   while (((bitmask >> k) & 1u) == 0) ++k;
   std::uint32_t remaining = bitmask >> k;
   unsigned n = 0;
   while ((remaining & 1u) != 0) {
      ++n;
      remaining >>= 1;
   }
   result.shift = static_cast<unsigned>(base - static_cast<int>(k + n)) & 31u;
   result.mask =
      (0xffffffffu << ((static_cast<unsigned>(base) - n) & 31u)) &
      (0xffu << (base - 8));
   return result;
}

// The reference loads 16-bit samples with a sign-extending move and then uses
// an arithmetic right shift in its rotate, so a sample with bit 15 set fills
// the high bits with ones before masking. Reproduce that exactly.
std::uint32_t extract_channel(std::int32_t value, const ChannelBits& bits) {
   if (bits.mask == 0) return 0;
   const std::uint32_t rotated = static_cast<std::uint32_t>(
      bits.shift == 0 ? value
                      : ((value >> ((32u - bits.shift) & 31u)) |
                         (value << bits.shift)));
   return rotated & bits.mask;
}

bool decode_bmp(const std::vector<std::uint8_t>& file, Image& image,
                std::string& error) {
   if (file.size() < 0x1e) return false;
   const std::uint32_t dib_size = le32_at(file, 0xe);
   const bool core = dib_size == 0xc;
   std::uint32_t width = 0;
   std::uint32_t height = 0;
   int bpp = 0;
   std::uint32_t compression = 0;
   std::size_t palette_start = 0;
   std::size_t palette_entry_size = 0;

   if (core) {
      if (le16_at(file, 0x16) != 1) return false;
      width = le16_at(file, 0x12);
      height = le16_at(file, 0x14);
      bpp = static_cast<int>(le16_at(file, 0x18));
      palette_start = 0xe + 0xc;
      palette_entry_size = 3;
   } else {
      if (le16_at(file, 0x1a) != 1) return false;
      const std::int32_t signed_width =
         static_cast<std::int32_t>(le32_at(file, 0x12));
      const std::int32_t signed_height =
         static_cast<std::int32_t>(le32_at(file, 0x16));
      if (signed_width < 1) return false;
      width = static_cast<std::uint32_t>(signed_width);
      height = static_cast<std::uint32_t>(
         signed_height < 0 ? -signed_height : signed_height);
      bpp = static_cast<int>(le16_at(file, 0x1c));
      compression = le32_at(file, 0x1e);
      palette_start = 0xe + dib_size;
      palette_entry_size = 4;
   }

   if (height == 0) return false;
   if (!(bpp == 1 || bpp == 4 || bpp == 8 || bpp == 16 || bpp == 24 ||
         bpp == 32)) {
      return false;
   }
   if (compression != 0 && compression != 3) return false;

   image.width = width;
   image.height = height;
   image.bit_depth = bpp;
   image.source_color_type = bpp <= 15 ? 3 : 2;
   image.channels = 1;

   const std::uint32_t pixel_offset = le32_at(file, 0xa);
   const std::size_t stride =
      ((static_cast<std::size_t>(width) * bpp + 31) >> 3) & ~std::size_t{3};

   if (bpp <= 15) {
      reset_palette_image(image);
      std::size_t cursor = palette_start;
      const std::size_t end =
         pixel_offset < file.size() ? pixel_offset : file.size();
      while (cursor < end && image.palette_rgba.size() < 256) {
         std::uint32_t entry = 0;
         if (palette_entry_size == 3) {
            entry = (static_cast<std::uint32_t>(byte_at(file, cursor + 2))
                     << 16) |
                    (static_cast<std::uint32_t>(byte_at(file, cursor + 1))
                     << 8) |
                    byte_at(file, cursor);
            cursor += 3;
         } else {
            entry = (static_cast<std::uint32_t>(byte_at(file, cursor + 2))
                     << 16) |
                    (static_cast<std::uint32_t>(byte_at(file, cursor + 1))
                     << 8) |
                    byte_at(file, cursor);
            cursor += 4;
         }
         image.palette_rgba.push_back(entry | 0xff000000u);
      }
      // The global palette starts zeroed, so an index past the table maps to
      // transparent black (the analysis then falls back to truecolor).
      image.palette_fill = 0;

      std::size_t row = pixel_offset;
      std::ptrdiff_t step = static_cast<std::ptrdiff_t>(stride);
      if (static_cast<std::int32_t>(le32_at(file, 0x16)) >= 0 && !core) {
         row += (static_cast<std::size_t>(height) - 1) * stride;
         step = -static_cast<std::ptrdiff_t>(stride);
      } else if (core) {
         row += (static_cast<std::size_t>(height) - 1) * stride;
         step = -static_cast<std::ptrdiff_t>(stride);
      }

      for (std::uint32_t y = 0; y < height; ++y) {
         const std::size_t row_base =
            static_cast<std::size_t>(static_cast<std::ptrdiff_t>(row) +
                                     static_cast<std::ptrdiff_t>(y) * step);
         for (std::uint32_t x = 0; x < width; ++x) {
            std::uint8_t index = 0;
            if (bpp == 1) {
               const std::size_t offset = row_base + (x >> 3);
               index = static_cast<std::uint8_t>(
                  (byte_at(file, offset) >> (7 - (x & 7))) & 1u);
            } else if (bpp == 4) {
               const std::size_t offset = row_base + (x >> 1);
               const unsigned shift = (x & 1) == 0 ? 4 : 0;
               index = static_cast<std::uint8_t>(
                  (byte_at(file, offset) >> shift) & 0xfu);
            } else {
               index = byte_at(file, row_base + x);
            }
            image.samples[static_cast<std::size_t>(y) * width + x] = index;
         }
      }
      return true;
   }

   // Truecolor: 16/24/32bpp.
   image.palette = false;
   image.channels = bpp == 32 ? 4 : 3;
   image.samples.assign(static_cast<std::size_t>(width) * height *
                           image.channels,
                        0);

   ChannelBits red;
   ChannelBits green;
   ChannelBits blue;
   if (compression == 3) {
      const std::size_t masks = 0xe + dib_size;
      red = make_channel(le32_at(file, masks + 0), 24);
      green = make_channel(le32_at(file, masks + 4), 16);
      blue = make_channel(le32_at(file, masks + 8), 8);
   } else if (bpp == 16) {
      red = make_channel(0x7c00, 24);
      green = make_channel(0x03e0, 16);
      blue = make_channel(0x001f, 8);
   } else {
      red = make_channel(0x00ff0000, 24);
      green = make_channel(0x0000ff00, 16);
      blue = make_channel(0x000000ff, 8);
   }

   const bool bottom_up = core || static_cast<std::int32_t>(le32_at(file, 0x16)) >= 0;
   std::size_t row = pixel_offset;
   std::ptrdiff_t step = static_cast<std::ptrdiff_t>(stride);
   if (bottom_up) {
      row += (static_cast<std::size_t>(height) - 1) * stride;
      step = -static_cast<std::ptrdiff_t>(stride);
   }

   for (std::uint32_t y = 0; y < height; ++y) {
      const std::size_t row_base =
         static_cast<std::size_t>(static_cast<std::ptrdiff_t>(row) +
                                  static_cast<std::ptrdiff_t>(y) * step);
      for (std::uint32_t x = 0; x < width; ++x) {
         std::uint32_t value = 0;
         std::uint32_t alpha = 0xff000000u;
         if (bpp == 16) {
            const std::int32_t raw = static_cast<std::int16_t>(
               static_cast<std::uint16_t>(le16_at(file, row_base + x * 2)));
            value = extract_channel(raw, red) |
                    extract_channel(raw, green) |
                    extract_channel(raw, blue);
            value |= alpha;
         } else if (bpp == 24) {
            const std::size_t offset = row_base + x * 3;
            value = alpha |
                    (static_cast<std::uint32_t>(byte_at(file, offset + 2))
                     << 16) |
                    (static_cast<std::uint32_t>(byte_at(file, offset + 1))
                     << 8) |
                    byte_at(file, offset);
         } else {
            const std::uint32_t raw = le32_at(file, row_base + x * 4);
            const std::int32_t signed_raw = static_cast<std::int32_t>(raw);
            value = extract_channel(signed_raw, red) |
                    extract_channel(signed_raw, green) |
                    extract_channel(signed_raw, blue) | alpha;
         }
         std::uint8_t* sample =
            &image.samples[(static_cast<std::size_t>(y) * width + x) *
                           image.channels];
         sample[0] = static_cast<std::uint8_t>((value >> 16) & 0xffu);
         sample[1] = static_cast<std::uint8_t>((value >> 8) & 0xffu);
         sample[2] = static_cast<std::uint8_t>(value & 0xffu);
         if (image.channels == 4) {
            sample[3] = static_cast<std::uint8_t>((value >> 24) & 0xffu);
         }
      }
   }
   (void)error;
   return true;
}

// ---------------------------------------------------------------------------
// TGA
//
// Types 1/2/3/9/10/11 are accepted. The output colour type comes from the
// pixel (or colour-map) entry size: 1 -> gray, 2/3 -> RGB, 4 -> RGBA. An
// 8-bit colour-mapped image indexes the gray ramp rather than the map data,
// matching the reference. Descriptor bits 4 and 5 flip the image axes.
// ---------------------------------------------------------------------------

std::uint32_t tga_ramp(std::uint8_t index) {
   return 0xff000000u | (static_cast<std::uint32_t>(index) * 0x00010101u);
}

std::uint32_t tga_rgb555(std::uint32_t value) {
   return 0xff000000u | ((value & 0x1fu) << 3) | ((value & 0x3e0u) << 6) |
          ((value & 0x7c00u) << 9);
}

bool decode_tga(const std::vector<std::uint8_t>& file, Image& image,
                std::string& error) {
   if (file.size() < 0x13) return false;
   const std::uint8_t id_length = byte_at(file, 0);
   const std::uint8_t color_map_type = byte_at(file, 1);
   const std::uint8_t image_type = byte_at(file, 2);
   if (color_map_type > 1) return false;
   if (!(image_type == 1 || image_type == 2 || image_type == 3 ||
         image_type == 9 || image_type == 10 || image_type == 11)) {
      return false;
   }
   const int bpp = static_cast<int>(byte_at(file, 0x10));
   if (!(bpp == 8 || bpp == 16 || bpp == 24 || bpp == 32)) return false;
   const std::uint8_t descriptor = byte_at(file, 0x11);
   if ((descriptor & 0xc0) != 0) return false;
   const std::uint32_t width = le16_at(file, 0xc);
   const std::uint32_t height = le16_at(file, 0xe);
   if (width == 0 || height == 0) return false;

   const std::uint32_t map_first = le16_at(file, 3);
   const std::uint32_t map_length = le16_at(file, 5);
   const int map_depth = static_cast<int>(byte_at(file, 7));

   const std::size_t pixel_data = 0x12 + id_length;
   const int pixel_bytes = (bpp + 7) >> 3;
   int entry_bytes = pixel_bytes;
   std::size_t map_start = pixel_data;
   std::size_t data_start = pixel_data;
   if (color_map_type == 1) {
      entry_bytes = (map_depth + 7) >> 3;
      data_start = pixel_data + static_cast<std::size_t>(map_length) *
                                    static_cast<std::size_t>(entry_bytes);
   }

   if (entry_bytes == 1) {
      image.source_color_type = 0;
      image.bit_depth = 8;
      image.channels = 1;
   } else if (entry_bytes == 2 || entry_bytes == 3) {
      image.source_color_type = 2;
      image.bit_depth = 8;
      image.channels = 3;
   } else {
      image.source_color_type = 6;
      image.bit_depth = 8;
      image.channels = 4;
   }
   image.width = width;
   image.height = height;
   image.palette = false;
   image.samples.assign(static_cast<std::size_t>(width) * height *
                           image.channels,
                        0);

   const auto decode_entry = [&](std::size_t offset) -> std::uint32_t {
      if (entry_bytes == 1) {
         return tga_ramp(byte_at(file, offset));
      }
      if (entry_bytes == 2) {
         return tga_rgb555(le16_at(file, offset));
      }
      if (entry_bytes == 3) {
         return 0xff000000u |
                (static_cast<std::uint32_t>(byte_at(file, offset + 2))
                 << 16) |
                (static_cast<std::uint32_t>(byte_at(file, offset + 1)) << 8) |
                byte_at(file, offset);
      }
      return (static_cast<std::uint32_t>(byte_at(file, offset + 3)) << 24) |
             (static_cast<std::uint32_t>(byte_at(file, offset + 2)) << 16) |
             (static_cast<std::uint32_t>(byte_at(file, offset + 1)) << 8) |
             byte_at(file, offset);
   };

   // Read the next pixel from the stream. A colour-mapped file stores an index
   // (one or two bytes, independent of the map entry size) that addresses the
   // colour map; otherwise the pixel bytes are decoded directly.
   std::size_t cursor = data_start;
   const auto read_pixel = [&]() -> std::uint32_t {
      if (color_map_type == 1) {
         std::uint32_t index = 0;
         if (pixel_bytes == 1) {
            index = byte_at(file, cursor);
         } else {
            index = le16_at(file, cursor);
         }
         cursor += static_cast<std::size_t>(pixel_bytes);
         const std::size_t entry =
            map_start + (static_cast<std::size_t>(index) -
                         static_cast<std::size_t>(map_first)) *
                            static_cast<std::size_t>(entry_bytes);
         return decode_entry(entry);
      }
      const std::uint32_t value = decode_entry(cursor);
      cursor += static_cast<std::size_t>(pixel_bytes);
      return value;
   };

   const bool rle = image_type >= 9;
   const bool left_to_right = (descriptor & 0x10) == 0;
   const bool top_to_bottom = (descriptor & 0x20) != 0;

   // The reference keeps a signed counter seeded with -1 for RLE and -2 for
   // plain data. A value below 0x80 consumes pixels; when its low seven bits
   // are all set it first reads a fresh packet header. Values of 0x80 and
   // above repeat the previous pixel without touching the stream.
   int run_remaining = rle ? -1 : -2;
   std::uint32_t run_value = 0;
   for (std::uint32_t row = 0; row < height; ++row) {
      const std::uint32_t y = top_to_bottom ? row : height - 1 - row;
      for (std::uint32_t column = 0; column < width; ++column) {
         const std::uint32_t x = left_to_right ? column : width - 1 - column;
         std::uint32_t value = run_value;
         if (run_remaining < 0x80) {
            if ((run_remaining & 0x7f) == 0x7f) {
               run_remaining = static_cast<int>(byte_at(file, cursor++));
            }
            value = read_pixel();
            run_value = value;
            if (run_remaining >= 0) --run_remaining;
         } else {
            --run_remaining;
         }

         std::uint8_t* sample =
            &image.samples[(static_cast<std::size_t>(y) * width + x) *
                           image.channels];
         if (image.channels == 1) {
            sample[0] = static_cast<std::uint8_t>(value & 0xffu);
         } else {
            sample[0] = static_cast<std::uint8_t>((value >> 16) & 0xffu);
            sample[1] = static_cast<std::uint8_t>((value >> 8) & 0xffu);
            sample[2] = static_cast<std::uint8_t>(value & 0xffu);
            if (image.channels == 4) {
               sample[3] = static_cast<std::uint8_t>((value >> 24) & 0xffu);
            }
         }
      }
   }
   (void)error;
   return true;
}

// ---------------------------------------------------------------------------
// PCX
//
// Only the 0x0A 0x05 0x01 0x08 header is accepted. One plane is an 8-bit
// palette image whose 256-colour table lives in the last 768 bytes. Three
// planes are interleaved R/G/B rows. Rows always span bytesPerLine samples
// (clipped to the logical width when writing).
// ---------------------------------------------------------------------------

bool decode_pcx(const std::vector<std::uint8_t>& file, Image& image,
                std::string& error) {
   if (file.size() < 0x80) return false;
   const std::uint32_t xmin = le16_at(file, 4);
   const std::uint32_t ymin = le16_at(file, 6);
   const std::uint32_t xmax = le16_at(file, 8);
   const std::uint32_t ymax = le16_at(file, 10);
   if (xmax < xmin || ymax < ymin) return false;
   const std::uint32_t width = xmax - xmin + 1;
   const std::uint32_t height = ymax - ymin + 1;
   if (width == 0 || height == 0) return false;
   const int planes = byte_at(file, 0x41);
   const std::uint32_t bytes_per_line = le16_at(file, 0x42);

   image.width = width;
   image.height = height;
   image.bit_depth = 8;
   if (planes == 1) {
      reset_palette_image(image);
      image.source_color_type = 3;
      image.palette_rgba.assign(256, 0xff000000u);
      if (file.size() >= 768) {
         const std::size_t base = file.size() - 768;
         for (std::size_t i = 0; i < 256; ++i) {
            const std::size_t offset = base + i * 3;
            image.palette_rgba[i] =
               0xff000000u |
               (static_cast<std::uint32_t>(byte_at(file, offset)) << 16) |
               (static_cast<std::uint32_t>(byte_at(file, offset + 1)) << 8) |
               byte_at(file, offset + 2);
         }
      }
   } else if (planes == 3) {
      image.palette = false;
      image.source_color_type = 2;
      image.channels = 3;
      image.samples.assign(static_cast<std::size_t>(width) * height * 3, 0);
   } else {
      return false;
   }

   const std::uint32_t span = bytes_per_line > width ? bytes_per_line : width;
   std::size_t cursor = 0x80;

   if (planes == 1) {
      for (std::uint32_t y = 0; y < height; ++y) {
         std::uint32_t x = 0;
         while (x < span) {
            const std::uint8_t control = byte_at(file, cursor++);
            std::uint32_t count = 1;
            std::uint8_t value = control;
            if (control >= 0xc0) {
               count = control & 0x3f;
               value = byte_at(file, cursor++);
               if (count == 0) continue;
            }
            for (std::uint32_t i = 0; i < count && x < span; ++i, ++x) {
               if (x < width) {
                  image.samples[static_cast<std::size_t>(y) * width + x] =
                     value;
               }
            }
         }
      }
      return true;
   }

   for (std::uint32_t y = 0; y < height; ++y) {
      for (int plane = 0; plane < 3; ++plane) {
         std::uint32_t x = 0;
         while (x < span) {
            const std::uint8_t control = byte_at(file, cursor++);
            std::uint32_t count = 1;
            std::uint8_t value = control;
            if (control >= 0xc0) {
               count = control & 0x3f;
               value = byte_at(file, cursor++);
               if (count == 0) continue;
            }
            for (std::uint32_t i = 0; i < count && x < span; ++i, ++x) {
               if (x < width) {
                  const std::size_t pixel =
                     (static_cast<std::size_t>(y) * width + x) * 3;
                  // Plane 0 is red, 1 green, 2 blue, matching the ARGB word
                  // the reference writes (red in the high byte).
                  image.samples[pixel + plane] = value;
               }
            }
         }
      }
   }
   (void)error;
   return true;
}

// ---------------------------------------------------------------------------
// GIF
//
// One frame is decoded into a screen-sized palette image. The global (or
// local) colour table becomes the palette; entries past the table are opaque
// black. A transparent index clears the alpha byte. Frames smaller than the
// screen are surrounded by the background colour (zero when the file has no
// global table). LZW follows the GIF specification; the reference implements
// the same table growth with its entry creation shifted one code earlier,
// which produces identical output.
// ---------------------------------------------------------------------------

struct GifBitReader {
   const std::vector<std::uint8_t>& data;
   std::size_t bit_position{0};

   bool read(int bits, std::uint32_t& value) {
      if (bit_position + static_cast<std::size_t>(bits) > data.size() * 8) {
         return false;
      }
      std::uint32_t result = 0;
      for (int i = 0; i < bits; ++i) {
         const std::size_t position = bit_position + static_cast<std::size_t>(i);
         result |= static_cast<std::uint32_t>(
                      (data[position >> 3] >> (position & 7)) & 1u)
                   << i;
      }
      bit_position += static_cast<std::size_t>(bits);
      value = result;
      return true;
   }
};

bool gif_lzw(const std::vector<std::uint8_t>& compressed, int min_code_size,
             std::vector<std::uint8_t>& output) {
   if (min_code_size < 1 || min_code_size > 11) return false;
   const int clear = 1 << min_code_size;
   const int end = clear + 1;
   int code_size = min_code_size + 1;
   int next_code = clear + 2;

   std::array<int, 4096> prefix{};
   std::array<std::uint8_t, 4096> suffix{};
   std::array<std::uint8_t, 4096> first{};
   for (int i = 0; i < clear; ++i) {
      prefix[i] = -1;
      suffix[i] = static_cast<std::uint8_t>(i);
      first[i] = static_cast<std::uint8_t>(i);
   }

   GifBitReader reader{compressed};
   int previous = -1;
   std::uint8_t previous_first = 0;
   std::array<std::uint8_t, 4096> stack{};

   while (true) {
      std::uint32_t code_value = 0;
      if (!reader.read(code_size, code_value)) break;
      const int code = static_cast<int>(code_value);
      if (code == clear) {
         code_size = min_code_size + 1;
         next_code = clear + 2;
         previous = -1;
         continue;
      }
      if (code == end) break;

      if (previous < 0) {
         if (code >= clear) break;
         output.push_back(static_cast<std::uint8_t>(code));
         previous = code;
         previous_first = static_cast<std::uint8_t>(code);
         continue;
      }

      int walk = code;
      std::uint8_t first_of_code = 0;
      std::size_t depth = 0;
      if (code < clear) {
         first_of_code = static_cast<std::uint8_t>(code);
      } else if (code < next_code) {
         first_of_code = first[code];
      } else {
         first_of_code = previous_first;
      }

      if (code >= next_code) {
         // KwKwK: the code being defined is the previous string plus its own
         // first character.
         stack[depth++] = previous_first;
         walk = previous;
      }
      while (walk >= clear) {
         stack[depth++] = suffix[walk];
         walk = prefix[walk];
         if (depth >= stack.size()) return false;
      }
      stack[depth++] = static_cast<std::uint8_t>(walk);
      for (std::size_t i = depth; i > 0; --i) {
         output.push_back(stack[i - 1]);
      }

      if (next_code < 4096) {
         prefix[next_code] = previous;
         suffix[next_code] = first_of_code;
         first[next_code] = previous_first;
         ++next_code;
      }
      if (next_code == (1 << code_size) && code_size < 12) {
         ++code_size;
      }

      previous = code;
      previous_first = first_of_code;
   }
   return true;
}

bool decode_gif(const std::vector<std::uint8_t>& file, Image& image,
                std::string& error) {
   if (file.size() < 13) return false;
   const std::uint32_t screen_width = le16_at(file, 6);
   const std::uint32_t screen_height = le16_at(file, 8);
   if (screen_width == 0 || screen_height == 0) return false;
   const std::uint8_t flags = byte_at(file, 10);
   const std::uint8_t background_index = byte_at(file, 11);
   const bool has_global_table = (flags & 0x80) != 0;
   const std::uint32_t global_size =
      1u << ((static_cast<unsigned>(flags) & 7u) + 1u);

   std::size_t cursor = 13;
   const std::size_t global_start = cursor;
   if (has_global_table) cursor += static_cast<std::size_t>(global_size) * 3;

   int transparent_index = -1;
   while (cursor < file.size() && byte_at(file, cursor) == 0x21) {
      const std::uint8_t label = byte_at(file, cursor + 1);
      if (label == 0xf9 && (byte_at(file, cursor + 3) & 1) != 0) {
         transparent_index = byte_at(file, cursor + 6);
      }
      cursor += 2;
      std::uint8_t block = 0;
      do {
         block = byte_at(file, cursor);
         cursor += static_cast<std::size_t>(block) + 1;
      } while (block != 0 && cursor < file.size());
   }
   if (cursor >= file.size() || byte_at(file, cursor) != 0x2c) {
      image.source_color_type = 3;
      image.bit_depth = 8;
      return false;
   }

   const std::uint32_t left = le16_at(file, cursor + 1);
   const std::uint32_t top = le16_at(file, cursor + 3);
   const std::uint32_t frame_width = le16_at(file, cursor + 5);
   const std::uint32_t frame_height = le16_at(file, cursor + 7);
   const std::uint8_t image_flags = byte_at(file, cursor + 9);
   std::size_t descriptor_end = cursor + 10;

   std::size_t palette_start = global_start;
   std::uint32_t palette_size = global_size;
   if ((image_flags & 0x80) != 0) {
      palette_size = 1u << ((static_cast<unsigned>(image_flags) & 7u) + 1u);
      palette_start = descriptor_end;
      descriptor_end += static_cast<std::size_t>(palette_size) * 3;
   }

   image.width = screen_width;
   image.height = screen_height;
   image.bit_depth = 8;
   image.source_color_type = 3;
   reset_palette_image(image);
   image.palette_fill = 0xff000000u;
   image.palette_rgba.assign(palette_size, 0xff000000u);
   for (std::uint32_t i = 0; i < palette_size; ++i) {
      const std::size_t offset = palette_start + static_cast<std::size_t>(i) * 3;
      image.palette_rgba[i] =
         0xff000000u |
         (static_cast<std::uint32_t>(byte_at(file, offset)) << 16) |
         (static_cast<std::uint32_t>(byte_at(file, offset + 1)) << 8) |
         byte_at(file, offset + 2);
   }
   if (transparent_index >= 0 &&
       static_cast<std::size_t>(transparent_index) < image.palette_rgba.size()) {
      image.palette_rgba[transparent_index] &= 0x00ffffffu;
   }

   // Background fill outside a partial frame. The reference consults the
   // active palette but only when the file declares a global table.
   std::uint32_t background = 0;
   if (has_global_table &&
       static_cast<std::size_t>(background_index) < image.palette_rgba.size()) {
      background = image.palette_rgba[background_index];
   }
   image.samples.assign(static_cast<std::size_t>(screen_width) * screen_height,
                        0);
   const bool full_frame = left == 0 && top == 0 &&
                           frame_width == screen_width &&
                           frame_height == screen_height;
   if (!full_frame) {
      // The reference fills everything outside the frame rectangle. Pixels
      // inside the frame that the stream never reaches stay zero.
      for (std::uint32_t y = 0; y < screen_height; ++y) {
         for (std::uint32_t x = 0; x < screen_width; ++x) {
            const bool inside = x >= left && x < left + frame_width &&
                                y >= top && y < top + frame_height;
            if (!inside) {
               // Store the ARGB value as an index only when it is present in
               // the palette; otherwise fall back to truecolor semantics by
               // using an out-of-range index with the matching fill.
               image.samples[static_cast<std::size_t>(y) * screen_width + x] =
                  0;
            }
         }
      }
   }

   const int min_code_size = static_cast<int>(byte_at(file, descriptor_end));
   std::size_t block_cursor = descriptor_end + 1;
   std::vector<std::uint8_t> compressed;
   while (block_cursor < file.size()) {
      const std::uint8_t block = byte_at(file, block_cursor++);
      if (block == 0) break;
      compressed.insert(compressed.end(), file.begin() + block_cursor,
                        file.begin() + block_cursor + block);
      block_cursor += block;
   }

   std::vector<std::uint8_t> indices;
   if (!gif_lzw(compressed, min_code_size, indices)) {
      (void)error;
      return false;
   }

   const bool interlaced = (image_flags & 0x40) != 0;
   std::array<std::uint32_t, 4> pass_start{0, 4, 2, 1};
   std::array<std::uint32_t, 4> pass_step{8, 8, 4, 2};
   std::size_t source = 0;
   if (interlaced) {
      for (int pass = 0; pass < 4; ++pass) {
         for (std::uint32_t row = pass_start[pass]; row < frame_height;
              row += pass_step[pass]) {
            for (std::uint32_t column = 0; column < frame_width; ++column) {
               if (source >= indices.size()) break;
               const std::uint32_t y = top + row;
               const std::uint32_t x = left + column;
               if (y < screen_height && x < screen_width) {
                  image.samples[static_cast<std::size_t>(y) * screen_width + x] =
                     indices[source];
               }
               ++source;
            }
         }
      }
   } else {
      for (std::uint32_t row = 0; row < frame_height; ++row) {
         for (std::uint32_t column = 0; column < frame_width; ++column) {
            if (source >= indices.size()) break;
            const std::uint32_t y = top + row;
            const std::uint32_t x = left + column;
            if (y < screen_height && x < screen_width) {
               image.samples[static_cast<std::size_t>(y) * screen_width + x] =
                  indices[source];
            }
            ++source;
         }
      }
   }

   // Partial frames need the background colour stored as a palette index. If
   // the colour is not in the table, force the truecolor fallback by writing
   // an index past the end with the background value as the fill colour.
   if (!full_frame) {
      bool present = false;
      std::size_t index = 0;
      for (std::size_t i = 0; i < image.palette_rgba.size(); ++i) {
         if (image.palette_rgba[i] == background) {
            present = true;
            index = i;
            break;
         }
      }
      for (std::uint32_t y = 0; y < screen_height; ++y) {
         for (std::uint32_t x = 0; x < screen_width; ++x) {
            const bool inside = x >= left && x < left + frame_width &&
                                y >= top && y < top + frame_height;
            if (inside) continue;
            if (present) {
               image.samples[static_cast<std::size_t>(y) * screen_width + x] =
                  static_cast<std::uint8_t>(index);
            } else {
               image.samples[static_cast<std::size_t>(y) * screen_width + x] =
                  static_cast<std::uint8_t>(image.palette_rgba.size() & 0xff);
               image.palette_fill = background;
            }
         }
      }
   }

   (void)error;
   return true;
}

} // namespace

bool decode_image(const std::vector<std::uint8_t>& file, Image& image,
                  std::string& error) {
   if (file.size() >= 8 && file[0] == 0x89 && file[1] == 0x50 &&
       file[2] == 0x4e && file[3] == 0x47 && file[4] == 0x0d &&
       file[5] == 0x0a && file[6] == 0x1a && file[7] == 0x0a) {
      return decode_png(file, image, error);
   }
   if (file.size() >= 2 && file[0] == 0xff && file[1] == 0xd8) {
      return decode_jpeg(file, image, error);
   }
   if (file.size() >= 3 && file[0] == 'G' && file[1] == 'I' && file[2] == 'F') {
      return decode_gif(file, image, error);
   }
   if (file.size() >= 2 && file[0] == 0x19 && file[1] == 0x91) {
      error = "unsupported 0x19 image format";
      return false;
   }
   if (file.size() >= 2 && file[0] == 'B' && file[1] == 'M') {
      return decode_bmp(file, image, error);
   }
   if (file.size() >= 4 && file[0] == 0x0a && file[1] == 0x05 &&
       file[2] == 0x01 && file[3] == 0x08) {
      return decode_pcx(file, image, error);
   }
   if (file.size() >= 8 && file[0] == 'D' && file[1] == 'D' &&
       file[2] == 'S' && file[3] == ' ' && le32_at(file, 4) == 0x7c) {
      error = "DDS decoding is not implemented";
      return false;
   }
   return decode_tga(file, image, error);
}

} // namespace pngout
