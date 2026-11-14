#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace pngout {

// One PNG chunk as read from the input file. `data` excludes the length and
// CRC fields. Unknown chunks are kept verbatim so -k can copy them.
struct PngChunk {
   std::uint32_t type{};
   std::vector<std::uint8_t> data;
   std::uint32_t offset{}; // file offset of the chunk's data (chunk start + 8)
   std::uint32_t crc{};    // CRC field as stored in the file
};

// Decoded input image. PNGOUT normalises every supported input format into a
// small set of sample layouts before choosing an output colour type:
//
//   samples: 1 byte per grayscale sample, 2 per gray+alpha, 3 per RGB, 4 per
//            RGBA, or one palette index per pixel when `palette` is non-empty.
//
// `bit_depth` is the source bit depth (1/2/4/8/16). `filters` holds the source
// row filter bytes for -f6.
struct Image {
   std::uint32_t width{};
   std::uint32_t height{};
   int channels{1}; // sample bytes per pixel when palette is empty
   int bit_depth{8};
   int source_color_type{0}; // PNG colour type of the input file
   bool palette{false};
   // ARGB value used for a palette index past the end of `palette_rgba`.
   // The reference decoder keeps a 256-entry global palette that starts out
   // zeroed, so PNG and BMP pad with transparent black while the GIF decoder
   // explicitly fills the unused tail with opaque black.
   std::uint32_t palette_fill{0};
   std::vector<std::uint8_t> samples; // width*height*channels or indices
   std::vector<std::uint32_t> palette_rgba; // 0xAARRGGBB
   // tRNS colour key for grayscale/RGB input (0xAARRGGBB, alpha 0). When
   // present the matching sample becomes fully transparent.
   bool has_transparent_key{false};
   std::uint32_t transparent_key{0};
   std::vector<std::uint8_t> filters; // per source row
   std::vector<PngChunk> chunks; // all input chunks in file order
};

// Decode a PNG file. Returns false and fills `error` on failure.
bool decode_png(const std::vector<std::uint8_t>& file, Image& image,
                std::string& error);

// Decode a JPEG file into truecolor or grayscale samples.
bool decode_jpeg(const std::vector<std::uint8_t>& file, Image& image,
                 std::string& error);

// Decode any supported input file. The reference sniffs the leading bytes and
// dispatches to PNG, JPEG, GIF, a small 0x19 0x91 format, BMP, PCX, DDS or
// TGA. Formats this reconstruction cannot decode yet report failure, which the
// CLI turns into the same "unsupported format" message.
bool decode_image(const std::vector<std::uint8_t>& file, Image& image,
                  std::string& error);

// Encode the image to a PNG byte stream. `options` mirrors the CLI.
struct EncodeOptions {
   int filter{5};       // -f0..-f6
   int color_type{-1};  // -c# or -1 for automatic
   int bit_depth{-1};   // -d# or -1 for automatic
   bool keep_chunks{false}; // -k1
   std::string keep_chunk_list; // -kNAME,NAME,...
   double block_threshold{256.0}; // -b#
   std::size_t block_count{0};    // -n#
   int strategy{0};     // -s#
   bool verbose{false};
   bool quiet{false};
};

struct EncodeStats {
   std::size_t input_size{};
   std::size_t output_size{};   // final size, stored fallback included
   std::size_t deflate_size{};  // chosen IDAT deflate payload
   std::size_t png_overhead{};  // compressed output size minus deflate payload
   bool stored{false};          // IDAT written as uncompressed stored blocks
   int color_type{};
   int bit_depth{};
   int filter{}; // filter byte actually written to every output row
   std::size_t palette_colors{};
   std::vector<std::size_t> pass_sizes;
   std::vector<std::int64_t> pass_deltas;
   std::vector<std::size_t> block_ends;
};

bool encode_png(const Image& image, const EncodeOptions& options,
                std::vector<std::uint8_t>& output, EncodeStats& stats,
                std::string& error);

// Chunk preservation helpers, shared by the encoder and the console front end.
//
// The reference always regenerates the structural chunks (IHDR, PLTE, tRNS,
// IDAT and IEND). Every other chunk is copied verbatim when the -k option
// selects it. tRNS is only structural for colour types that can carry a colour
// key; an input that already has per-pixel alpha reports it as removable.
bool chunk_is_required(std::uint32_t type, int requested_color_type);

// True when `-k` selects `type`. The option string is the text after "-k":
// empty selects nothing, "1" or a case-insensitive "ALL" selects everything,
// and any other value is a comma separated list of exact 4-character names.
bool chunk_is_kept(std::uint32_t type, const EncodeOptions& options);

// The bKGD colour an input file declares, as 0xAARRGGBB (alpha zero), or 0
// when the file has no bKGD chunk.
std::uint32_t input_bkgd_color(const Image& image);

} // namespace pngout
