#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace pngout {

// Raw DEFLATE (RFC 1951) decoder used to read PNG IDAT payloads. The target
// carries its own inflate implementation rather than linking zlib, so this
// decoder is self contained as well. `input` must start at the first deflate
// bit; the returned vector holds exactly the decoded bytes.
//
// Returns false on malformed input. Trailing bytes after the final block are
// ignored, matching the target's tolerance for concatenated IDAT data.
bool inflate_raw(std::span<const std::uint8_t> input,
                 std::vector<std::uint8_t>& output);

// zlib wrapper (RFC 1950): two header bytes followed by a raw deflate stream
// and a big-endian Adler-32. Used for PNG IDAT chunks.
bool inflate_zlib(std::span<const std::uint8_t> input,
                  std::vector<std::uint8_t>& output);

} // namespace pngout
