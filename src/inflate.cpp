#include "inflate.hpp"

#include <array>
#include <cstring>

namespace pngout {
namespace {

// Bit reader for the LSB-first deflate stream.
class BitReader {
public:
   BitReader(std::span<const std::uint8_t> input) : input_(input) {}

   [[nodiscard]] bool ok() const noexcept { return ok_; }

   // Read `count` bits (count <= 24) LSB first.
   std::uint32_t bits(unsigned count) {
      while (bit_count_ < count) {
         if (position_ >= input_.size()) {
            ok_ = false;
            return 0;
         }
         accumulator_ |= static_cast<std::uint64_t>(input_[position_++])
                         << bit_count_;
         bit_count_ += 8;
      }
      const std::uint32_t value =
         static_cast<std::uint32_t>(accumulator_ & ((std::uint64_t{1} << count) - 1));
      accumulator_ >>= count;
      bit_count_ -= count;
      return value;
   }

   void align() noexcept {
      accumulator_ >>= (bit_count_ & 7);
      bit_count_ -= (bit_count_ & 7);
   }

   [[nodiscard]] std::uint32_t peek(unsigned count) {
      while (bit_count_ < count) {
         if (position_ >= input_.size()) {
            ok_ = false;
            return 0;
         }
         accumulator_ |= static_cast<std::uint64_t>(input_[position_++])
                         << bit_count_;
         bit_count_ += 8;
      }
      return static_cast<std::uint32_t>(accumulator_ & ((std::uint64_t{1} << count) - 1));
   }

   void drop(unsigned count) noexcept {
      accumulator_ >>= count;
      bit_count_ -= count;
   }

   [[nodiscard]] std::size_t byte_position() const noexcept {
      return position_ - bit_count_ / 8;
   }

private:
   std::span<const std::uint8_t> input_;
   std::uint64_t accumulator_{};
   unsigned bit_count_{};
   std::size_t position_{};
   bool ok_{true};
};

// Canonical Huffman table built from a code-length list. Codes are read
// LSB-first, so the table is indexed by the bit-reversed code.
struct HuffmanTable {
   // counts[len] = number of codes of that length
   std::array<std::uint16_t, 16> counts{};
   // symbols ordered by code length then symbol
   std::vector<std::uint16_t> symbols;

   [[nodiscard]] bool build(std::span<const std::uint8_t> lengths) {
      counts.fill(0);
      for (std::uint8_t length : lengths) {
         if (length > 15) return false;
         if (length != 0) ++counts[length];
      }
      // Reject over-subscribed and incomplete codes the same way the target's
      // decoder does (incomplete single-symbol tables are tolerated below).
      std::int32_t left = 1;
      for (unsigned length = 1; length <= 15; ++length) {
         left <<= 1;
         left -= counts[length];
         if (left < 0) return false;
      }
      std::array<std::uint16_t, 16> offsets{};
      offsets[1] = 0;
      for (unsigned length = 1; length < 15; ++length) {
         offsets[length + 1] =
            static_cast<std::uint16_t>(offsets[length] + counts[length]);
      }
      symbols.assign(offsets[15] + counts[15], 0);
      for (std::size_t symbol = 0; symbol < lengths.size(); ++symbol) {
         const std::uint8_t length = lengths[symbol];
         if (length != 0) {
            symbols[offsets[length]++] = static_cast<std::uint16_t>(symbol);
         }
      }
      return true;
   }

   // Decode one symbol by walking the canonical table one bit at a time. The
   // slow path is deliberate: it keeps the decoder small and avoids a large
   // lookup table allocation for every block.
   [[nodiscard]] std::int32_t decode(BitReader& reader) const {
      std::int32_t code = 0;
      std::int32_t first = 0;
      std::int32_t index = 0;
      for (unsigned length = 1; length <= 15; ++length) {
         code |= static_cast<std::int32_t>(reader.bits(1));
         const std::int32_t count = counts[length];
         if (code - first < count) {
            return symbols[static_cast<std::size_t>(index + (code - first))];
         }
         index += count;
         first = (first + count) << 1;
         code <<= 1;
      }
      return -1;
   }
};

constexpr std::array<std::uint16_t, 29> kLengthBase = {
   3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
   35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
constexpr std::array<std::uint8_t, 29> kLengthExtra = {
   0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
   3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
constexpr std::array<std::uint16_t, 30> kDistanceBase = {
   1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129,
   193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097,
   6145, 8193, 12289, 16385, 24577};
constexpr std::array<std::uint8_t, 30> kDistanceExtra = {
   0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6,
   6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

bool decode_compressed(BitReader& reader, const HuffmanTable& literals,
                       const HuffmanTable& distances,
                       std::vector<std::uint8_t>& output) {
   for (;;) {
      const std::int32_t symbol = literals.decode(reader);
      if (symbol < 0) return false;
      if (symbol < 256) {
         output.push_back(static_cast<std::uint8_t>(symbol));
      } else if (symbol == 256) {
         return true;
      } else {
         const unsigned length_index = static_cast<unsigned>(symbol - 257);
         if (length_index >= kLengthBase.size()) return false;
         const std::uint32_t length =
            kLengthBase[length_index] + reader.bits(kLengthExtra[length_index]);
         const std::int32_t distance_symbol = distances.decode(reader);
         if (distance_symbol < 0 ||
             static_cast<std::size_t>(distance_symbol) >= kDistanceBase.size()) {
            return false;
         }
         const std::uint32_t distance =
            kDistanceBase[static_cast<std::size_t>(distance_symbol)] +
            reader.bits(kDistanceExtra[static_cast<std::size_t>(distance_symbol)]);
         if (distance == 0 || distance > output.size()) return false;
         const std::size_t source = output.size() - distance;
         for (std::uint32_t i = 0; i < length; ++i) {
            output.push_back(output[source + i]);
         }
      }
   }
}

bool decode_stored(BitReader& reader, std::vector<std::uint8_t>& output) {
   reader.align();
   const std::uint32_t length = reader.bits(16);
   const std::uint32_t inverse = reader.bits(16);
   if ((length ^ 0xffffu) != inverse) return false;
   for (std::uint32_t i = 0; i < length; ++i) {
      output.push_back(static_cast<std::uint8_t>(reader.bits(8)));
   }
   return reader.ok();
}

bool decode_fixed(BitReader& reader, std::vector<std::uint8_t>& output) {
   std::array<std::uint8_t, 288> literal_lengths{};
   for (std::size_t i = 0; i < 144; ++i) literal_lengths[i] = 8;
   for (std::size_t i = 144; i < 256; ++i) literal_lengths[i] = 9;
   for (std::size_t i = 256; i < 280; ++i) literal_lengths[i] = 7;
   for (std::size_t i = 280; i < 288; ++i) literal_lengths[i] = 8;
   std::array<std::uint8_t, 30> distance_lengths{};
   distance_lengths.fill(5);
   HuffmanTable literals;
   HuffmanTable distances;
   if (!literals.build(literal_lengths) || !distances.build(distance_lengths)) {
      return false;
   }
   return decode_compressed(reader, literals, distances, output);
}

bool decode_dynamic(BitReader& reader, std::vector<std::uint8_t>& output) {
   const std::uint32_t literal_count = reader.bits(5) + 257;
   const std::uint32_t distance_count = reader.bits(5) + 1;
   const std::uint32_t code_length_count = reader.bits(4) + 4;
   if (literal_count > 286 || distance_count > 30) return false;
   static constexpr std::array<std::uint8_t, 19> kOrder = {
      16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
   std::array<std::uint8_t, 19> code_length_lengths{};
   for (std::uint32_t i = 0; i < code_length_count; ++i) {
      code_length_lengths[kOrder[i]] =
         static_cast<std::uint8_t>(reader.bits(3));
   }
   HuffmanTable code_length_table;
   if (!code_length_table.build(code_length_lengths)) return false;

   std::vector<std::uint8_t> lengths;
   lengths.reserve(literal_count + distance_count);
   while (lengths.size() < literal_count + distance_count) {
      const std::int32_t symbol = code_length_table.decode(reader);
      if (symbol < 0) return false;
      if (symbol < 16) {
         lengths.push_back(static_cast<std::uint8_t>(symbol));
      } else if (symbol == 16) {
         if (lengths.empty()) return false;
         const std::uint8_t previous = lengths.back();
         const std::uint32_t repeat = reader.bits(2) + 3;
         if (lengths.size() + repeat > literal_count + distance_count) return false;
         lengths.insert(lengths.end(), repeat, previous);
      } else if (symbol == 17) {
         const std::uint32_t repeat = reader.bits(3) + 3;
         if (lengths.size() + repeat > literal_count + distance_count) return false;
         lengths.insert(lengths.end(), repeat, 0);
      } else if (symbol == 18) {
         const std::uint32_t repeat = reader.bits(7) + 11;
         if (lengths.size() + repeat > literal_count + distance_count) return false;
         lengths.insert(lengths.end(), repeat, 0);
      } else {
         return false;
      }
   }

   HuffmanTable literals;
   HuffmanTable distances;
   if (!literals.build(std::span<const std::uint8_t>(lengths.data(), literal_count))) {
      return false;
   }
   if (!distances.build(std::span<const std::uint8_t>(
          lengths.data() + literal_count, distance_count))) {
      return false;
   }
   return decode_compressed(reader, literals, distances, output);
}

std::uint32_t adler32(std::span<const std::uint8_t> data) {
   std::uint32_t a = 1;
   std::uint32_t b = 0;
   for (std::uint8_t byte : data) {
      a = (a + byte) % 65521;
      b = (b + a) % 65521;
   }
   return (b << 16) | a;
}

} // namespace

bool inflate_raw(std::span<const std::uint8_t> input,
                 std::vector<std::uint8_t>& output) {
   output.clear();
   BitReader reader(input);
   for (;;) {
      const std::uint32_t final_block = reader.bits(1);
      const std::uint32_t type = reader.bits(2);
      bool ok = false;
      if (type == 0) {
         ok = decode_stored(reader, output);
      } else if (type == 1) {
         ok = decode_fixed(reader, output);
      } else if (type == 2) {
         ok = decode_dynamic(reader, output);
      } else {
         return false;
      }
      if (!ok) return false;
      if (final_block) break;
   }
   return true;
}

bool inflate_zlib(std::span<const std::uint8_t> input,
                  std::vector<std::uint8_t>& output) {
   if (input.size() < 6) return false;
   const std::uint8_t cmf = input[0];
   const std::uint8_t flg = input[1];
   if ((cmf & 0x0f) != 8) return false; // deflate method
   if (((static_cast<unsigned>(cmf) << 8) | flg) % 31 != 0) return false;
   if (flg & 0x20) return false; // preset dictionary unsupported
   if (!inflate_raw(input.subspan(2), output)) return false;
   const std::size_t trailer = input.size() - 4;
   const std::uint32_t expected =
      (static_cast<std::uint32_t>(input[trailer]) << 24) |
      (static_cast<std::uint32_t>(input[trailer + 1]) << 16) |
      (static_cast<std::uint32_t>(input[trailer + 2]) << 8) |
      static_cast<std::uint32_t>(input[trailer + 3]);
   return adler32(output) == expected;
}

} // namespace pngout
