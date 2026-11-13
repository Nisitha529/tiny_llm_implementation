// Tests for common/bf16.hpp (to_float and to_bf16).
//
// Every input and expected value is written as a bit pattern, never as a
// decimal literal, because a literal like 3.14159f is itself already rounded.
// Any failed check prints file:line plus the got/want values and exits with 1.
#include "tinyllm/common/bf16.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "check.hpp"

using tinyllm::bf16;
using tinyllm::to_bf16;
using tinyllm::to_float;

namespace {

float from_bits(std::uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

std::uint32_t bits_of(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

// True for an FP32 NaN: exponent all ones and a non-zero mantissa.
bool is_nan_bits(std::uint32_t u) {
    return (u & 0x7FFFFFFF) > 0x7F800000;
}

struct Case {
    std::uint32_t in;
    std::uint32_t want;
};

// to_float is exact: the 16 BF16 bits become the top half of the FP32 value.
void test_to_float() {
    const Case cases[] = {
        {0x3F80, 0x3F800000},  // 1.0
        {0xBF80, 0xBF800000},  // -1.0
        {0x4049, 0x40490000},  // 3.140625
        {0x7F7F, 0x7F7F0000},  // largest finite BF16 (3.38953139e38)
        {0x0001, 0x00010000},  // smallest BF16 (9.18354962e-41)
        {0x0000, 0x00000000},  // +0.0
        {0x8000, 0x80000000},  // -0.0
        {0x7F80, 0x7F800000},  // +infinity
        {0xFF80, 0xFF800000},  // -infinity
    };
    for (const Case& c : cases) {
        REQUIRE_EQ_HEX(bits_of(to_float(bf16{std::uint16_t(c.in)})), c.want);
    }
}

// to_bf16 rounds the dropped low 16 bits: to nearest, ties to even.
void test_to_bf16() {
    const Case cases[] = {
        {0x3F800000, 0x3F80},  // 1.0, exact
        {0x40490FD0, 0x4049},  // 3.14159, rounds down
        {0x3F807FFF, 0x3F80},  // just below halfway: down
        {0x3F808001, 0x3F81},  // just above halfway: up
        {0x3F808000, 0x3F80},  // exactly halfway, 0x3F80 is even: down
        {0x3F818000, 0x3F82},  // exactly halfway, 0x3F81 is odd: up to even
        {0xBF808000, 0xBF80},  // negative tie
        {0x3FFFFFFF, 0x4000},  // carry out of the mantissa into the exponent
        {0x7F7F7FFF, 0x7F7F},  // near the maximum, stays finite
        {0x7F7FFFFF, 0x7F80},  // largest FP32 overflows to +infinity
        {0x7F800000, 0x7F80},  // +infinity
        {0xFF800000, 0xFF80},  // -infinity
        {0x00000000, 0x0000},  // +0.0
        {0x80000000, 0x8000},  // -0.0 keeps its sign
        {0x00000001, 0x0000},  // tiny value rounds to zero
        {0x00018000, 0x0002},  // tiny value, tie to even
        {0x80000001, 0x8000},  // tiny negative value rounds to -0.0
    };
    for (const Case& c : cases) {
        REQUIRE_EQ_HEX(to_bf16(from_bits(c.in)).bits, c.want);
    }
}

// Every NaN becomes the single quiet NaN 0x7FC0. Without the NaN check, the
// rounding add would turn 0x7F800001 into infinity and 0x7FFFFFFF into -0.0.
void test_nan() {
    const std::uint32_t inputs[] = {
        0x7F800001,  // smallest NaN mantissa
        0x7FFFFFFF,  // largest NaN mantissa
        0x7FC00000,  // standard quiet NaN
        0xFFC00000,  // negative quiet NaN
        0xFF800001,  // negative NaN, smallest mantissa
    };
    for (std::uint32_t in : inputs) {
        REQUIRE_EQ_HEX(to_bf16(from_bits(in)).bits, 0x7FC0u);
    }
}

// All 65,536 BF16 patterns: converting to float and back must give the same
// bits, except NaNs, which must give 0x7FC0.
void test_round_trip_all() {
    for (std::uint32_t b = 0; b < 65536; ++b) {
        const std::uint16_t out = to_bf16(to_float(bf16{std::uint16_t(b)})).bits;
        const std::uint32_t want = is_nan_bits(b << 16) ? 0x7FC0u : b;
        REQUIRE_EQ_HEX(out, want);
    }
}

}  // namespace

int main() {
    test_to_float();
    test_to_bf16();
    test_nan();
    test_round_trip_all();
    std::printf("test_bf16: passed\n");
    return 0;
}
