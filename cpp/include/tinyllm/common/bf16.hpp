// BF16 storage type: the top 16 bits of an IEEE FP32 value.
// Format notes: docs/number_formats.md, section "BF16".
//
// Implement at src/common/bf16.hpp
#pragma once

#include <cstdint>

namespace tinyllm {

  struct bf16 {
    std::uint16_t bits;
  };

  // Exact: places the 16 bits in the top half of an FP32 value.
  float to_float(bf16 v);

  // Rounds to nearest, ties to even. NaN stays NaN.
  bf16 to_bf16(float v);

}  // namespace tinyllm
