// Read-only view of one 4-bit matrix in the checkpoint format.
// Format: docs/number_formats.md, section "Weights: 4-bit, groups of 128".
//
// Implementation: src/common/qweights.cpp.
#pragma once

#include <cstdint>

#include "tinyllm/common/bf16.hpp"

namespace tinyllm {

struct QMatrix {
    const std::uint32_t* codes;  // [rows][cols / 8]; column j uses bits 4*(j%8) .. 4*(j%8)+3
    const bf16* scales;          // [rows][cols / 128]
    const bf16* biases;          // [rows][cols / 128]
    int rows;
    int cols;
};

// 4-bit code (0..15) of element (row, col).
int code_at(const QMatrix& m, int row, int col);

// Weight value q * s + b of element (row, col), computed in float.
float weight_at(const QMatrix& m, int row, int col);

}  // namespace tinyllm
