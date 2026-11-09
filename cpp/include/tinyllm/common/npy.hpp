// Minimal .npy reader and writer for weights (weights/npy/) and reference
// data (reference/data/). Supports little-endian, C-order arrays.
// BF16 tensors are stored as "<u2" (uint16) holding the raw BF16 bits.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tinyllm {

struct NpyArray {
    std::string dtype;                // numpy type string, e.g. "<f4", "<u2", "<u4"
    std::vector<std::int64_t> shape;
    std::vector<std::uint8_t> data;   // raw little-endian bytes
};

// Throws std::runtime_error if the file is missing or malformed.
NpyArray read_npy(const std::string& path);

void write_npy(const std::string& path, const NpyArray& array);

}  // namespace tinyllm
