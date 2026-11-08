// Model dimensions shared by the C++ kernels, the full C++ model and the RTL
// testbenches. Values for Qwen3-0.6B come from the checkpoint
// Qwen/Qwen3-0.6B-MLX-4bit (config.json and safetensors tensor shapes).
//
// Written in C++14 style (no inline variables) so headers stay usable from
// Vitis HLS 2020.2.
#pragma once

#include <cstdint>

namespace tinyllm {

struct ModelConfig {
    int layers;
    int hidden;       // hidden_size
    int heads;        // query heads
    int kv_heads;     // key/value heads (grouped-query attention)
    int head_dim;
    int mlp;          // intermediate_size
    int vocab;
    int max_position; // max_position_embeddings
    double rope_theta;
    double rms_eps;

    constexpr int q_dim() const { return heads * head_dim; }
    constexpr int kv_dim() const { return kv_heads * head_dim; }
    constexpr int gqa_group() const { return heads / kv_heads; }

    // Multiply-accumulates in the seven projections of one layer.
    constexpr std::int64_t layer_macs() const {
        return std::int64_t(hidden) * q_dim()        // q_proj
             + 2 * std::int64_t(hidden) * kv_dim()   // k_proj, v_proj
             + std::int64_t(q_dim()) * hidden        // o_proj
             + 3 * std::int64_t(hidden) * mlp;       // gate, up, down
    }
    // Weight multiply-accumulates per token, including the tied lm_head.
    constexpr std::int64_t token_weight_macs() const {
        return layers * layer_macs() + std::int64_t(vocab) * hidden;
    }
    // KV cache bytes per context position, stored as BF16.
    constexpr std::int64_t kv_bytes_per_position() const {
        return std::int64_t(layers) * 2 * kv_dim() * 2;
    }
};

// 4-bit weight format of the checkpoint (see docs/number_formats.md).
namespace quant {
constexpr int kBits = 4;
constexpr int kGroupSize = 128;               // inputs sharing one scale and offset
constexpr int kCodesPerWord = 32 / kBits;     // 8 codes per uint32
// Bytes per weight: 4 bits of code plus BF16 scale + BF16 offset per group.
constexpr double kBytesPerWeight = kBits / 8.0 + 4.0 / kGroupSize;  // 0.53125
}  // namespace quant

//                                  layers hidden heads kv  head mlp   vocab   max_pos rope_theta rms_eps
constexpr ModelConfig kQwen3_0_6B = {28,   1024,  16,   8,  128, 3072, 151936, 65536,  1.0e6,     1.0e-6};

// Small model with the same structure, for simulating the full RTL with
// random weights. Every input width is a multiple of quant::kGroupSize.
constexpr ModelConfig kTinyTest   = {2,    256,   4,    2,  64,  768,  1000,   512,    1.0e6,     1.0e-6};

}  // namespace tinyllm
