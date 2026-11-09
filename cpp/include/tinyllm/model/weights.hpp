// All Qwen3 tensors in memory, loaded from weights/npy/ (written by
// reference/scripts/export_weights.py). Tensor list: docs/model.md.
//
// Planned layout; change it freely while writing the loader.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tinyllm/common/bf16.hpp"
#include "tinyllm/common/qweights.hpp"
#include "tinyllm/qwen3_config.hpp"

namespace tinyllm {

struct LayerWeights {
    std::vector<bf16> input_norm;      // input_layernorm.weight [hidden]
    std::vector<bf16> post_attn_norm;  // post_attention_layernorm.weight [hidden]
    std::vector<bf16> q_norm;          // self_attn.q_norm.weight [head_dim]
    std::vector<bf16> k_norm;          // self_attn.k_norm.weight [head_dim]
    QMatrix q_proj, k_proj, v_proj, o_proj;
    QMatrix gate_proj, up_proj, down_proj;
};

struct ModelWeights {
    ModelConfig config;
    QMatrix embed_tokens;              // also the output layer (tied embeddings)
    std::vector<bf16> final_norm;      // model.norm.weight [hidden]
    std::vector<LayerWeights> layers;
    // Owns the bytes that the QMatrix views point into.
    std::vector<std::vector<std::uint8_t>> storage;
};

// Throws std::runtime_error if a tensor is missing or has the wrong shape.
ModelWeights load_weights(const std::string& npy_dir, const ModelConfig& config);

}  // namespace tinyllm
