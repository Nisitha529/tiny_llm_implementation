// Float KV cache for the fp model. For each layer, keys and values for
// positions 0..max_pos-1 in the layout fp::attention expects:
// [pos][kv_heads][head_dim].
#pragma once

#include <vector>

#include "tinyllm/qwen3_config.hpp"

namespace tinyllm {

struct KvCache {
    int max_pos;
    int kv_dim;                              // kv_heads * head_dim
    std::vector<std::vector<float>> k, v;    // [layer][pos * kv_dim + i]
};

KvCache make_kv_cache(const ModelConfig& config, int max_pos);

}  // namespace tinyllm
