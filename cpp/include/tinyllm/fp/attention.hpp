// Float decode attention with grouped-query attention.
// Math: docs/kernels.md, section "attention".
#pragma once

namespace tinyllm {
namespace fp {

// q:       [heads][head_dim]            current query, after QK-norm and RoPE
// k_cache: [n_pos][kv_heads][head_dim]  keys for positions 0..n_pos-1, new one included
// v_cache: [n_pos][kv_heads][head_dim]  values, same layout
// out:     [heads][head_dim]
// Query head h uses KV head h / (heads / kv_heads). Scores are scaled by
// 1 / sqrt(head_dim) before the softmax.
void attention(const float* q, const float* k_cache, const float* v_cache, int n_pos,
               int heads, int kv_heads, int head_dim, float* out);

}  // namespace fp
}  // namespace tinyllm
