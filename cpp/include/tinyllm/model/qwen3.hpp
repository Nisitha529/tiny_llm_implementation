// Full decode step built from the fp kernels, following docs/model.md
// ("One decode step"). A version built from the hw kernels comes later.
#pragma once

#include "tinyllm/model/kv_cache.hpp"
#include "tinyllm/model/weights.hpp"

namespace tinyllm {
namespace fp {

// Runs `token` at position `pos`, writes its keys and values into the cache,
// and returns the greedy next token. If logits is not null, it receives the
// config.vocab output scores.
int decode_step(const ModelWeights& w, KvCache& cache, int token, int pos, float* logits);

}  // namespace fp
}  // namespace tinyllm
