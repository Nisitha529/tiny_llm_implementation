// Checks that the model constants match the numbers in docs/model.md.
#include "tinyllm/qwen3_config.hpp"

#include <cstdio>

using tinyllm::kQwen3_0_6B;
using tinyllm::kTinyTest;
namespace quant = tinyllm::quant;

static_assert(kQwen3_0_6B.q_dim() == 2048, "query width");
static_assert(kQwen3_0_6B.kv_dim() == 1024, "key/value width");
static_assert(kQwen3_0_6B.gqa_group() == 2, "query heads per KV head");
static_assert(kQwen3_0_6B.token_weight_macs() == 595984384, "weight MACs per token");
static_assert(kQwen3_0_6B.kv_bytes_per_position() == 112 * 1024, "KV cache per position");
static_assert(quant::kBytesPerWeight == 0.53125, "4-bit format size");

// Every quantized matrix input width must split into whole groups.
static_assert(kQwen3_0_6B.hidden % quant::kGroupSize == 0, "hidden width");
static_assert(kQwen3_0_6B.q_dim() % quant::kGroupSize == 0, "o_proj input");
static_assert(kQwen3_0_6B.mlp % quant::kGroupSize == 0, "down_proj input");
static_assert(kTinyTest.hidden % quant::kGroupSize == 0, "tiny hidden width");
static_assert(kTinyTest.q_dim() % quant::kGroupSize == 0, "tiny o_proj input");
static_assert(kTinyTest.mlp % quant::kGroupSize == 0, "tiny down_proj input");

int main() {
    const double mb = kQwen3_0_6B.token_weight_macs() * quant::kBytesPerWeight / 1e6;
    std::printf("Qwen3-0.6B: %lld weight MACs/token, %.1f MB weights/token, %lld B KV/position\n",
                static_cast<long long>(kQwen3_0_6B.token_weight_macs()), mb,
                static_cast<long long>(kQwen3_0_6B.kv_bytes_per_position()));
    std::printf("Tiny test:  %lld weight MACs/token\n",
                static_cast<long long>(kTinyTest.token_weight_macs()));
    return 0;
}
