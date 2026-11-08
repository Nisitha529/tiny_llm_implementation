# Qwen3-0.6B model

This is the model tiny-llm uses (studied at tiny-llm commit `17a8cc6`). The
values below come from the checkpoint `Qwen/Qwen3-0.6B-MLX-4bit`, specifically
its `config.json` and the tensor shapes in `model.safetensors`.

## Configuration

| Parameter | Value | Notes |
|---|---|---|
| Layers | 28 | |
| Hidden size | 1024 | |
| Query heads | 16 | |
| KV heads | 8 | Each KV head is shared by 2 query heads (grouped-query attention) |
| Head size | 128 | Query width is 16 × 128 = 2048, not 1024 |
| MLP size | 3072 | |
| Vocabulary | 151,936 | |
| Tied embeddings | Yes | The output layer reuses the embedding table |
| RoPE θ | 1,000,000 | |
| RMSNorm ε | 1e-6 | |
| Max position | 65,536 | Limited in practice by KV cache memory |
| Weights | 4-bit, groups of 128 | See [number_formats.md](number_formats.md) |

## Tensors

The checkpoint has one copy of each tensor below per layer (`N` = 0…27). For a
quantized matrix `[out, in]`, `.weight` is `U32 [out, in/8]`, and `.scales`
and `.biases` are each `BF16 [out, in/128]`.

| Tensor | Shape (out × in) | Quantized |
|---|---|---|
| `model.embed_tokens` | 151,936 × 1024 | Yes |
| `model.layers.N.input_layernorm.weight` | 1024 | No (BF16) |
| `model.layers.N.self_attn.q_proj` | 2048 × 1024 | Yes |
| `model.layers.N.self_attn.k_proj` | 1024 × 1024 | Yes |
| `model.layers.N.self_attn.v_proj` | 1024 × 1024 | Yes |
| `model.layers.N.self_attn.q_norm.weight` | 128 | No (BF16) |
| `model.layers.N.self_attn.k_norm.weight` | 128 | No (BF16) |
| `model.layers.N.self_attn.o_proj` | 1024 × 2048 | Yes |
| `model.layers.N.post_attention_layernorm.weight` | 1024 | No (BF16) |
| `model.layers.N.mlp.gate_proj` | 3072 × 1024 | Yes |
| `model.layers.N.mlp.up_proj` | 3072 × 1024 | Yes |
| `model.layers.N.mlp.down_proj` | 1024 × 3072 | Yes |
| `model.norm.weight` | 1024 | No (BF16) |

None of the projections has a bias vector. The `.biases` tensors belong to
the 4-bit format (the per-group offset).

## One decode step

Input: token id `t` at position `p`. Output: the next token id.

```
x = embedding(t)                                  # [1024]
for layer in 0..27:
    # attention block
    h = rmsnorm(x, input_layernorm)               # [1024]
    q = qmatvec(q_proj, h)                        # [2048] = 16 heads x 128
    k = qmatvec(k_proj, h)                        # [1024] =  8 heads x 128
    v = qmatvec(v_proj, h)                        # [1024]
    q[i] = rope(rmsnorm(q[i], q_norm), p)  for each of 16 heads
    k[j] = rope(rmsnorm(k[j], k_norm), p)  for each of 8 heads
    kv_cache[layer][p] = (k, v)
    a = attention(q, kv_cache[layer][0..p])       # [2048]
    x = x + qmatvec(o_proj, a)                    # [1024]

    # MLP block
    h = rmsnorm(x, post_attention_layernorm)
    g = qmatvec(gate_proj, h)                     # [3072]
    u = qmatvec(up_proj, h)                       # [3072]
    x = x + qmatvec(down_proj, swiglu(g, u))      # [1024]

x = rmsnorm(x, model.norm)
logits = qmatvec(embed_tokens, x)                 # [151,936], tied weights
next = argmax(logits)
```

## Work per token

| Item | Amount |
|---|---|
| Weight multiply-accumulates | 595,984,384 (`lm_head` alone is 26.1%) |
| Attention multiply-accumulates | About 14.7M at context 128, growing linearly with context |
| Weight bytes read | 316.6 MB (0.53125 bytes per weight) |
| KV cache | 112 KiB per context token (BF16), read in full each step |
| Total DDR traffic at context 128 | About 331 MB |

Of all multiplies at context 128, `qmatvec` does 97.5%, attention 2.4%, and
everything else under 0.2%. At context 1,024, attention's share grows to 16.5%.
