# Kernels

These are the operations the hardware must perform, with the same names in
`cpp/`, `rtl/src/` and the docs. Shapes are for Qwen3-0.6B. The math is
written in real numbers. The bit-level behaviour of each kernel (formats,
rounding, order of operations) is set by its hardware-exact C++ version in
`cpp/src/hw/` and recorded in [number_formats.md](number_formats.md).

## Summary

| Kernel | Calls per token | Size per call | Share of multiplies |
|---|---|---|---|
| `qmatvec` | 197 (7 per layer + `lm_head`) | Up to 151,936 × 1024 | 97.5% |
| `attention` | 28 (one per layer, all 16 heads) | Context length × 128 per head | 2.4% at context 128 |
| `rmsnorm` | 57 of width 1024, 672 of width 128 | 1024 or 128 | 0.07% |
| `rope` | 672 (24 heads × 28 layers) | 128 | 0.03% |
| `swiglu` | 28 | 3072 | 0.03% |
| `embedding` | 1 | One row of 1024 | — |
| `vecops` | 56 residual adds, 1 argmax | 1024, or 151,936 for argmax | — |

## `qmatvec`: 4-bit matrix × vector

`y[o] = Σ_i W[o][i] · x[i]`, where each weight is stored as a 4-bit code `q`
with a scale `s` and offset `b` shared by its group of 128 inputs:
`W[o][i] = q[o][i] · s[o][g] + b[o][g]`, with `g = i / 128`.

Each group can be computed as `s · Σ q·x + b · Σ x`. The integer sum `Σ q·x`
is the main datapath, and `Σ x` for a group is the same for every output row,
so it is computed once per input vector.

| Use | out × in | Per layer |
|---|---|---|
| `q_proj` | 2048 × 1024 | 1 |
| `k_proj`, `v_proj` | 1024 × 1024 | 2 |
| `o_proj` | 1024 × 2048 | 1 |
| `gate_proj`, `up_proj` | 3072 × 1024 | 2 |
| `down_proj` | 1024 × 3072 | 1 |
| `lm_head` (tied to embedding) | 151,936 × 1024 | once per token |

On the chip, one `qmatvec` engine serves all of these. Weights stream from
DDR, and its throughput sets the token rate.

## `rmsnorm`

`y[i] = x[i] / sqrt(mean(x²) + ε) · w[i]`, with ε = 1e-6 and `w` a BF16
vector. Qwen3 multiplies by `w` directly (not `1 + w`).

- Width 1024: before attention, before the MLP, and once at the end.
- Width 128 (QK-norm): on each query head and each key head before RoPE,
  with the `q_norm` / `k_norm` weights.

It needs one reciprocal square root per call.

## `rope`: rotary position embedding

For position `p` and a 128-wide head vector, Qwen3 uses the split-half layout.
For `i` in 0…63, element `i` is paired with element `i + 64`:

```
f_i = θ^(-2i/128),   θ = 1e6
a   = p · f_i
y[i]      = x[i]·cos(a) − x[i+64]·sin(a)
y[i + 64] = x[i+64]·cos(a) + x[i]·sin(a)
```

`cos`/`sin` can come from a table indexed by `(p, i)`, or be updated from
position to position. That choice is open.

## `attention`

For layer `L`, at position `p`, for each query head `h` (16 heads) using KV
head `h / 2`:

```
s[j]   = (q_h · K[j]) / sqrt(128)     for j = 0..p
w      = softmax(s)                    (subtract max, exp, divide by sum)
out_h  = Σ_j w[j] · V[j]
```

Before this, the step's new `k` and `v` are written to the KV cache at
position `p`. In decode, every past position is visible, so no mask is
needed. The output is the 16 heads joined into 2048 values. The cache stays
in DDR: 28 layers × 2 × 8 heads × 128 × 2 bytes = 112 KiB per position.

## `swiglu`

`y[i] = silu(g[i]) · u[i]`, with `silu(z) = z / (1 + e^(−z))`, width 3072.

## `embedding`

Dequantize row `t` of the 151,936 × 1024 table: `x[i] = q·s + b`. It uses the
same 4-bit format as `qmatvec`, so the RTL can share the dequantize logic.

## `vecops`

- Residual add: `x = x + y`, width 1024, twice per layer.
- Argmax over the 151,936 logits. It can run while `lm_head` produces its
  outputs, so the logits never need storing.
