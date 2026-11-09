# Number formats

## Weights: 4-bit, groups of 128

This format is fixed by the checkpoint (`Qwen/Qwen3-0.6B-MLX-4bit`). Every
quantized matrix `[out, in]` is stored as three tensors:

| Tensor | Type | Shape | Contents |
|---|---|---|---|
| `.weight` | U32 | `[out, in/8]` | 8 unsigned 4-bit codes `q` (0…15) per word |
| `.scales` | BF16 | `[out, in/128]` | One scale `s` per group of 128 inputs |
| `.biases` | BF16 | `[out, in/128]` | One offset `b` per group |

The weight value is `w = q · s + b`. For a row of 1024 inputs that is 512
bytes of codes plus 8 × 4 bytes of scale and offset, which is 544 bytes, or
0.53125 bytes per weight.

**Code order inside each 32-bit word:** element `j` sits in bits
`[4·(j%8)+3 : 4·(j%8)]`, lowest nibble first. tiny-llm's own
`dequantize_weights` (`src/tiny_llm_ref/quantize.py`, commit `31361c3`) unpacks
with shifts 0, 4, …, 28, and its Week 2 Day 2 tests compare that against
`mx.dequantize`. That evidence comes from reading the code only, so still
check it once in Phase 0 (`reference/scripts/check_format.py`).

Norm weights (`*_layernorm`, `q_norm`, `k_norm`, `model.norm`) are plain BF16.

## BF16

1 sign bit, 8 exponent bits and 7 mantissa bits: the top 16 bits of an FP32
value. Converting BF16 to FP32 is a 16-bit shift. Converting FP32 to BF16
needs a rounding rule (round-to-nearest-even is what MLX is expected to use;
confirm it).

## Activations: open decision

The reference computes with BF16 values and FP32 accumulation. The hardware
does not have to copy that, but every choice changes the outputs slightly.
Phase 1 measures each option on real weights before RTL starts.

| Option | Datapath | Cost on the Z7-20 | Expected accuracy |
|---|---|---|---|
| BF16 activations, FP32 accumulate | Floating-point add per multiply | Highest: FP adders in LUTs | Closest to the reference |
| Shared exponent per group of 128 | Integer `q × mantissa` MACs, one alignment per group | Medium | To be measured |
| int8 activations, scale per vector | Integer `q × int8` MACs | Lowest; several MACs can share a DSP | To be measured |

What to measure for each option, compared with the float C++ model:
- Per-layer error (max absolute error and cosine similarity).
- Logit error.
- Whether greedy decoding produces the same tokens over a fixed set of prompts.

Non-linear functions (`rsqrt`, `exp`, `sin`/`cos`, `silu`) also need a chosen
method (lookup table, polynomial or CORDIC) and precision. Record those here
too.

## Decision log

| Date | Decision | Reason | Evidence |
|---|---|---|---|
| 2026-11-07 | Weights stay in the checkpoint's 4-bit format | Avoids re-quantizing; matches tiny-llm | — |
