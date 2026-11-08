# Verification

Each level is checked against the one above it, so a mismatch points to one
specific step.

| Level | What | Checked against | Pass rule |
|---|---|---|---|
| L0 | mlx-lm running the checkpoint | — (this is the reference) | — |
| L1 | C++ float kernels and model (`cpp/src/float`, `cpp/src/model`) | L0 dumps | Per-layer tolerance, same greedy tokens |
| L2 | C++ hardware-exact kernels (`cpp/src/hw`) | L1 | Error budget from [number_formats.md](number_formats.md), same greedy tokens |
| L3 | RTL in Verilator (`rtl/`) | L2 | **Bit-exact** |
| L4 | FPGA on the Zybo (`fpga/`) | L2 running on the ARM | **Bit-exact** |

L3 and L4 must match exactly, because any difference there is a bug, not
rounding. Tolerance-based checks stop at L2.

## Reference data (L0)

`reference/scripts/` (Phase 0) runs one prompt through mlx-lm one step at a
time and saves the value at each tap point as a `.npy` file. Output goes to
`reference/data/<run>/`, which git ignores:

| File | Contents |
|---|---|
| `tokens.npy` | Prompt token ids followed by the greedy output ids |
| `pos{P}/embed.npy` | Embedding output |
| `pos{P}/L{NN}_attn_norm.npy` | After `input_layernorm` |
| `pos{P}/L{NN}_q.npy`, `_k.npy`, `_v.npy` | Projection outputs |
| `pos{P}/L{NN}_q_rope.npy`, `_k_rope.npy` | After QK-norm and RoPE |
| `pos{P}/L{NN}_attn.npy` | Attention output (2048 values) |
| `pos{P}/L{NN}_resid1.npy` | After the first residual add |
| `pos{P}/L{NN}_mlp_norm.npy` | After `post_attention_layernorm` |
| `pos{P}/L{NN}_gate.npy`, `_up.npy`, `_act.npy` | MLP projections and SwiGLU output |
| `pos{P}/L{NN}_out.npy` | Layer output |
| `pos{P}/final_norm.npy`, `logits.npy` | Final norm and logits |

`P` is the token position and `NN` the layer number (00–27). This list is a
plan; adjust it in the script if needed and update this table to match.

## Unit tests (L1, L2)

Each kernel gets:
1. A test on small hand-checkable inputs.
2. A test on random inputs against a simple float implementation.
3. A test on real tap-point data from `reference/data/`.

## RTL tests (L3)

Each Verilator testbench is C++ and links the `cpp/` library. It drives
random and real inputs into the RTL module, runs the L2 kernel on the same
inputs, and requires identical output bits. The full accelerator is first
simulated with a tiny random-weight model (`kTinyTest` in
`cpp/include/tinyllm/qwen3_config.hpp`, about 1.8M MACs per token), because
the real model is too slow to simulate.
