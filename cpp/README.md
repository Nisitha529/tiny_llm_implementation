# cpp/: Phase 1, C++ kernels

C++ versions of every kernel in [../docs/kernels.md](../docs/kernels.md), plus
a full model that chains them. Later phases use this code as their checker:
the RTL testbenches link this library, and the Zybo's ARM runs it to check
the FPGA.

## Layout

| Path | Contents |
|---|---|
| `include/tinyllm/` | Headers. `qwen3_config.hpp` holds the model dimensions (Qwen3-0.6B and a tiny test model). |
| `src/float/` | Float kernels: plain, readable math in `float`. Checked against mlx-lm (level L1). |
| `src/hw/` | Hardware-exact kernels: the number formats, rounding and operation order the RTL will use. Their output is the bit-level spec (level L2). |
| `src/model/` | Weight loading and the full decode step, built from either kernel set. |
| `tests/` | Unit tests and comparisons against `reference/data/`. |

Each kernel has one file in each set, with the same name: `src/float/rmsnorm.cpp`
and `src/hw/rmsnorm.cpp`.

## Writing hardware-exact kernels

Keep the code close to what the RTL will do:

- Use fixed-width integer types (`int8_t`, `int32_t`, ...) and spell out
  every rounding and saturation step.
- Use simple loops with sizes from `ModelConfig`, and no dynamic allocation
  inside a kernel.
- Accumulate in the order the hardware will (for example per group of 128
  in `qmatvec`), because changing the order changes floating-point results.
- Keep those headers C++14 so Vitis HLS 2020.2 can read them, in case you
  want early resource estimates from HLS.

## Build

```bash
cmake -S cpp -B build/cpp
cmake --build build/cpp
ctest --test-dir build/cpp --output-on-failure
```

## Suggested order

1. `qmatvec` (97.5% of the work, and it fixes the weight format)
2. `rmsnorm`, `rope`, `swiglu`, `vecops`, `embedding`
3. `attention` with the KV cache
4. Full model (`src/model/`). Phase 1 is done when it produces the same
   greedy tokens as mlx-lm.
5. Activation format experiments (see [../docs/number_formats.md](../docs/number_formats.md))
