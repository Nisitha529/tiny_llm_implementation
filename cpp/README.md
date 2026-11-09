# cpp/: Phase 1, C++ kernels

C++ versions of every kernel in [../docs/kernels.md](../docs/kernels.md), plus
a full model that chains them. Later phases use this code as their checker:
the RTL testbenches link this library, and the Zybo's ARM runs it to check
the FPGA.

## Layout

```
cpp/
├── CMakeLists.txt              library `tinyllm` + one ctest per test file
├── include/tinyllm/
│   ├── qwen3_config.hpp        model sizes (Qwen3-0.6B and a tiny test model)
│   ├── common/                 bf16.hpp, npy.hpp, qweights.hpp (4-bit matrix view)
│   ├── fp/                     float kernels, namespace tinyllm::fp
│   ├── hw/                     hardware-exact kernels, namespace tinyllm::hw
│   │                           + formats.hpp (number formats), funcs.hpp (rsqrt, exp, sigmoid, sin/cos)
│   └── model/                  weights.hpp, kv_cache.hpp, qwen3.hpp (decode step)
├── src/                        one .cpp per header above (common/, fp/, hw/, model/)
├── tests/
│   ├── check.hpp               shared test helpers
│   ├── common/ fp/ hw/ model/  one test file per header
│   └── test_config.cpp
└── tools/                      run_model.cpp, format_sweep.cpp
```

Kernel names are the same in `fp/`, `hw/` and `rtl/src/`: `rmsnorm`,
`embedding`, `qmatvec`, `vecops`, `swiglu`, `rope`, `attention`.

| Set | Contents | Checked against |
|---|---|---|
| `fp/` | Plain, readable math in `float` | mlx-lm reference data (level L1) |
| `hw/` | The number formats, rounding and operation order the RTL will use. Its output is the bit-level spec for the RTL. | `fp/`, within the error budget (level L2) |

The folder is `fp`, not `float`, because `float` is a C++ keyword and can't
be a namespace name.

## Current state

Every file exists. Headers in `common/`, `fp/` and `model/` declare the planned
interfaces. `hw/` headers are placeholders until the activation format is
chosen. Source files contain only a TODO, and each test file lists its planned
checks as comments.

## Rules

- **Kernels look like RTL modules.** They take raw pointers and sizes, write
  into buffers the caller provides, and never allocate memory.
- **No external libraries.** That means no downloads, and the same files can
  be built bare-metal for the ARM later.
- **Headers stay C++14,** so Vitis HLS 2020.2 can read them. That means
  `namespace a { namespace b {`, not `namespace a::b`. Sources may use C++17.
- **Write `hw/` kernels in hardware style:**
  - Use fixed-width integer types, and spell out every rounding and
    saturation step.
  - Accumulate in the order the hardware will, for example per group of 128
    in `qmatvec`, because changing the order changes floating-point results.

## Adding a kernel or test

1. Implement the function in `src/<set>/<kernel>.cpp`. The library already
   compiles every source file.
2. Write the test in `tests/<set>/test_<kernel>.cpp` with a `main()` that
   returns non-zero on failure.
3. Uncomment its `tinyllm_test(...)` line in `CMakeLists.txt`.

## Build

```bash
cmake -S cpp -B build/cpp
cmake --build build/cpp
ctest --test-dir build/cpp --output-on-failure
```

## Order

1. `common/bf16` and `tests/check.hpp`
2. `fp/rmsnorm`, the easiest kernel, which exercises the test setup
3. `common/npy` and `common/qweights`, then `fp/embedding`, which settles the
   4-bit format on one row
4. `fp/qmatvec`, 97.5% of the work
5. `fp/vecops`, `fp/swiglu`, `fp/rope`
6. `fp/attention` and `model/kv_cache`
7. `model/weights` and `model/qwen3`. Phase 1's float part is done when
   `test_generate` produces the same greedy tokens as mlx-lm.
8. `tools/format_sweep`, then `hw/formats`, `hw/funcs` and the `hw/` kernels
