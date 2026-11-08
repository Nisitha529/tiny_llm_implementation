# tiny_llm_implementation

A hardware implementation of the Qwen3-0.6B language model from the
[tiny-llm](https://github.com/skyzh/tiny-llm) course, targeting the Digilent
**Zybo Z7-20** (Zynq-7020) FPGA board.

tiny-llm runs the model in Python on Apple's MLX library. This project
reimplements the same math in three steps: C++ models of each operation,
SystemVerilog RTL that matches those models bit for bit, and finally an FPGA
build on the Zybo. The FPGA logic does the heavy computation, while the Zybo's
ARM cores load weights and handle text input and output.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Reference data from mlx-lm (`reference/`) | Not started |
| 1 | C++ kernels, float and hardware-exact (`cpp/`) | Config header only |
| 2 | RTL kernels plus Verilator testbenches (`rtl/`) | Not started |
| 3 | Zybo Z7-20 build and ARM software (`fpga/`) | Not started |

## Target

| | |
|---|---|
| Model | Qwen3-0.6B, checkpoint `Qwen/Qwen3-0.6B-MLX-4bit` (4-bit weights, groups of 128, 317 MB) |
| Board | Zybo Z7-20: xc7z020, 53.2k LUTs, 220 DSPs, 140 BRAM36, 1 GB DDR3 (4.27 GB/s) |
| Speed limit | About 12.9 tokens/s, because each token reads about 331 MB from DDR (see [docs/zybo_z7_20.md](docs/zybo_z7_20.md)) |
| Decoding | One token at a time with greedy (argmax) sampling. The prompt is fed through the same path. |

## Repository layout

```
docs/         Design notes shared by all phases (start here)
reference/    Phase 0: Python scripts that save golden outputs from mlx-lm
weights/      Checkpoint and converted weight files (not tracked by git)
cpp/          Phase 1: C++ kernels and the full C++ model
  include/tinyllm/   Headers (model config, kernel interfaces)
  src/float/         Float reference kernels
  src/hw/            Hardware-exact kernels (the RTL's bit-level spec)
  src/model/         Full model that chains the kernels
  tests/             Unit tests and comparisons against reference data
rtl/          Phase 2: SystemVerilog
  src/<kernel>/      One folder per kernel
  src/common/        Shared blocks (FIFOs, BF16 helpers, AXI)
  src/top/           Accelerator top level
  tb/                Verilator testbenches (C++) that call the cpp/ checkers
fpga/         Phase 3: Zybo Z7-20
  zybo_z7_20/tcl/          Vivado project and block design scripts
  zybo_z7_20/constraints/  XDC files
  sw/                      ARM software: weight loader, driver, tokenizer
```

The same kernel names are used in every phase: `embedding`, `qmatvec`,
`rmsnorm`, `rope`, `attention`, `swiglu` and `vecops` (residual add and
argmax). [docs/kernels.md](docs/kernels.md) defines each one.

## Docs

| File | Contents |
|---|---|
| [docs/model.md](docs/model.md) | Qwen3-0.6B shapes, tensor list, one decode step, work per token |
| [docs/kernels.md](docs/kernels.md) | Exact math, inputs, outputs and calls per token for each kernel |
| [docs/number_formats.md](docs/number_formats.md) | 4-bit weight storage, BF16, activation format choices, decision log |
| [docs/verification.md](docs/verification.md) | How each phase is checked against the one before it |
| [docs/zybo_z7_20.md](docs/zybo_z7_20.md) | Board budget, ARM/FPGA split, DDR memory map |

## Tools

Tested versions on the development machine (Ubuntu 22.04, x86):

- Vivado and Vitis HLS 2020.2
- Verilator 5.006, Icarus Verilog 11.0
- g++ 11.4, CMake 3.22
- Python 3.10, with mlx 0.32.4 (CPU), mlx-lm 0.31.3 and numpy 2.2.6

## Quick start

```bash
# Phase 1 skeleton: build and run the config check
cmake -S cpp -B build/cpp
cmake --build build/cpp
ctest --test-dir build/cpp --output-on-failure
```

Phase 0 setup (Python environment and checkpoint download) is described in
[reference/README.md](reference/README.md).
