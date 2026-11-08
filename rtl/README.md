# rtl/: Phase 2, SystemVerilog kernels

RTL versions of the kernels in [../docs/kernels.md](../docs/kernels.md). Each
must produce exactly the same bits as its hardware-exact C++ version in
`cpp/src/hw/` (level L3 in [../docs/verification.md](../docs/verification.md)).

## Layout

| Path | Contents |
|---|---|
| `src/<kernel>/` | One folder per kernel: `embedding`, `qmatvec`, `rmsnorm`, `rope`, `attention`, `swiglu`, `vecops` |
| `src/common/` | Shared blocks: FIFOs, BF16 helpers, reciprocal/exp units, AXI master and AXI-Lite slave |
| `src/top/` | Accelerator top level: control registers, sequencing of one decode step, shared `qmatvec` engine |
| `tb/` | Verilator testbenches in C++, one per kernel (`tb_<kernel>.cpp`) |

## Conventions

- **Language:** SystemVerilog (`.sv`), one module per file, with the file
  named after the module. Module names start with `tl_` (for example
  `tl_rmsnorm`).
- **Clock and reset:** `clk`, plus `rst`, which is synchronous and active
  high. AXI's active-low `aresetn` is converted only at the top level.
- **Data streams:** `valid` / `ready` handshakes. A transfer happens on a
  clock edge where both are high.
- **Sizes:** module parameters. Defaults are Qwen3-0.6B; testbenches can
  set smaller sizes (`kTinyTest`).
- **Portable code:** kernel and common modules use no Xilinx IP or
  primitives. Memories and DSPs are inferred, so the same files run in
  Verilator, Icarus and Vivado. Xilinx IP (Zynq PS, AXI interconnect) lives
  only in `fpga/`.
- **Lint:** every module passes `verilator --lint-only -Wall`.

## Testbenches

Each testbench is a C++ program that:
1. builds the RTL module with Verilator,
2. links the `cpp/` library (`tinyllm`),
3. feeds the same inputs to the RTL and the `cpp/src/hw` kernel (random
   inputs first, then real data from `reference/data/`),
4. fails on the first output bit that differs.

A `Makefile` here will build and run them (planned: `make -C rtl test
KERNEL=rmsnorm`).

## Suggested order

Follow the C++ order in [../cpp/README.md](../cpp/README.md): start with
`qmatvec`, then the small kernels, then `attention`. Finish with `src/top/`
simulated on the tiny random-weight model.
