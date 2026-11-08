# fpga/: Phase 3, Zybo Z7-20

Vivado build and ARM software for the Zybo Z7-20. Board details, the
ARM/FPGA split and the DDR memory map are in
[../docs/zybo_z7_20.md](../docs/zybo_z7_20.md).

## Layout

| Path | Contents |
|---|---|
| `zybo_z7_20/tcl/` | Scripts that create the Vivado project and block design from scratch |
| `zybo_z7_20/constraints/` | XDC files (clocks, pins, timing exceptions) |
| `sw/` | ARM software (Vitis 2020.2, bare-metal first), plus the tool that turns the checkpoint into the board's weight image |

## Project from Tcl

Only the Tcl scripts and sources are tracked. The Vivado project is generated
into `zybo_z7_20/vivado/`, which git ignores:

```bash
vivado -mode batch -source fpga/zybo_z7_20/tcl/create_project.tcl   # planned
```

It uses the `zybo-z7-20` board file included with Vivado 2020.2, which sets up
the DDR, UART and SD card in the Zynq block.

## Block design (planned)

- Zynq PS7 block with the board preset.
- Accelerator (`rtl/src/top/`):
  - Control registers on AXI-Lite through `M_AXI_GP0`.
  - DDR reads and writes through the high-performance ports `S_AXI_HP0`–`HP3`.
- Fabric clock from `FCLK_CLK0`, with a 100–150 MHz target.

## ARM software (planned)

1. Read the weight image from the SD card into DDR (FAT file system through
   Xilinx's `xilffs` library).
2. Drive the accelerator through its registers, one decode step at a time.
3. UART console. At first, prompts are tokenized on the PC and sent as token
   ids. On-board tokenizing comes later.
4. The `cpp/` model built for the ARM, to check the FPGA's results bit for
   bit and to give a software-only speed baseline.

## Bring-up order

1. ARM only: hello world, DDR test, read a file from the SD card.
2. One small kernel (`rmsnorm`) behind AXI-Lite, with test data in DDR.
3. `qmatvec` reading real weights from DDR, compared with the ARM result.
4. One full layer.
5. The full model generating text.
