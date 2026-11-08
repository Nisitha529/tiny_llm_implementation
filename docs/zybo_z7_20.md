# Zybo Z7-20 target

## Resources

| | Amount | Source |
|---|---|---|
| FPGA part | xc7z020clg400-1 | Vivado board file `zybo-z7-20` |
| LUTs / flip-flops | 53,200 / 106,400 | Zynq-7000 product table (DS190) |
| DSP48E1 | 220 | DS190 |
| Block RAM | 140 × 36 Kb (about 630 KB) | DS190 |
| CPU | Dual ARM Cortex-A9 | |
| DDR | 1 GB DDR3L, 32-bit, 1066 MT/s, peak 4.27 GB/s | Board file preset (MT41K256M16, 533 MHz) |
| FPGA access to DDR | 4 high-performance AXI ports (HP0–HP3), 64-bit each | |

## Speed limit

Each token reads about 331 MB at context 128 (weights plus KV cache), so the
peak bandwidth caps decoding at 4.27 GB/s ÷ 331 MB ≈ **12.9 tokens/s**. A
well-designed accelerator should reach 60–80% of that (an estimate).

To keep up with the DDR, `qmatvec` needs about 40 4-bit MACs per cycle at
150 MHz (40 × 0.53125 bytes × 150 MHz ≈ 3.2 GB/s). That fits within 220 DSPs,
so DDR bandwidth, not compute, is the bottleneck.

## ARM and FPGA split

| ARM (software, `fpga/sw/`) | FPGA logic (`rtl/`) |
|---|---|
| Boot and DDR setup (required: the DDR controller is on the ARM side) | All kernels in [kernels.md](kernels.md) |
| Load weights from the SD card into DDR | Read weights and KV cache from DDR over the HP ports |
| Tokenize the prompt, turn output ids into text | Run one decode step per command |
| Send commands through memory-mapped registers (AXI-Lite) | Report status and the next token id |

The accelerator takes commands only through AXI-Lite registers and reads DDR
by itself. Any CPU can drive it this way, so a RISC-V core could replace the
ARM later without changes to the accelerator.

## DDR memory map (draft)

| Range | Size | Use |
|---|---|---|
| `0x0000_0000`–`0x0FFF_FFFF` | 256 MB | ARM software, tokenizer tables, OS if used |
| `0x1000_0000`–`0x2FFF_FFFF` | 512 MB | Weights (317 MB used) |
| `0x3000_0000`–`0x3EFF_FFFF` | 240 MB | KV cache: up to 2,194 positions at 112 KiB each |
| `0x3F00_0000`–`0x3FFF_FFFF` | 16 MB | Activation scratch and debug dumps |

The weight layout inside its region (which tensor goes where, alignment and
interleaving across HP ports) is decided in Phase 3, together with the
weight converter in `fpga/sw/`.
