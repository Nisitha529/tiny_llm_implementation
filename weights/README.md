# weights/

Git ignores everything here except this file.

| Path | Contents | Made by |
|---|---|---|
| `Qwen3-0.6B-MLX-4bit/` | Original checkpoint (`config.json`, `model.safetensors`, tokenizer files) | `snapshot_download`, see [../reference/README.md](../reference/README.md) |
| `npy/` | One `.npy` file per tensor, read by the C++ model | `reference/scripts/export_weights.py` |
| `zybo/` | Weight image laid out for the board's DDR | Weight converter in `fpga/sw/` (Phase 3) |
