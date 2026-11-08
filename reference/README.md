# reference/ — Phase 0: golden data

This folder produces the reference (L0 in
[../docs/verification.md](../docs/verification.md)). It runs the official
mlx-lm library on the CPU and saves the values every later phase is checked
against.

## Setup

```bash
python3 -m venv reference/.venv
reference/.venv/bin/pip install -r reference/requirements.txt
```

The pinned versions installed and ran on this machine's Linux CPU. MLX's
Linux build is CPU-only, which is enough here.

## Checkpoint

Download `Qwen/Qwen3-0.6B-MLX-4bit` (about 317 MB) into `weights/`:

```bash
reference/.venv/bin/python -c "from huggingface_hub import snapshot_download; \
snapshot_download('Qwen/Qwen3-0.6B-MLX-4bit', local_dir='weights/Qwen3-0.6B-MLX-4bit')"
```

`huggingface_hub` is installed as a dependency of mlx-lm. Re-running the
command resumes an interrupted download, which helps on a slow connection.

## Scripts (to write)

| Script | Purpose |
|---|---|
| `scripts/check_format.py` | Confirm the 4-bit packing order and BF16 rounding (see [../docs/number_formats.md](../docs/number_formats.md)) |
| `scripts/dump_reference.py` | Run a prompt step by step through mlx-lm's own modules and save every tap point to `reference/data/<run>/` |
| `scripts/generate.py` | Plain greedy generation, to record the expected output tokens |

Phase 0 is done when `dump_reference.py` writes a full set of files for a
short prompt and the greedy tokens from `generate.py` match the `tokens.npy`
it saved.
