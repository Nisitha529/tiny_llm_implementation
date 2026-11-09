// Planned checks for fp::rmsnorm:
// 1. x all ones, w all ones: every y = 1 / sqrt(1 + eps).
// 2. Random x and w at widths 1024 and 128, against a double-precision version.
// 3. In place (x == y) gives the same result.
// 4. Real data: pos0/embed.npy with layer 0 input_layernorm gives L00_attn_norm.npy.
