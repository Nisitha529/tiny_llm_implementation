// Planned checks for fp::qmatvec:
// 1. Hand-built 2 x 128 matrix: exact result.
// 2. Random matrices (cols 128..3072) against a double-precision dot product
//    over weight_at.
// 3. Real data: layer 0 q_proj on L00_attn_norm.npy gives L00_q.npy.
