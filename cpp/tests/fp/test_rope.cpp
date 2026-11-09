// Planned checks for fp::rope:
// 1. Position 0 leaves the vector unchanged.
// 2. Every pair (i, i + head_dim/2) keeps its length.
// 3. Random inputs against a double-precision version.
// 4. Real data: L00_q.npy, after rmsnorm with q_norm and rope, gives L00_q_rope.npy.
