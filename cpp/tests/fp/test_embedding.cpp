// Planned checks for fp::embedding:
// 1. Hand-built 2 x 128 table with known codes, scales and biases: exact values.
// 2. Random table: matches weight_at for every column.
// 3. Real data: the first prompt token gives pos0/embed.npy.
