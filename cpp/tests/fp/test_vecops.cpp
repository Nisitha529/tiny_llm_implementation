// Planned checks for fp::residual_add and fp::argmax:
// 1. residual_add on small hand-checked vectors.
// 2. argmax: a tie returns the lowest index; all-negative input; n = 151,936.
// 3. Real data: argmax of logits.npy equals the next id in tokens.npy.
