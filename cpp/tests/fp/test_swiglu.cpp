// Planned checks for fp::swiglu:
// 1. silu(0) = 0; silu(z) is close to z for large positive z and to 0 for large negative z.
// 2. Random inputs against a double-precision version.
// 3. Real data: L00_gate.npy and L00_up.npy give L00_act.npy.
