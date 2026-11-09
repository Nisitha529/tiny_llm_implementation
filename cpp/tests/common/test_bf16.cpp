// Planned checks for common/bf16.hpp:
// 1. to_float(to_bf16(x)) == x for all 65,536 bf16 bit patterns (NaN stays NaN).
// 2. to_bf16 rounds to nearest even on values exactly halfway between two bf16s.
// 3. Infinity, zero and negative zero keep their bit patterns.
