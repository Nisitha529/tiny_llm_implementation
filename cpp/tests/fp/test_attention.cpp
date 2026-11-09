// Planned checks for fp::attention:
// 1. One position: each output head equals its V vector.
// 2. Two positions with equal keys: output is the average of the two V vectors.
// 3. Grouped-query attention: query heads 2h and 2h+1 read KV head h.
// 4. Random inputs against a double-precision version.
// Real data needs the KV cache of earlier positions, so it is covered by the model tests.
