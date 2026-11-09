// Float rotary position embedding. Math: docs/kernels.md, section "rope".
#pragma once

namespace tinyllm {
namespace fp {

// Rotates `heads` vectors of head_dim values, stored back to back, in place
// for position pos. Split-half layout: element i pairs with i + head_dim/2.
void rope(float* x, int heads, int head_dim, int pos, double theta);

}  // namespace fp
}  // namespace tinyllm
