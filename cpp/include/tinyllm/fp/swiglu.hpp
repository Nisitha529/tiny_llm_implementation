// Float SwiGLU. Math: docs/kernels.md, section "swiglu".
#pragma once

namespace tinyllm {
namespace fp {

// y[i] = silu(gate[i]) * up[i], with silu(z) = z / (1 + exp(-z)).
// y may be the same buffer as gate or up.
void swiglu(const float* gate, const float* up, float* y, int n);

}  // namespace fp
}  // namespace tinyllm
