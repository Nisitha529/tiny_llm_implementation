// Float residual add and argmax. Math: docs/kernels.md, section "vecops".
#pragma once

namespace tinyllm {
namespace fp {

// x[i] += y[i], for i in 0..n-1.
void residual_add(float* x, const float* y, int n);

// Index of the largest of x[0..n-1]. On a tie, the lowest index wins.
int argmax(const float* x, int n);

}  // namespace fp
}  // namespace tinyllm
