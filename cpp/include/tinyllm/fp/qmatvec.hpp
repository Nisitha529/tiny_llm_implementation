// Float 4-bit matrix x vector. Math: docs/kernels.md, section "qmatvec".
#pragma once

#include "tinyllm/common/qweights.hpp"

namespace tinyllm {
namespace fp {

// y[o] = sum_i W[o][i] * x[i], with W in the 4-bit format.
// x has w.cols values, y has w.rows values. x and y must not overlap.
void qmatvec(const QMatrix& w, const float* x, float* y);

}  // namespace fp
}  // namespace tinyllm
