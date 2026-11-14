// Float RMSNorm. Math: docs/kernels.md, section "rmsnorm".
#pragma once

#include "tinyllm/common/bf16.hpp"

namespace tinyllm {
  namespace fp {

    // y[i] = x[i] / sqrt(mean(x^2) + eps) * w[i], for i in 0..n-1.
    // Used at width 1024 (layer norms) and 128 (QK-norm on each head).
    // x and y may be the same buffer.
    void rmsnorm(const float* x, const bf16* w, float* y, int x_len, float eps);

  } // namespace fp
}   // namespace tinyllm
