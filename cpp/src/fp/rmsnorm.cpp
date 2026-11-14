// Float rmsnorm. Interface: include/tinyllm/fp/rmsnorm.hpp. Math: docs/kernels.md.
#include "tinyllm/fp/rmsnorm.hpp"

#include <cmath>

namespace tinyllm {
  namespace fp {
  
    void rmsnorm (const float* x, const bf16* w, float* y, int x_len, float eps) {
      double sum     = 0;
       
      for (int i = 0; i < x_len; i++) {
        sum += double(x[i]) * x[i];
      }

      const double inv_val = 1 / std::sqrt (sum / x_len + eps);

      for (int i = 0; i < x_len; i++) {
        y[i] = float(x[i] * inv_val * to_float (w[i]));
      }
    }
  }
}
