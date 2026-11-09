// Float embedding lookup. Math: docs/kernels.md, section "embedding".
#pragma once

#include "tinyllm/common/qweights.hpp"

namespace tinyllm {
namespace fp {

// y[i] = dequantized table[token][i], for i in 0..table.cols-1.
void embedding(const QMatrix& table, int token, float* y);

}  // namespace fp
}  // namespace tinyllm
