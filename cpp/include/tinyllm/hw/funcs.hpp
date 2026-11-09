// Bit-exact models of the RTL function units:
//   rsqrt    (rmsnorm)
//   exp      (softmax in attention)
//   sigmoid  (swiglu)
//   sin/cos  (rope)
// Each function here defines exactly what its RTL unit outputs for every
// input: the method (table, polynomial, CORDIC, ...), widths and rounding.
#pragma once

#include "tinyllm/hw/formats.hpp"
