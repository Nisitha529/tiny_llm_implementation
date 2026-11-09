// Hardware-exact qmatvec.
//
// Same math as fp/qmatvec.hpp, but with the number formats in hw/formats.hpp,
// the function units in hw/funcs.hpp and the RTL's order of operations.
// Its output, bit for bit, is the specification for rtl/src/qmatvec/.
//
// Declare the function here once the activation format is chosen
// (docs/number_formats.md).
#pragma once

#include "tinyllm/hw/formats.hpp"
