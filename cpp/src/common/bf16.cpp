#include "tinyllm/common/bf16.hpp"

#include <cstring>

namespace tinyllm {

  float to_float (bf16 v){
    std::uint32_t intmd_val = std::uint16_t (v.bits) << 16;
    float ret_val;
  
    std::memcpy (&ret_val, &intmd_val, 4);

    return ret_val;
  }

  bf16 to_bf16 (float v){
    std::uint32_t intmd_val;
    std::memcpy (&intmd_val, &v, 4);

    if ((intmd_val & 0x7FFFFFFF) > 0x7F800000) {
      return bf16{0x7FC0};
    } else {
      intmd_val += 0x7FFF + ((intmd_val >> 16) & 1);
      return bf16{std::uint16_t(intmd_val >> 16)};
    }
  }

}