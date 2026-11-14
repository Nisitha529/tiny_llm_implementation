cmake_minimum_required(VERSION 3.16)
project(tinyllm_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------- library
set(TINYLLM_COMMON_SOURCES
  src/common/bf16.cpp  
  src/common/npy.cpp
  src/common/qweights.cpp
)
set(TINYLLM_FP_SOURCES
  src/fp/rmsnorm.cpp
  src/fp/embedding.cpp
  src/fp/qmatvec.cpp
  src/fp/vecops.cpp
  src/fp/swiglu.cpp
  src/fp/rope.cpp
  src/fp/attention.cpp
)
set(TINYLLM_HW_SOURCES
  src/hw/funcs.cpp
  src/hw/rmsnorm.cpp
  src/hw/embedding.cpp
  src/hw/qmatvec.cpp
  src/hw/vecops.cpp
  src/hw/swiglu.cpp
  src/hw/rope.cpp
  src/hw/attention.cpp
)
set(TINYLLM_MODEL_SOURCES
  src/model/weights.cpp
  src/model/kv_cache.cpp
  src/model/qwen3.cpp
)

add_library(tinyllm STATIC
  ${TINYLLM_COMMON_SOURCES}
  ${TINYLLM_FP_SOURCES}
  ${TINYLLM_HW_SOURCES}
  ${TINYLLM_MODEL_SOURCES}
)
target_include_directories(tinyllm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(tinyllm PUBLIC -Wall -Wextra -Wpedantic)

# ---------------------------------------------------------------- tests
enable_testing()

# tinyllm_test(<name> <source>): build one test program and register it with ctest.
function(tinyllm_test name source)
  add_executable(${name} ${source} ${CMAKE_CURRENT_SOURCE_DIR}/tests/check.cpp)
  target_link_libraries(${name} PRIVATE tinyllm)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

tinyllm_test(test_config tests/test_config.cpp)

# Uncomment each line once its test file has a main() and real checks.
# common
tinyllm_test(test_bf16       tests/common/test_bf16.cpp)
# tinyllm_test(test_npy        tests/common/test_npy.cpp)
# tinyllm_test(test_qweights   tests/common/test_qweights.cpp)
# fp kernels
tinyllm_test(test_fp_rmsnorm   tests/fp/test_rmsnorm.cpp)
# tinyllm_test(test_fp_embedding tests/fp/test_embedding.cpp)
# tinyllm_test(test_fp_qmatvec   tests/fp/test_qmatvec.cpp)
# tinyllm_test(test_fp_vecops    tests/fp/test_vecops.cpp)
# tinyllm_test(test_fp_swiglu    tests/fp/test_swiglu.cpp)
# tinyllm_test(test_fp_rope      tests/fp/test_rope.cpp)
# tinyllm_test(test_fp_attention tests/fp/test_attention.cpp)
# hw kernels
# tinyllm_test(test_hw_funcs     tests/hw/test_funcs.cpp)
# tinyllm_test(test_hw_rmsnorm   tests/hw/test_rmsnorm.cpp)
# tinyllm_test(test_hw_embedding tests/hw/test_embedding.cpp)
# tinyllm_test(test_hw_qmatvec   tests/hw/test_qmatvec.cpp)
# tinyllm_test(test_hw_vecops    tests/hw/test_vecops.cpp)
# tinyllm_test(test_hw_swiglu    tests/hw/test_swiglu.cpp)
# tinyllm_test(test_hw_rope      tests/hw/test_rope.cpp)
# tinyllm_test(test_hw_attention tests/hw/test_attention.cpp)
# model
# tinyllm_test(test_layer      tests/model/test_layer.cpp)
# tinyllm_test(test_generate   tests/model/test_generate.cpp)

# ---------------------------------------------------------------- tools
# Uncomment once written.
# add_executable(run_model tools/run_model.cpp)
# target_link_libraries(run_model PRIVATE tinyllm)
# add_executable(format_sweep tools/format_sweep.cpp)
# target_link_libraries(format_sweep PRIVATE tinyllm)
