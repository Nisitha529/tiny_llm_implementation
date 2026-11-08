cmake_minimum_required(VERSION 3.16)
project(tinyllm_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Header-only for now. When kernels are added under src/float, src/hw and
# src/model, change this to a STATIC library and list the sources.
add_library(tinyllm INTERFACE)
target_include_directories(tinyllm INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(tinyllm INTERFACE -Wall -Wextra -Wpedantic)

enable_testing()

add_executable(test_config tests/test_config.cpp)
target_link_libraries(test_config PRIVATE tinyllm)
add_test(NAME config COMMAND test_config)
