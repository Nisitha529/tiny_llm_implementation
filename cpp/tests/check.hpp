// Shared test helpers. Definitions: tests/check.cpp.
//
// Macros (they must be macros so they can print __FILE__ and __LINE__):
//   REQUIRE(cond)               fail if cond is false
//   REQUIRE_EQ_HEX(got, want)   fail if two integers / bit patterns differ
//   REQUIRE_LE(got, limit)      fail if got > limit (or got is NaN)
// Each failure prints file:line and exits with 1.
//
// Still planned:
//   cosine_similarity(a, b, n)
//   ref_path(name)              path to a file in reference/data/
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define REQUIRE(cond) do { \
  if (!(cond)) { \
    std::fprintf (stderr, "%s:%d: REQUIRE (%s) failed.\n", __FILE__, __LINE__, #cond); \
    std::exit (1); \
  } \
}  while (0)

#define REQUIRE_EQ_HEX(got, want) do { \
  unsigned long g_ = (got), w_ = (want); \
  if (g_ != w_) { \
    std::fprintf (stderr, "%s:%d: got 0x%lX, want 0x%lX\n", __FILE__, __LINE__, g_, w_); \
    std::exit (1); \
  } \
} while (0)

// Written as !(g_ <= l_) so that a NaN error also fails.
#define REQUIRE_LE(got, limit) do { \
  double g_ = (got), l_ = (limit); \
  if (!(g_ <= l_)) { \
    std::fprintf (stderr, "%s:%d: %s = %g, limit %g\n", __FILE__, __LINE__, #got, g_, l_); \
    std::exit (1); \
  } \
} while (0)

namespace check {

// Largest |a[i] - b[i]|.
double max_abs_error(const float* a, const float* b, int n);

// Largest |a[i] - b[i]| / max(|b[i]|, floor). b is the reference.
double max_rel_error(const float* a, const float* b, int n, double floor = 1e-6);

// n repeatable uniform values in [lo, hi). Uses its own xorshift32 generator,
// so the numbers are the same on every compiler and machine.
std::vector<float> random_vector(int n, std::uint32_t seed, float lo, float hi);

// The 32 bits of a float.
std::uint32_t float_bits(float f);

}  // namespace check
