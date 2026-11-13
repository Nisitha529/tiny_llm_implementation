// Shared test helpers (planned):
//   REQUIRE(cond)                 print file:line and exit(1) if cond is false
//   max_abs_error(a, b, n)
//   max_rel_error(a, b, n)
//   cosine_similarity(a, b, n)
//   random_vector(n, seed)        repeatable random inputs
//   ref_path(name)                path to a file in reference/data/
#pragma once

#include <cstdio>
#include <cstdlib>

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
