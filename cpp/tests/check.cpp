// Shared test helpers. Interface: tests/check.hpp.
#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace check {

double max_abs_error(const float* a, const float* b, int n) {
    double worst = 0;
    for (int i = 0; i < n; ++i) {
        const double e = std::fabs(double(a[i]) - double(b[i]));
        if (!(e <= worst)) worst = e;  // keeps a NaN once one appears
    }
    return worst;
}

double max_rel_error(const float* a, const float* b, int n, double floor) {
    double worst = 0;
    for (int i = 0; i < n; ++i) {
        const double denom = std::max(std::fabs(double(b[i])), floor);
        const double e = std::fabs(double(a[i]) - double(b[i])) / denom;
        if (!(e <= worst)) worst = e;
    }
    return worst;
}

std::vector<float> random_vector(int n, std::uint32_t seed, float lo, float hi) {
    std::uint32_t s = seed ? seed : 1;  // xorshift32 must not start at 0
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        const double u = double(s >> 8) / 16777216.0;  // 24 bits -> [0, 1)
        v[i] = float(lo + (double(hi) - lo) * u);
    }
    return v;
}

std::uint32_t float_bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

}  // namespace check
