// Tests for fp/rmsnorm.hpp.
//
// fp::rmsnorm is the golden reference, so most checks are tight: exact bits
// where both sides round the same way, otherwise relative error <= 1e-6.
// Any failed check prints file:line and exits with 1.
#include "tinyllm/fp/rmsnorm.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

#include "check.hpp"
#include "tinyllm/common/bf16.hpp"
#include "tinyllm/qwen3_config.hpp"

using tinyllm::bf16;
using tinyllm::to_bf16;
using tinyllm::to_float;
using tinyllm::fp::rmsnorm;

namespace {

const float kEps = float(tinyllm::kQwen3_0_6B.rms_eps);
const float kOne = 1.0f;

std::vector<bf16> constant_weights(int n, float value) {
    return std::vector<bf16>(n, to_bf16(value));
}

std::vector<bf16> random_weights(int n, std::uint32_t seed) {
    const std::vector<float> f = check::random_vector(n, seed, 0.5f, 1.5f);
    std::vector<bf16> w(n);
    for (int i = 0; i < n; ++i) w[i] = to_bf16(f[i]);
    return w;
}

// Independent, plain double-precision version used as the expected value.
std::vector<float> ref_rmsnorm(const std::vector<float>& x, const std::vector<bf16>& w,
                               float eps) {
    const int n = int(x.size());
    double sum = 0;
    for (int i = 0; i < n; ++i) sum += double(x[i]) * double(x[i]);
    const double scale = 1.0 / std::sqrt(sum / n + double(eps));
    std::vector<float> y(n);
    for (int i = 0; i < n; ++i) y[i] = float(double(x[i]) * scale * double(to_float(w[i])));
    return y;
}

// 1. x = 1, w = 1: every y is 1 / sqrt(1 + eps), rounded once to float.
void test_all_ones() {
    const int n = 1024;
    const std::vector<float> x(n, kOne);
    const std::vector<bf16> w = constant_weights(n, kOne);
    std::vector<float> y(n);
    rmsnorm(x.data(), w.data(), y.data(), n, kEps);

    const float want = float(1.0 / std::sqrt(1.0 + double(kEps)));
    for (int i = 0; i < n; ++i) REQUIRE_EQ_HEX(check::float_bits(y[i]), check::float_bits(want));
}

// 2. Worked by hand: x = {3, 4}, w = {1, 2}.
//    mean square = (9 + 16) / 2 = 12.5, scale = 1 / sqrt(12.5) = 0.28284271
//    y = {3 * 0.28284271 * 1, 4 * 0.28284271 * 2} = {0.84852814, 2.26274170}
void test_hand_worked() {
    const float x[] = {3.0f, 4.0f};
    const bf16 w[] = {to_bf16(1.0f), to_bf16(2.0f)};
    const float want[] = {0.84852814f, 2.26274170f};
    float y[2];
    rmsnorm(x, w, y, 2, kEps);
    REQUIRE_LE(check::max_rel_error(y, want, 2), 1e-6);
}

// 3. x = 0: eps keeps the scale finite (1 / sqrt(eps) = 1000), so y = 0, not NaN.
void test_all_zeros() {
    const int n = 128;
    const std::vector<float> x(n, 0.0f);
    const std::vector<bf16> w = random_weights(n, 3);
    std::vector<float> y(n, -1.0f);
    rmsnorm(x.data(), w.data(), y.data(), n, kEps);
    for (int i = 0; i < n; ++i) REQUIRE_EQ_HEX(check::float_bits(y[i]), 0u);
}

// 4. x = 2^70: x^2 = 2^140 overflows float (max about 2^128) but not double.
//    With the square done in double, y = 2^70 * 2^-70 = 1 exactly.
//    A float square gives inf, scale 0 and y = 0.
void test_large_values() {
    const int n = 1024;
    const std::vector<float> x(n, std::ldexp(1.0f, 70));
    const std::vector<bf16> w = constant_weights(n, kOne);
    std::vector<float> y(n);
    rmsnorm(x.data(), w.data(), y.data(), n, kEps);
    for (int i = 0; i < n; ++i) REQUIRE_EQ_HEX(check::float_bits(y[i]), check::float_bits(kOne));
}

// 5. Random x and w against ref_rmsnorm, at both widths the model uses.
void test_random_vs_reference(int n, std::uint32_t seed) {
    const std::vector<float> x = check::random_vector(n, seed, -4.0f, 4.0f);
    const std::vector<bf16> w = random_weights(n, seed + 1000);
    std::vector<float> y(n);
    rmsnorm(x.data(), w.data(), y.data(), n, kEps);

    const std::vector<float> want = ref_rmsnorm(x, w, kEps);
    REQUIRE_LE(check::max_rel_error(y.data(), want.data(), n), 1e-6);
}

// 6. With w = 1 the output's RMS is sqrt(ms / (ms + eps)), which is 1 to
//    within about 1e-7 here. A property check that doesn't reuse the formula.
void test_output_rms() {
    const int n = 1024;
    const std::vector<float> x = check::random_vector(n, 6, -4.0f, 4.0f);
    const std::vector<bf16> w = constant_weights(n, kOne);
    std::vector<float> y(n);
    rmsnorm(x.data(), w.data(), y.data(), n, kEps);

    double sum = 0;
    for (int i = 0; i < n; ++i) sum += double(y[i]) * y[i];
    const double rms = std::sqrt(sum / n);
    REQUIRE_LE(std::fabs(rms - 1.0), 1e-5);
}

// 7. In place (x == y) must give exactly the same bits as out of place.
void test_in_place() {
    const int n = 1024;
    const std::vector<float> x = check::random_vector(n, 7, -4.0f, 4.0f);
    const std::vector<bf16> w = random_weights(n, 8);

    std::vector<float> out(n);
    rmsnorm(x.data(), w.data(), out.data(), n, kEps);

    std::vector<float> buf = x;
    rmsnorm(buf.data(), w.data(), buf.data(), n, kEps);

    for (int i = 0; i < n; ++i) REQUIRE_EQ_HEX(check::float_bits(buf[i]), check::float_bits(out[i]));
}

// 8. TODO once common/npy and the reference dumps exist:
//    pos0/embed.npy with layer 0 input_layernorm.weight should give
//    pos0/L00_attn_norm.npy. MLX rounds its output to bf16, so the tolerance
//    is about 2^-8 relative, not exact.

}  // namespace

int main() {
    test_all_ones();
    test_hand_worked();
    test_all_zeros();
    test_large_values();
    test_random_vs_reference(1024, 1);  // layer norm width
    test_random_vs_reference(128, 2);   // QK-norm width (one head)
    test_output_rms();
    test_in_place();
    std::printf("test_fp_rmsnorm: passed\n");
    return 0;
}
