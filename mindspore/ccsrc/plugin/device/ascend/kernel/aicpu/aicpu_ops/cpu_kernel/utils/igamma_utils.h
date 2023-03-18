/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef AICPU_KERNELS_NORMALIZED_IGAMMA_UTILS_H_
#define AICPU_KERNELS_NORMALIZED_IGAMMA_UTILS_H_

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace {
/**
 * Coefficients for the Lanczos approximation of the gamma function. The
 * coefficients are uniquely determined by the choice of g and n (kLanczosGamma
 * and kLanczosCoefficients.size() + 1). The coefficients below correspond to
 * [7, 9]. [5, 7], [7, 9], [9, 10], and [607/128.0, 15] were evaluated and [7,
 * 9] seemed to be the least sensitive to the quality of the log function. In
 * particular, [5, 7] is the only choice where -1.5e-5 <= lgamma(2) <= 1.5e-5
 * for a particularly inaccurate log function.
 * */
static constexpr double kLanczosGamma = 7;  // aka g
static constexpr double kBaseLanczosCoeff = 0.99999999999980993227684700473478;
static constexpr std::array<double, 8> kLanczosCoefficients = {
  676.520368121885098567009190444019, -1259.13921672240287047156078755283,
  771.3234287776530788486528258894,   -176.61502916214059906584551354,
  12.507343278686904814458936853,     -0.13857109526572011689554707,
  9.984369578019570859563e-6,         1.50563273514931155834e-7};
double log_lanczos_gamma_plus_one_half = std::log(kLanczosGamma + 0.5);
static constexpr int VALUE = 1;
static constexpr int DERIVATIVE = 2;
static constexpr int SAMPLE_DERIVATIVE = 3;
}  // namespace

/** Compute the Lgamma function using Lanczos' approximation from "A Precision
 * Approximation of the Gamma Function". SIAM Journal on Numerical Analysis
 * series B. Vol. 1:
 * lgamma(z + 1) = (log(2) + log(pi)) / 2 + (z + 1/2) * log(t(z)) - t(z) + A(z)
 * t(z) = z + kLanczosGamma + 1/2
 * A(z) = kBaseLanczosCoeff + sigma(k = 1, n, kLanczosCoefficients[i] / (z + k))
 */
template <typename T>
T Lgamma(const T &input) {
  T log_pi = std::log(M_PI);
  T log_sqrt_two_pi = (std::log(2) + std::log(M_PI)) / 2;

  /** If the input is less than 0.5 use Euler's reflection formula:
   * gamma(x) = pi / (sin(pi * x) * gamma(1 - x))
   */
  bool need_to_reflect = (input < 0.5);
  T input_after_reflect = need_to_reflect ? -input : input - 1;  // aka z

  T sum = kBaseLanczosCoeff;  // aka x
  for (int i = 0, end = kLanczosCoefficients.size(); i < end; ++i) {
    T lanczos_coefficient = kLanczosCoefficients[i];

    sum += lanczos_coefficient / (input_after_reflect + i + 1);
  }

  /** To improve accuracy on platforms with less-precise log implementations,
   * compute log(lanczos_gamma_plus_one_half) at compile time and use log1p on
   * the device.
   * log(t) = log(kLanczosGamma + 0.5 + z)
   *        = log(kLanczosGamma + 0.5) + log1p(z / (kLanczosGamma + 0.5))
   * */
  T gamma_plus_onehalf_plus_z = kLanczosGamma + 0.5 + input_after_reflect;  // aka t

  T log_t = log_lanczos_gamma_plus_one_half + std::log1pf(input_after_reflect / (kLanczosGamma + 0.5));

  /** Compute the final result (modulo reflection).  t(z) may be large, and we
   * need to be careful not to overflow to infinity in the first term of
   *   (z + 1/2) * log(t(z)) - t(z).
   * Therefore we compute this as
   *   (z + 1/2 - t(z) / log(t(z))) * log(t(z)).
   */
  T log_y = log_sqrt_two_pi + (input_after_reflect + 0.5 - gamma_plus_onehalf_plus_z / log_t) * log_t + std::log(sum);

  /** Compute the reflected value, used when x < 0.5:
   *
   *   lgamma(x) = log(pi) - lgamma(1-x) - log(abs(sin(pi * x))).
   *
   * (The abs is because lgamma is the log of the absolute value of the gamma
   * function.)
   *
   * We have to be careful when computing the final term above. gamma(x) goes
   * to +/-inf at every integer x < 0, and this is controlled by the
   * sin(pi * x) term.  The slope is large, so precision is particularly
   * important.
   *
   * Because abs(sin(pi * x)) has period 1, we can equivalently use
   * abs(sin(pi * frac(x))), where frac(x) is the fractional part of x.  This
   * is more numerically accurate: It doesn't overflow to inf like pi * x can,
   * and if x is an integer, it evaluates to 0 exactly, which is significant
   * because we then take the log of this value, and log(0) is inf.
   *
   * We don't have a frac(x) primitive in XLA and computing it is tricky, but
   * because abs(sin(pi * x)) = abs(sin(pi * abs(x))), it's good enough for
   * our purposes to use abs(frac(x)) = abs(x) - floor(abs(x)).
   *
   * Furthermore, pi * abs(frac(x)) loses precision when abs(frac(x)) is close
   * to 1.  To remedy this, we can use the fact that sin(pi * x) in the domain
   * [0, 1] is symmetric across the line Y=0.5.
   */
  T abs_input = std::abs(input);
  T abs_frac_input = abs_input - std::floor(abs_input);

  /* Convert values of abs_frac_input > 0.5 to (1 - frac_input) to improve
   * precision of pi * abs_frac_input for values of abs_frac_input close to 1.
   */
  T reduced_frac_input = (abs_frac_input > 0.5) ? 1 - abs_frac_input : abs_frac_input;
  T reflection_denom = std::log(std::sin(M_PI * reduced_frac_input));

  /* Avoid computing -inf - inf, which is nan.  If reflection_denom is +/-inf,
   * then it "wins" and the result is +/-inf.
   */
  T reflection = std::isfinite(reflection_denom) ? log_pi - reflection_denom - log_y : -reflection_denom;

  T result = need_to_reflect ? reflection : log_y;

  return std::isinf(input) ? std::numeric_limits<T>::infinity() : result;
};

/* Compute the Digamma function using Lanczos' approximation from "A Precision
 * Approximation of the Gamma Function". SIAM Journal on Numerical Analysis
 * series B. Vol. 1:
 * digamma(z + 1) = log(t(z)) + A'(z) / A(z) - kLanczosGamma / t(z)
 * t(z) = z + kLanczosGamma + 1/2
 * A(z) = kBaseLanczosCoeff + sigma(k = 1, n, kLanczosCoefficients[i] / (z + k))
 * A'(z) = sigma(k = 1, n, kLanczosCoefficients[i] / (z + k) / (z + k))
 */
template <typename T>
T Digamma(const T &input) {
  /* If the input is less than 0.5 use Euler's reflection formula:
   * digamma(x) = digamma(1 - x) - pi * cot(pi * x)
   */
  bool need_to_reflect = (input < 0.5);
  T reflected_input = need_to_reflect ? -input : input - 1;  // aka z

  T num = 0;
  T denom = kBaseLanczosCoeff;

  for (int i = 0, end = kLanczosCoefficients.size(); i < end; ++i) {
    T lanczos_coefficient = kLanczosCoefficients[i];
    num -= lanczos_coefficient / ((reflected_input + i + 1) * (reflected_input + i + 1));
    denom += lanczos_coefficient / (reflected_input + i + 1);
  }

  /* To improve accuracy on platforms with less-precise log implementations,
   * compute log(lanczos_gamma_plus_one_half) at compile time and use log1p on
   * the device.
   * log(t) = log(kLanczosGamma + 0.5 + z)
   *        = log(kLanczosGamma + 0.5) + log1p(z / (kLanczosGamma + 0.5))
   */

  T gamma_plus_onehalf_plus_z = kLanczosGamma + 0.5 + reflected_input;  // aka t
  T log_t = log_lanczos_gamma_plus_one_half + std::log1pf(reflected_input / (kLanczosGamma + 0.5));

  T result = log_t + num / denom - kLanczosGamma / gamma_plus_onehalf_plus_z;  // aka y

  /* We need to be careful how we compute cot(pi * input) below: For
   * near-integral values of `input`, pi * input can lose precision.
   *
   * Input is already known to be less than 0.5 (otherwise we don't have to
   * reflect).  We shift values smaller than -0.5 into the range [-.5, .5] to
   * increase precision of pi * input and the resulting cotangent.
   */

  T reduced_input = input + std::abs(std::floor(input + 0.5));
  T reflection = result - M_PI * std::cos(M_PI * reduced_input) / std::sin(M_PI * reduced_input);
  T real_result = need_to_reflect ? reflection : result;

  // Digamma has poles at negative integers and zero; return nan for those.
  return (input < 0 && input == std::floor(input)) ? std::numeric_limits<T>::quiet_NaN() : real_result;
};

template <typename T>
void IgammaSeriesLoop(std::vector<T> &vals, const int &mode) {
  while (vals[0]) {
    T enabled = vals[0];
    T r = vals[1];
    T c = vals[2];
    T ans = vals[3];
    T x = vals[4];
    T dc_da = vals[5];
    T dans_da = vals[6];

    r += 1;
    dc_da = dc_da * (x / r) + (-1 * c * x) / (r * r);
    dans_da = dans_da + dc_da;
    c = c * (x / r);
    ans = ans + c;
    T conditional;
    if (mode == VALUE) {
      conditional = enabled && (c / ans > std::numeric_limits<T>::epsilon());
    } else {
      conditional = enabled && (std::abs(dc_da / dans_da) > std::numeric_limits<T>::epsilon());
    }

    vals[0] = conditional;
    if (enabled) {
      vals = {conditional, r, c, ans, x, dc_da, dans_da};
    }
  }
}

// Helper function for computing Igamma using a power series.
template <typename T>
T IgammaSeries(const T &ax, const T &x, const T &a, const T &enabled, const int &mode) {
  /* vals: (enabled, r, c, ans, x)
   * 'enabled' is a predication mask that says for which elements we should
   * execute the loop body. Disabled elements have no effect in the loop body.
   */
  std::vector<T> vals = {enabled, a, 1, 1, x, 0, 0};
  IgammaSeriesLoop<T>(vals, mode);

  T ans = vals[3];
  T dans_da = vals[6];
  auto base_num = a == 0 ? 1 : a;
  if (mode == VALUE) {
    return (ans * ax) / base_num;
  }

  T dlogax_da = std::log(x) - Digamma<T>(a + 1);
  switch (mode) {
    case DERIVATIVE:
      return ax * (ans * dlogax_da + dans_da) / base_num;
    default:
      return -(dans_da + ans * dlogax_da) * x / base_num;
  }
}

template <typename T>
void IgammacCFLoop(std::vector<T> &vals, const int &mode) {
  while (vals[0] && vals[5] < 2000) {
    T enabled = vals[0];
    T ans = vals[1];
    T tmp_var_t = vals[2];
    T tmp_var_y = vals[3];
    T tmp_var_z = vals[4];
    T tmp_var_c = vals[5];
    T pkm1 = vals[6];
    T qkm1 = vals[7];
    T pkm2 = vals[8];
    T qkm2 = vals[9];
    T dpkm2_da = vals[10];
    T dqkm2_da = vals[11];
    T dpkm1_da = vals[12];
    T dqkm1_da = vals[13];
    T dans_da = vals[14];

    tmp_var_c += 1;
    tmp_var_y += 1;
    tmp_var_z += 2;

    T yc = tmp_var_y * tmp_var_c;
    T pk = pkm1 * tmp_var_z - pkm2 * yc;
    T qk = qkm1 * tmp_var_z - qkm2 * yc;
    bool qk_is_nonzero = (qk != 0);
    T r = pk / qk;

    T t = qk_is_nonzero ? std::abs((ans - r) / r) : 1;
    ans = qk_is_nonzero ? r : ans;

    T dpk_da = dpkm1_da * tmp_var_z - pkm1 - dpkm2_da * yc + pkm2 * tmp_var_c;
    T dqk_da = dqkm1_da * tmp_var_z - qkm1 - dqkm2_da * yc + qkm2 * tmp_var_c;
    T dans_da_new = qk_is_nonzero ? (dpk_da - ans * dqk_da) / qk : dans_da;
    T grad_conditional = qk_is_nonzero ? std::abs(dans_da_new - dans_da) : 1;

    pkm2 = pkm1;
    pkm1 = pk;
    qkm2 = qkm1;
    qkm1 = qk;

    dpkm2_da = dpkm1_da;
    dqkm2_da = dqkm1_da;
    dpkm1_da = dpk_da;
    dqkm1_da = dqk_da;

    bool rescale = std::abs(pk) > (1 / std::numeric_limits<T>::epsilon());

    pkm2 = rescale ? pkm2 * std::numeric_limits<T>::epsilon() : pkm2;
    pkm1 = rescale ? pkm1 * std::numeric_limits<T>::epsilon() : pkm1;
    qkm2 = rescale ? qkm2 * std::numeric_limits<T>::epsilon() : qkm2;
    qkm1 = rescale ? qkm1 * std::numeric_limits<T>::epsilon() : qkm1;

    dpkm2_da = rescale ? dpkm2_da * std::numeric_limits<T>::epsilon() : dpkm2_da;
    dqkm2_da = rescale ? dqkm2_da * std::numeric_limits<T>::epsilon() : dqkm2_da;
    dpkm1_da = rescale ? dpkm1_da * std::numeric_limits<T>::epsilon() : dpkm1_da;
    dqkm1_da = rescale ? dqkm1_da * std::numeric_limits<T>::epsilon() : dqkm1_da;

    T conditional;

    if (mode == VALUE) {
      conditional = enabled && (t > std::numeric_limits<T>::epsilon());
    } else {
      conditional = enabled && (grad_conditional > std::numeric_limits<T>::epsilon());
    }

    vals[0] = conditional;
    vals[5] = tmp_var_c;
    if (enabled) {
      vals = {conditional, ans,  tmp_var_t, tmp_var_y, tmp_var_z, tmp_var_c, pkm1,       qkm1,
              pkm2,        qkm2, dpkm2_da,  dqkm2_da,  dpkm1_da,  dqkm1_da,  dans_da_new};
    }
  }
}

template <typename T>
T IgammacContinuedFraction(const T &ax, const T &x, const T &a, const T &enabled, const int &mode) {
  // vals: enabled, ans, t, y, z, c, pkm1, qkm1, pkm2, qkm2
  T y = 1 - a;
  T z = x + y + 1;
  T c = 0;
  T pkm2 = 1;
  T qkm2 = x;
  T pkm1 = x + 1;
  T qkm1 = z * x;
  T ans = pkm1 / qkm1;
  T t = 1;
  T dpkm2_da = 0;
  T dqkm2_da = 0;
  T dpkm1_da = 0;
  T dqkm1_da = -x;
  T dans_da = (dpkm1_da - ans * dqkm1_da) / qkm1;
  std::vector<T> vals = {enabled, ans,  t,        y,        z,        c,        pkm1,   qkm1,
                         pkm2,    qkm2, dpkm2_da, dqkm2_da, dpkm1_da, dqkm1_da, dans_da};

  IgammacCFLoop<T>(vals, mode);

  ans = vals[1];
  if (mode == VALUE) {
    return ans * ax;
  }

  dans_da = vals[14];
  T dlogax_da = std::log(x) - Digamma<T>(a);
  switch (mode) {
    case DERIVATIVE:
      return ax * (ans * dlogax_da + dans_da);
    default:
      return -(dans_da + ans * dlogax_da) * x;
  }
}

template <typename T>
T IgammaSingle(const T &a, const T &x) {
  if (!std::isinf(a) && (a > 0) && std::isinf(x) && x > 0) {
    return 1;
  }

  T is_nan = std::isnan(a) || std::isnan(x);
  T x_is_zero = (x == 0);
  T domain_error = (x < 0) || (a <= 0);
  T use_igammac = (x > 1) && (x > a);

  T ax = a * std::log(x) - x - Lgamma<T>(a);

  T underflow = (ax < -std::log(std::numeric_limits<T>::max()));

  ax = std::exp(ax);
  T enabled = !(x_is_zero || domain_error || underflow || is_nan);

  T output = use_igammac ? 1 - IgammacContinuedFraction<T>(ax, x, a, enabled && use_igammac, VALUE)
                         : IgammaSeries<T>(ax, x, a, (enabled && !(use_igammac)), VALUE);

  output = (domain_error || is_nan || std::isnan(output)) ? std::numeric_limits<double>::quiet_NaN() : output;

  output = x_is_zero ? 0 : output;
  return output;
}

template <typename T>
void Igamma(T *a, T *x, T *output, int size) {
  for (int i = 0; i < size; i++) {
    *(output + i) = IgammaSingle<T>(*(a + i), *(x + i));
  }
}

template <typename T>
T IgammacSingle(const T &a, const T &x) {
  T out_of_range = (x <= 0) || (a <= 0);
  T use_igamma = (x < 1) || (x < a);
  T ax = a * std::log(x) - x - Lgamma<double>(a);
  T underflow = (ax < -std::log(std::numeric_limits<T>::max()));

  T enabled = !(out_of_range || underflow);

  ax = std::exp(ax);
  T output = use_igamma ? 1 - IgammaSeries<T>(ax, x, a, (enabled && use_igamma), VALUE)
                        : IgammacContinuedFraction<T>(ax, x, a, enabled && !use_igamma, VALUE);

  output = out_of_range ? 1 : output;

  output = x < 0 || a <= 0 || std::isnan(x) || (std::isinf(x) && (x > 0)) || std::isnan(a)
             ? std::numeric_limits<T>::quiet_NaN()
             : output;
  output = std::isinf(x) && x > 0 && a > 0 ? 0 : output;

  return output;
}

template <typename T>
void Igammac(T *a, T *x, T *output, int size) {
  for (int i = 0; i < size; i++) {
    *(output + i) = IgammacSingle<T>(*(a + i), *(x + 1));
  }
}

template <typename T>
T IgammaGradASingle(const T &a, const T &x) {
  T is_nan = std::isnan(a) || std::isnan(x);
  T x_is_zero = (x == 0);
  T domain_error = (x < 0) || (a <= 0);
  T use_igammac = (x > 1) && (x > a);
  T ax = a * std::log(x) - x - Lgamma<T>(a);
  T underflow = (ax < -std::log(std::numeric_limits<T>::max()));
  ax = std::exp(ax);
  T enabled = !(x_is_zero || domain_error || underflow || is_nan);
  T output = use_igammac ? -IgammacContinuedFraction<T>(ax, x, a, enabled && use_igammac, DERIVATIVE)
                         : IgammaSeries<T>(ax, x, a, (enabled && !(use_igammac)), DERIVATIVE);

  output = (domain_error || is_nan || std::isnan(output)) ? std::numeric_limits<double>::quiet_NaN() : output;
  output = x_is_zero || (std::isinf(x) && !is_nan && !domain_error && !std::isinf(a)) ? 0 : output;

  return output;
}

template <typename T>
void IgammaGradA(T *a, T *x, T *output, int size) {
  for (int i = 0; i < size; i++) {
    *(output + i) = IgammaGradASingle<T>(*(a + i), *(x + i));
  }
}
#endif