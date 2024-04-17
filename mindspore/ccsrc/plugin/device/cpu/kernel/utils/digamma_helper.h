/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_UTILS_DIGAMMA_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_UTILS_DIGAMMA_HELPER_H_
#include <cmath>
#include <limits>

namespace mindspore {
namespace kernel {
template <typename T>
T Polevl(const T x, const T A[], size_t len) {
  T result = 0;
  for (size_t i = 0; i <= len; i++) {
    result = result * x + A[i];
  }
  return result;
}

template <typename T>
T CalcDigamma(T x) {
  static constexpr T kPSI_10 = 2.25175258906672110764;
  static constexpr T kPI = 3.141592653589793238462;
  static constexpr int64_t kTEN = 10;
  static constexpr T kHALF = 0.5;
  static constexpr int64_t kSIX = 6;
  if (x == 0) {
    return std::copysign(INFINITY, -x);
  }

  bool x_is_integer = x == trunc(x);
  if (x < 0) {
    if (x_is_integer) {
      return std::numeric_limits<T>::quiet_NaN();
    }
    double q;
    double r = 0;
    r = std::modf(x, &q);
    return CalcDigamma(1 - x) - kPI / tan(kPI * r);
  }

  T result = 0;
  while (x < kTEN) {
    result -= 1 / x;
    x += 1;
  }
  if (x == kTEN) {
    return result + kPSI_10;
  }

  static const T A[] = {
    8.33333333333333333333E-2, -2.10927960927960927961E-2, 7.57575757575757575758E-3, -4.16666666666666666667E-3,
    3.96825396825396825397E-3, -8.33333333333333333333E-3, 8.33333333333333333333E-2,
  };

  T y = 0;
  if (x < 1.0e17) {
    T z = 1.0 / (x * x);
    y = z * Polevl(z, A, kSIX);
  }
  return result + log(x) - (kHALF / x) - y;
}

template <>
inline Eigen::half CalcDigamma(Eigen::half x) {
  auto result = static_cast<Eigen::half>(CalcDigamma(static_cast<std::float_t>(x)));
  return result;
}
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_UTILS_DIGAMMA_HELPER_H_
