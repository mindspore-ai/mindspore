/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef AICPU_UTILS_LINALG_UTIL_H_
#define AICPU_UTILS_LINALG_UTIL_H_

#include <Eigen/Dense>
#include <cstdint>
#include <complex>

namespace aicpu {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
using float16 = Eigen::half;
using bfloat16 = Eigen::bfloat16;

template <typename T_in, typename T_out>
void Cast(const T_in *in, T_out *out) {
  *out = static_cast<T_out>(*in);
}

void Cast(const uint8_t *in, complex64 *out) {
  float realValue = static_cast<float>(*in);
  *out = complex64(realValue, 0.0f);
}

void Cast(const uint8_t *in, complex128 *out) {
  double realValue = static_cast<double>(*in);
  *out = complex128(realValue, 0.0);
}

void Cast(const uint16_t *in, complex64 *out) {
  float realValue = static_cast<float>(*in);
  *out = complex64(realValue, 0.0f);
}

void Cast(const uint16_t *in, complex128 *out) {
  double realValue = static_cast<double>(*in);
  *out = complex128(realValue, 0.0);
}

void Cast(const uint32_t *in, complex64 *out) {
  float realValue = static_cast<float>(*in);
  *out = complex64(realValue, 0.0f);
}

void Cast(const uint32_t *in, complex128 *out) {
  double realValue = static_cast<double>(*in);
  *out = complex128(realValue, 0.0);
}

void Cast(const uint64_t *in, complex64 *out) {
  float realValue = static_cast<float>(*in);
  *out = complex64(realValue, 0.0f);
}

void Cast(const uint64_t *in, complex128 *out) {
  double realValue = static_cast<double>(*in);
  *out = complex128(realValue, 0.0);
}

void Cast(const int8_t *in, complex64 *out) {
  float realValue = static_cast<float>(*in);
  *out = complex64(realValue, 0.0f);
}

void Cast(const int8_t *in, complex128 *out) {
  double realValue = static_cast<double>(*in);
  *out = complex128(realValue, 0.0);
}

void Cast(const int16_t *in, complex64 *out) {
  float realValue = static_cast<float>(*in);
  *out = complex64(realValue, 0.0f);
}

void Cast(const int16_t *in, complex128 *out) {
  double realValue = static_cast<double>(*in);
  *out = complex128(realValue, 0.0);
}

void Cast(const int32_t *in, complex64 *out) {
  float realValue = static_cast<float>(*in);
  *out = complex64(realValue, 0.0f);
}

void Cast(const int32_t *in, complex128 *out) {
  double realValue = static_cast<double>(*in);
  *out = complex128(realValue, 0.0);
}

void Cast(const int64_t *in, complex64 *out) {
  float realValue = static_cast<float>(*in);
  *out = complex64(realValue, 0.0f);
}

void Cast(const int64_t *in, complex128 *out) {
  double realValue = static_cast<double>(*in);
  *out = complex128(realValue, 0.0);
}

void Cast(const float16 *in, complex64 *out) {
  float realValue = static_cast<float>(*in);
  *out = complex64(realValue, 0.0f);
}

void Cast(const float16 *in, complex128 *out) {
  double realValue = static_cast<double>(*in);
  *out = complex128(realValue, 0.0);
}

void Cast(const float *in, complex64 *out) {
  float realValue = static_cast<float>(*in);
  *out = complex64(realValue, 0.0f);
}

void Cast(const float *in, complex128 *out) {
  double realValue = static_cast<double>(*in);
  *out = complex128(realValue, 0.0);
}

void Cast(const double *in, complex64 *out) {
  float realValue = static_cast<float>(*in);
  *out = complex64(realValue, 0.0f);
}

void Cast(const double *in, complex128 *out) {
  double realValue = static_cast<double>(*in);
  *out = complex128(realValue, 0.0);
}

void Cast(const bfloat16 *in, complex64 *out) {
  float realValue = static_cast<float>(*in);
  *out = complex64(realValue, 0.0f);
}

void Cast(const bool *in, complex64 *out) {
  float realValue = static_cast<float>(*in);
  *out = complex64(realValue, 0.0f);
}

void Cast(const bool *in, complex128 *out) {
  double realValue = static_cast<double>(*in);
  *out = complex128(realValue, 0.0);
}

void Cast(const bfloat16 *in, complex128 *out) {
  double realValue = static_cast<double>(*in);
  *out = complex128(realValue, 0.0);
}

void Cast(const complex64 *in, uint8_t *out) { *out = static_cast<uint8_t>(std::real(*in)); }

void Cast(const complex128 *in, uint8_t *out) { *out = static_cast<uint8_t>(std::real(*in)); }

void Cast(const complex64 *in, uint16_t *out) { *out = static_cast<uint16_t>(std::real(*in)); }

void Cast(const complex128 *in, uint16_t *out) { *out = static_cast<uint16_t>(std::real(*in)); }

void Cast(const complex64 *in, uint32_t *out) { *out = static_cast<uint32_t>(std::real(*in)); }

void Cast(const complex128 *in, uint32_t *out) { *out = static_cast<uint32_t>(std::real(*in)); }

void Cast(const complex64 *in, uint64_t *out) { *out = static_cast<uint64_t>(std::real(*in)); }

void Cast(const complex128 *in, uint64_t *out) { *out = static_cast<uint64_t>(std::real(*in)); }

void Cast(const complex64 *in, int8_t *out) { *out = static_cast<int8_t>(std::real(*in)); }

void Cast(const complex128 *in, int8_t *out) { *out = static_cast<int8_t>(std::real(*in)); }

void Cast(const complex64 *in, int16_t *out) { *out = static_cast<int16_t>(std::real(*in)); }

void Cast(const complex128 *in, int16_t *out) { *out = static_cast<int16_t>(std::real(*in)); }

void Cast(const complex64 *in, int32_t *out) { *out = static_cast<int32_t>(std::real(*in)); }

void Cast(const complex128 *in, int32_t *out) { *out = static_cast<int32_t>(std::real(*in)); }

void Cast(const complex64 *in, int64_t *out) { *out = static_cast<int64_t>(std::real(*in)); }

void Cast(const complex128 *in, int64_t *out) { *out = static_cast<int64_t>(std::real(*in)); }

void Cast(const complex64 *in, float16 *out) { *out = static_cast<float16>(std::real(*in)); }

void Cast(const complex128 *in, float16 *out) { *out = static_cast<float16>(std::real(*in)); }

void Cast(const complex64 *in, float *out) { *out = static_cast<float>(std::real(*in)); }

void Cast(const complex128 *in, float *out) { *out = static_cast<float>(std::real(*in)); }

void Cast(const complex64 *in, double *out) { *out = static_cast<double>(std::real(*in)); }

void Cast(const complex128 *in, double *out) { *out = static_cast<double>(std::real(*in)); }

void Cast(const complex64 *in, bool *out) { *out = static_cast<bool>(std::real(*in)); }

void Cast(const complex128 *in, bool *out) { *out = static_cast<bool>(std::real(*in)); }

void Cast(const complex64 *in, bfloat16 *out) { *out = static_cast<bfloat16>(std::real(*in)); }

void Cast(const complex128 *in, bfloat16 *out) { *out = static_cast<bfloat16>(std::real(*in)); }
}  // namespace aicpu
#endif  // AICPU_UTILS_LINALG_UTIL_H_
