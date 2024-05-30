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

#ifndef AICPU_UTILS_FFT_HELPER_H_
#define AICPU_UTILS_FFT_HELPER_H_
#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <numeric>
#include <functional>
#include "pocketfft_hdronly.h"

namespace aicpu {
enum NormMode : int64_t { BACKWARD = 0, FORWARD = 1, ORTHO = 2 };

int64_t GetCalculateElementNum(std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>, int64_t);

double GetNormalized(int64_t, NormMode, bool);

bool IsForwardOp(const std::string &);

template <typename S, typename T>
void Cast(const S *in, T *out) {
  if constexpr (std::is_same_v<S, T>) {
    *out = static_cast<T>(*in);
  } else if constexpr (std::is_same_v<S, bool> && std::is_same_v<T, std::complex<float>>) {
    *out = std::complex<float>(*in ? 1.0f : 0.0f, 0.0f);
  } else if constexpr (std::is_same_v<S, bool> && std::is_same_v<T, std::complex<double>>) {
    *out = std::complex<double>(*in ? 1.0 : 0.0, 0.0);
  } else if constexpr ((std::is_same_v<S, std::complex<float>>) || (std::is_same_v<S, std::complex<double>>)) {
    *out = static_cast<T>(std::real(*in));
  } else if constexpr ((std::is_same_v<T, std::complex<float>>) || (std::is_same_v<T, std::complex<double>>)) {
    double realValue = static_cast<double>(*in);
    std::complex<double> complexValue(realValue, 0.0);
    *out = (std::is_same_v<T, std::complex<float>>) ? static_cast<T>(complexValue) : complexValue;
  } else {
    *out = static_cast<T>(*in);
  }
}

template <typename T_in, typename T_out>
void ShapeCopy(T_in *input, T_out *output, const std::vector<int64_t> input_shape,
               const std::vector<int64_t> output_shape) {
  auto x_rank = input_shape.size();
  std::vector<int64_t> shape_min(x_rank, 0);
  std::vector<int64_t> input_pos(x_rank, 0);
  std::vector<int64_t> output_pos(x_rank, 0);
  for (size_t i = 0; i < x_rank; i++) {
    shape_min[i] = std::min(input_shape[i], output_shape[i]);
  }

  int64_t input_nums = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int64_t>());
  int64_t output_nums = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>());
  std::vector<int64_t> input_buffer(x_rank + 1, input_nums);
  std::vector<int64_t> output_buffer(x_rank + 1, output_nums);
  for (size_t i = 1; i < x_rank; i++) {
    input_buffer[i] = input_buffer[i - 1] / input_shape[i - 1];
    output_buffer[i] = output_buffer[i - 1] / output_shape[i - 1];
  }
  input_buffer[x_rank] = 1;
  output_buffer[x_rank] = 1;
  int64_t copy_num = std::accumulate(shape_min.begin(), shape_min.end(), 1, std::multiplies<int64_t>());
  int64_t input_index = 0;
  int64_t output_index = 0;
  for (int64_t i = 0; i < copy_num; ++i) {
    Cast(&input[input_index], &output[output_index]);

    size_t j = x_rank - 1;
    input_pos[j]++;
    input_index++;
    while (j > 0 && input_pos[j] == shape_min[j]) {
      if (input_pos[j] == output_shape[j]) {
        input_index += (input_shape[j] - output_shape[j]) * input_buffer[j + 1];
      }
      input_pos[j] = 0;
      j--;
      input_pos[j]++;
    }

    j = x_rank - 1;
    output_pos[j]++;
    output_index++;
    while (j > 0 && output_pos[j] == shape_min[j]) {
      if (output_pos[j] == input_shape[j]) {
        output_index += (output_shape[j] - input_shape[j]) * output_buffer[j + 1];
      }
      output_pos[j] = 0;
      j--;
      output_pos[j]++;
    }
  }
}

template <typename T>
void PocketFFTC2R(std::complex<T> *calculate_input, T *output_ptr, bool forward, T fct,
                  const std::vector<int64_t> &calculate_shape, const std::vector<int64_t> &dim) {
  pocketfft::shape_t shape(calculate_shape.begin(), calculate_shape.end());
  pocketfft::stride_t stride_in(shape.size());
  pocketfft::stride_t stride_out(shape.size());
  size_t tmp_in = sizeof(std::complex<T>);
  size_t tmp_out = sizeof(T);
  for (int i = shape.size() - 1; i >= 0; --i) {
    stride_in[i] = tmp_in;
    tmp_in *= shape[i];
    stride_out[i] = tmp_out;
    tmp_out *= shape[i];
  }
  pocketfft::shape_t axes;
  for (size_t i = 0; i < dim.size(); i++) {
    (void)axes.push_back(static_cast<size_t>(dim[i]));
  }
  pocketfft::c2r(shape, stride_in, stride_out, axes, forward, calculate_input, output_ptr, fct);
}

template <typename T>
void PocketFFTR2C(T *calculate_input, std::complex<T> *output_ptr, bool forward, T fct,
                  const std::vector<int64_t> &calculate_shape, const std::vector<int64_t> &dim) {
  pocketfft::shape_t shape(calculate_shape.begin(), calculate_shape.end());
  pocketfft::stride_t stride_in(shape.size());
  pocketfft::stride_t stride_out(shape.size());
  size_t tmp_in = sizeof(T);
  size_t tmp_out = sizeof(std::complex<T>);
  for (int i = shape.size() - 1; i >= 0; --i) {
    stride_in[i] = tmp_in;
    tmp_in *= shape[i];
    stride_out[i] = tmp_out;
    if (i == dim.back()) {
      tmp_out *= shape[i] / 2 + 1;
    } else {
      tmp_out *= shape[i];
    }
  }
  pocketfft::shape_t axes;
  for (size_t i = 0; i < dim.size(); i++) {
    (void)axes.push_back(static_cast<size_t>(dim[i]));
  }
  pocketfft::r2c(shape, stride_in, stride_out, axes, forward, calculate_input, output_ptr, fct);
}

template <typename T>
void PocketFFTC2C(std::complex<T> *calculate_input, std::complex<T> *output_ptr, bool forward, T fct,
                  const std::vector<int64_t> &calculate_shape, const std::vector<int64_t> &dim) {
  pocketfft::shape_t shape(calculate_shape.begin(), calculate_shape.end());
  pocketfft::stride_t stride_in(shape.size());
  pocketfft::stride_t stride_out(shape.size());
  size_t tmp_in = sizeof(std::complex<T>);
  size_t tmp_out = sizeof(std::complex<T>);
  for (int i = shape.size() - 1; i >= 0; --i) {
    stride_in[i] = tmp_in;
    tmp_in *= shape[i];
    stride_out[i] = tmp_out;
    tmp_out *= shape[i];
  }
  pocketfft::shape_t axes;
  for (size_t i = 0; i < dim.size(); i++) {
    (void)axes.push_back(static_cast<size_t>(dim[i]));
  }
  pocketfft::c2c(shape, stride_in, stride_out, axes, forward, calculate_input, output_ptr, fct);
}
}  // namespace aicpu

#endif  // AICPU_UTILS_FFT_HELPER_H_
