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

namespace aicpu {
const std::string op_prefix = "Cust";
const std::string fft_prefix = "FFT";
enum NormMode : int64_t { BACKWARD = 0, FORWARD = 1, ORTHO = 2 };

int64_t GetCalculateElementNum(std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>, int64_t);

std::string GetOpName(std::string);

double GetNormalized(int64_t, NormMode, bool);

bool IsForwardOp(const std::string &);

template <typename T_in, typename T_out>
bool ShapeCopy(T_in *input, T_out *output, const std::vector<int64_t> input_shape,
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
    output[output_index] = static_cast<T_out>(input[input_index]);
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
  return true;
}
}  // namespace aicpu

#endif  // AICPU_UTILS_FFT_HELPER_H_
