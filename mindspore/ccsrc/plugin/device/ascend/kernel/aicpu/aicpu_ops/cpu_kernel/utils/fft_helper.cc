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

#include "cpu_kernel/utils/fft_helper.h"
#include <algorithm>
#include <numeric>

namespace aicpu {

std::string GetOpName(std::string op_name) {
  if (!op_name.compare(0, op_prefix.size(), op_prefix) && op_name.find(fft_prefix) != std::string::npos) {
    op_name.erase(op_name.begin(), op_name.begin() + op_prefix.size());
  }
  return op_name;
}

bool IsForwardOp(const std::string &op_name) {
  static const std::vector<std::string> forward_op_name = {"FFT", "FFT2", "FFTN", "RFFT"};
  bool is_forward_op = std::any_of(forward_op_name.begin(), forward_op_name.end(),
                                   [&op_name](const std::string &forward_op) { return op_name == forward_op; });
  return is_forward_op;
}

int64_t GetCalculateElementNum(std::vector<int64_t> tensor_shape, std::vector<int64_t> dim, std::vector<int64_t> s,
                               int64_t input_element_nums) {
  int64_t result = input_element_nums;
  for (size_t i = 0; i < dim.size(); i++) {
    result = result / tensor_shape[dim[i]] * s[i];
  }
  return result;
}

double GetNormalized(int64_t element_nums, NormMode norm_type, bool forward) {
  double result = 1.0;
  if (forward) {
    if (norm_type == NormMode::FORWARD) {
      result = 1.0 / element_nums;
    } else if (norm_type == NormMode::ORTHO) {
      result = 1.0 / sqrt(static_cast<double>(element_nums));
    }
  } else {
    if (norm_type == NormMode::FORWARD) {
      result = 1.0 * element_nums;
    } else if (norm_type == NormMode::ORTHO) {
      result = 1.0 * sqrt(static_cast<double>(element_nums));
    }
  }
  return result;
}

}  // namespace aicpu
