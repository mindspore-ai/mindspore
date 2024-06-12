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
bool IsForwardOp(const std::string &op_name) {
  static const std::vector<std::string> forward_op_name = {"FFT",  "FFT2",  "FFTN",  "RFFT", "RFFT2", "RFFTN",
                                                           "HFFT", "HFFT2", "HFFTN", "DCT",  "DCTN"};
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
  if (norm_type == NormMode::ORTHO) {
    result = 1.0 / sqrt(static_cast<double>(element_nums));
  }
  if (forward && norm_type == NormMode::FORWARD) {
    result = 1.0 / element_nums;
  }
  if (!forward && norm_type == NormMode::BACKWARD) {
    result = 1.0 / element_nums;
  }
  return result;
}

}  // namespace aicpu
