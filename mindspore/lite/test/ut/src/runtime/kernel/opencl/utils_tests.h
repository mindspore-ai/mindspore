/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef TESTS_UT_OPENCL_KERNEL_TESTS_UTILS_H_
#define TESTS_UT_OPENCL_KERNEL_TESTS_UTILS_H_

#include <string>
#include <iostream>
#include <vector>
#include <tuple>
#include <map>
#include "mindspore/lite/src/tensor.h"
#include "mindspore/lite/src/common/file_utils.h"

using mindspore::lite::Tensor;

namespace mindspore {

void LoadTestData(void *dst, size_t dst_size, const std::string &file_path);

template <typename T>
void CompareOutput(void *output, void *expect, size_t elem_num, T atol, float rtol = 1e-5) {
  T *output_data = reinterpret_cast<T *>(output);
  T *expect_data = reinterpret_cast<T *>(expect);

  std::cout << std::setprecision(5) << std::setiosflags(std::ios::fixed) << std::setw(7);
  std::cout << "output[0:12]:";
  for (int i = 0; i < 12 && i < elem_num; i++) {
    std::cout << output_data[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "expect[0:12]:";
  for (int i = 0; i < 12 && i < elem_num; i++) {
    std::cout << expect_data[i] << " ";
  }
  std::cout << std::endl;
  for (int i = 0; i < elem_num; ++i) {
    auto left = static_cast<float>(std::fabs(output_data[i] - expect_data[i]));
    auto right = static_cast<float>(atol + rtol * std::fabs(expect_data[i]));
    if (left > right) {
      std::cout << "error at idx[" << i << "] expect=" << expect_data[i] << " output=" << output_data[i] << std::endl;
    }
    ASSERT_LE(left, right);
  }
  std::cout << "compare success!" << std::endl;
}

template <typename T>
void CompareOutput(lite::Tensor *output_tensor, const std::string &file_path, T atol, float rtol = 1e-5) {
  size_t output_size;
  auto expect_data = mindspore::lite::ReadFile(file_path.c_str(), &output_size);
  CompareOutput(output_tensor->data_c(), expect_data, output_tensor->ElementsNum(), atol, rtol);
}

void TestMain(const std::vector<std::tuple<std::vector<int>, float *, Tensor::Category>> &input_infos,
              std::tuple<std::vector<int>, float *> output_info, OpParameter *op_parameter, bool fp16_enable = false,
              float atol = 10e-9, bool print_output = false);

}  // namespace mindspore

#endif  // TESTS_UT_OPENCL_KERNEL_TESTS_UTILS_H_
