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

#include <string>
#include <iostream>
#include "tests/ut/cpp/common/common_test.h"
#include "utils/log_adapter.h"
#include "mindspore/lite/src/common/file_utils.h"
#include "mindspore/lite/src/runtime/opencl/opencl_runtime.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"

#ifndef TESTS_UT_OPENCL_KERNEL_TESTS_UTILS_H_
#define TESTS_UT_OPENCL_KERNEL_TESTS_UTILS_H_

namespace mindspore {

void LoadTestData(void *dst, size_t dst_size, const std::string &file_path);

template <typename T>
void CompareOutput(lite::tensor::Tensor *output_tensor, const std::string &file_path, T atol, float rtol = 1e-5) {
  T *output_data = reinterpret_cast<T *>(output_tensor->Data());
  size_t output_size = output_tensor->Size();
  T *expect_data = reinterpret_cast<T *>(mindspore::lite::ReadFile(file_path.c_str(), &output_size));

  printf("output[0:12]:");
  for (int i = 0; i < 12; i++) {
    printf("[%d]:%.3f ", i, output_data[i]);
  }
  printf("\n");
  printf("expect[0:12]:");
  for (int i = 0; i < 12; i++) {
    printf("[%d]:%.3f ", i, expect_data[i]);
  }
  printf("\n");
  for (int i = 0; i < output_tensor->ElementsNum(); ++i) {
    if (std::fabs(output_data[i] - expect_data[i]) > atol + rtol * std::fabs(expect_data[i])) {
      printf("error at idx[%d] expect=%.3f output=%.3f \n", i, expect_data[i], output_data[i]);
      return;
    }
  }
  printf("compare success!\n");
}

}  // namespace mindspore

#endif  // TESTS_UT_OPENCL_KERNEL_TESTS_UTILS_H_
