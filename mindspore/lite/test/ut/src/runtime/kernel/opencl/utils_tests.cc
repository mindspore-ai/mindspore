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
#include "utils/log_adapter.h"
#include "mindspore/lite/src/common/file_utils.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "mindspore/lite/test/ut/src/runtime/kernel/opencl/utils_tests.h"

namespace mindspore {

void LoadTestData(void *dst, size_t dst_size, const std::string &file_path) {
  if (file_path.empty()) {
    memset(dst, 0x00, dst_size);
  } else {
    auto src_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(file_path.c_str(), &dst_size));
    if (src_data != nullptr) {
      memcpy(dst, src_data, dst_size);
    } else {
      MS_LOG(ERROR) << "read file empty.";
    }
  }
}

void CompareOutput(lite::tensor::Tensor *output_tensor, const std::string &file_path) {
  float *output_data = reinterpret_cast<float *>(output_tensor->Data());
  size_t output_size = output_tensor->Size();
  float *expect_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(file_path.c_str(), &output_size));

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

  constexpr float atol = 1e-5;
  for (int i = 0; i < output_tensor->ElementsNum(); ++i) {
    if (std::fabs(output_data[i] - expect_data[i]) > atol) {
      printf("error at idx[%d] expect=%.3f output=%.3f \n", i, expect_data[i], output_data[i]);
      printf("error at idx[%d] expect=%.3f output=%.3f \n", i, expect_data[i], output_data[i]);
      printf("error at idx[%d] expect=%.3f output=%.3f \n", i, expect_data[i], output_data[i]);
      return;
    }
  }
  printf("compare success!\n");
  printf("compare success!\n");
  printf("compare success!\n\n\n");
}

}  // namespace mindspore
