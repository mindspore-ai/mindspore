/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <algorithm>

#include "custom_kernel_input_info.h"

using mindspore::kernel::op_plugin::KernelInputInfo;

extern "C" {

// Mock implementation of the sum_ext operator.
// Test the case that tuple input is the second argument.
int SumExt(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream, void *extra) {
  std::cout << "op_plugin mock: SumExt called" << std::endl;
  auto kernel_input_info = static_cast<KernelInputInfo *>(extra);
  if (kernel_input_info == nullptr) {
    std::cout << "Invalid kernel input info for sum_ext operator" << std::endl;
    return -1;
  }
  const std::vector<int64_t> expected_dim = {0, 1};
  const auto dim = kernel_input_info->GetIntVecInput(1);
  if (dim != expected_dim) {
    std::cout << "dim value is not the same as expected." << std::endl;
    return -1;
  }
  return 0;
}

}  // extern "C"
