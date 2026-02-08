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
#include <cstring>

#include "custom_kernel_input_info.h"

using mindspore::kernel::op_plugin::KernelInputInfo;

extern "C" {

// Mock implementation of the add_scalar operator.
// Output is always filled with -1.0f to ensure mock op is called.
// Only supports float32 dtype.
int AddScalar(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream, void *extra) {
  std::cout << "op_plugin mock: AddScalar called" << std::endl;
  constexpr int expected_nparam = 4;
  if (nparam != expected_nparam || params == nullptr || ndims == nullptr || shapes == nullptr || dtypes == nullptr) {
    std::cout << "Invalid parameters for AddScalar operator" << std::endl;
    return -1;
  }

  // Check that output dtype is float32
  constexpr const char *expected_dtype = "float32";
  if (std::strcmp(dtypes[1], expected_dtype) != 0) {
    std::cout << "Expected float32 dtype for output, but got " << dtypes[1] << std::endl;
    return -1;
  }

  // Get output tensor
  float *out = static_cast<float *>(params[nparam - 1]);
  int out_dims = ndims[nparam - 1];
  if (out_dims < 0) {
    std::cout << "Invalid dims for AddScalar output: " << out_dims << std::endl;
    return -1;
  }

  // Calculate number of elements in output
  size_t numel = 1;
  for (int i = 0; i < out_dims; ++i) {
    int64_t d = shapes[nparam - 1][i];
    if (d <= 0) {
      std::cout << "Invalid shape for AddScalar output at dim " << i << ": " << d << std::endl;
      return -1;
    }
    numel *= static_cast<size_t>(d);
  }

  // Fill output with -1.0f
  for (size_t i = 0; i < numel; ++i) {
    out[i] = -1.0f;
  }

  return 0;
}

}  // extern "C"
