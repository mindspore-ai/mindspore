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
#include <cstring>

#include "custom_kernel_input_info.h"

using mindspore::kernel::op_plugin::KernelInputInfo;

extern "C" {

// Mock implementation of the stack operator.
// Test the case when the input is tuple tensor
// Only supports 1D tensors, dim=0, and float32 dtype.
int StackExt(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream, void *extra) {
  std::cout << "op_plugin mock: StackExt called" << std::endl;

  if (nparam < 2 || params == nullptr || ndims == nullptr || shapes == nullptr) {
    std::cout << "Invalid parameters for stack operator" << std::endl;
    return -1;
  }

  // Get axis parameter from extra
  auto kernel_input_info = static_cast<KernelInputInfo *>(extra);
  if (kernel_input_info == nullptr) {
    std::cout << "Invalid kernel input info for stack operator" << std::endl;
    return -1;
  }

  size_t total_inputs = kernel_input_info->GetInputSize();
  if (total_inputs == 0) {
    std::cout << "No inputs found for stack operator" << std::endl;
    return -1;
  }

  int64_t dim = kernel_input_info->GetIntInput(total_inputs - 1);

  int num_inputs = static_cast<int>(total_inputs) - 1;

  if (num_inputs <= 0) {
    std::cout << "Invalid number of input tensors: " << num_inputs << std::endl;
    return -1;
  }

  if (dim != 0) {
    std::cout << "Expected dim = 0, but got " << dim << std::endl;
    return -1;
  }

  // Validate all inputs are 1D tensors with the same size
  int64_t input_size = shapes[0][0];
  if (input_size <= 0) {
    std::cout << "Invalid shape for input 0" << std::endl;
    return -1;
  }

  for (int i = 0; i < num_inputs; ++i) {
    if (ndims[i] != 1) {
      std::cout << "Expected 1D tensor for input " << i << ", but got " << ndims[i] << "D" << std::endl;
      return -1;
    }
    if (shapes[i][0] != input_size) {
      std::cout << "All input tensors must have the same size. Input 0 has size " << input_size << ", but input " << i
                << " has size " << shapes[i][0] << std::endl;
      return -1;
    }
  }

  // Output tensor is the last parameter in params array
  // For stack: 1D inputs [N] become 2D output [num_inputs, N] when dim=0
  int out_idx = nparam - 1;
  if (ndims[out_idx] != 2) {
    std::cout << "Expected 2D tensor for output, but got " << ndims[out_idx] << "D" << std::endl;
    return -1;
  }

  // Validate output shape: [num_inputs, input_size]
  if (shapes[out_idx][0] != num_inputs || shapes[out_idx][1] != input_size) {
    std::cout << "Output shape mismatch: expected [" << num_inputs << ", " << input_size << "], but got ["
              << shapes[out_idx][0] << ", " << shapes[out_idx][1] << "]" << std::endl;
    return -1;
  }

  // Check that all inputs and output are float32
  constexpr const char *expected_dtype = "float32";
  for (int i = 0; i < nparam - 2; ++i) {
    if (std::strcmp(dtypes[i], expected_dtype) != 0) {
      std::cout << "Expected float32 dtype, but got " << dtypes[i] << std::endl;
      return -1;
    }
  }

  // Perform stacking along dim=0 for 1D tensors
  // Stack creates a new dimension: [N] -> [num_inputs, N]
  // Each input tensor becomes a row in the output
  float *out_ptr = static_cast<float *>(params[out_idx]);

  for (int i = 0; i < num_inputs; ++i) {
    const float *in_ptr = static_cast<const float *>(params[i]);

    // Copy each input tensor to the corresponding row in the output
    // Output is row-major: out[i][j] = in[i][j]
    for (int64_t j = 0; j < input_size; ++j) {
      out_ptr[i * input_size + j] = in_ptr[j];
    }
  }

  return 0;
}

}  // extern "C"
