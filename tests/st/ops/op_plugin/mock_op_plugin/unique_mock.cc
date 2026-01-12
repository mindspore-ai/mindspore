/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "custom_kernel_input_info.h"

using mindspore::kernel::op_plugin::KernelInputInfo;
using mindspore::kernel::op_plugin::OpPluginOutputShapeInfo;

extern "C" {

// Mock implementation of the Unique operator.
// This is a compute-depend op that updates output shape dynamically.
// Input: 1D tensor
// Output: unique values tensor, inverse indices tensor
int Unique(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream, void *extra) {
  std::cout << "op_plugin mock: Unique called" << std::endl;
  constexpr int expected_nparam = 3;  // input, output_values, output_indices
  if (nparam != expected_nparam || params == nullptr || ndims == nullptr || shapes == nullptr) {
    std::cout << "Invalid parameters for Unique operator, nparam=" << nparam << std::endl;
    return -1;
  }

  auto kernel_input_info = static_cast<KernelInputInfo *>(extra);
  if (kernel_input_info == nullptr) {
    std::cout << "Invalid kernel input info for Unique operator" << std::endl;
    return -1;
  }

  // Get input tensor info
  int input_rank = ndims[0];
  if (input_rank != 1) {
    std::cout << "Expected input_rank = 1, but got " << input_rank << std::endl;
    return -1;
  }
  int64_t input_size = shapes[0][0];

  const int32_t *input = static_cast<const int32_t *>(params[0]);
  int32_t *output_values = static_cast<int32_t *>(params[1]);
  int32_t *output_indices = static_cast<int32_t *>(params[2]);

  // Compute unique values and inverse indices
  std::unordered_map<int32_t, int32_t> value_to_index;
  std::vector<int32_t> unique_values;

  for (int64_t i = 0; i < input_size; ++i) {
    int32_t val = input[i];
    auto it = value_to_index.find(val);
    if (it == value_to_index.end()) {
      int32_t new_idx = static_cast<int32_t>(unique_values.size());
      value_to_index[val] = new_idx;
      unique_values.push_back(val);
      output_indices[i] = new_idx;
    } else {
      output_indices[i] = it->second;
    }
  }

  // Copy unique values to output
  int64_t unique_count = static_cast<int64_t>(unique_values.size());
  for (size_t i = 0; i < unique_values.size(); ++i) {
    output_values[i] = unique_values[i];
  }

  // Set dynamic output shapes
  auto *output_shape_info = kernel_input_info->GetOutputShapeInfo();
  if (output_shape_info == nullptr) {
    auto shape_info = std::make_unique<OpPluginOutputShapeInfo>();
    shape_info->output_shapes.resize(2);
    shape_info->shape_calculated.resize(2, false);
    kernel_input_info->SetOutputShapeInfo(std::move(shape_info));
    output_shape_info = kernel_input_info->GetOutputShapeInfo();
  }

  // Output 0: unique values shape = [unique_count]
  output_shape_info->output_shapes[0] = {unique_count};
  output_shape_info->shape_calculated[0] = true;

  // Output 1: inverse indices shape = [input_size]
  output_shape_info->output_shapes[1] = {input_size};
  output_shape_info->shape_calculated[1] = true;

  std::cout << "Unique: input_size=" << input_size << ", unique_count=" << unique_count << std::endl;

  return 0;
}

}  // extern "C"
