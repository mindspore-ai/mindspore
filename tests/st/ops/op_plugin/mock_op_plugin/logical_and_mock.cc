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

#include "custom_kernel_input_info.h"
#include "bool_tensor_iterator.h"

using mindspore::kernel::op_plugin::KernelInputInfo;

extern "C" {
// Mock implementation of the logical_and operator.
// Test cases:
// 1. when there's a existing cpu kernelmod for logical_and.
// 2. non-contiguous input
int LogicalAnd(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
               void *extra) {
  std::cout << "op_plugin mock: LogicalAnd called" << std::endl;
  constexpr int expected_nparam = 3;
  if (nparam != expected_nparam || params == nullptr || ndims == nullptr || shapes == nullptr) {
    std::cout << "Invalid parameters for LogicalAnd operator" << std::endl;
    return -1;
  }

  const bool *x = static_cast<const bool *>(params[0]);
  const bool *y = static_cast<const bool *>(params[1]);
  bool *out = static_cast<bool *>(params[2]);

  int dims = ndims[0];
  if (dims < 0) {
    std::cout << "Invalid dims for LogicalAnd operator: " << dims << std::endl;
    return -1;
  }
  if (ndims[1] != dims || ndims[2] != dims) {
    std::cout << "Invalid ndims for LogicalAnd operator" << std::endl;
    return -1;
  }

  size_t numel = 1;
  for (int i = 0; i < dims; ++i) {
    int64_t d0 = shapes[0][i];
    int64_t d1 = shapes[1][i];
    int64_t d2 = shapes[2][i];
    if (d0 <= 0 || d1 <= 0 || d2 <= 0) {
      std::cout << "Invalid shapes for LogicalAnd operator: d0 <= 0 || d1 <= 0 || d2 <= 0" << std::endl;
      return -1;
    }
    if (d0 != d1 || d0 != d2) {
      std::cout << "Invalid shapes for LogicalAnd operator: d0 != d1 || d0 != d2" << std::endl;
      return -1;
    }
    numel *= static_cast<size_t>(d0);
  }

  // Extract tensor layout (strides, storage_offset) from extra if available.
  auto kernel_input_info = static_cast<KernelInputInfo *>(extra);
  std::vector<int64_t> shape_vec;
  shape_vec.reserve(static_cast<size_t>(dims));
  for (int i = 0; i < dims; ++i) {
    shape_vec.push_back(shapes[0][i]);
  }

  auto make_elem_strides = [&](size_t input_index) -> std::pair<std::vector<int64_t>, int64_t> {
    std::vector<int64_t> strides_elems(shape_vec.size(), 0);
    int64_t offset_elems = 0;
    if (kernel_input_info != nullptr) {
      auto layout_opt = kernel_input_info->GetInputTensorLayout(input_index);
      if (layout_opt.has_value()) {
        const auto &layout = layout_opt.value();
        if (layout.strides.size() == shape_vec.size()) {
          for (size_t i = 0; i < layout.strides.size(); ++i) {
            strides_elems[i] = layout.strides[i];
          }
          offset_elems = static_cast<int64_t>(layout.storage_offset);
          return {strides_elems, offset_elems};
        }
      }
    }
    // Fallback to contiguous layout in elements (row-major).
    if (!shape_vec.empty()) {
      strides_elems.back() = 1;
      for (int64_t d = static_cast<int64_t>(shape_vec.size()) - 2; d >= 0; --d) {
        strides_elems[static_cast<size_t>(d)] =
          strides_elems[static_cast<size_t>(d + 1)] * shape_vec[static_cast<size_t>(d + 1)];
      }
    }
    return {strides_elems, 0};
  };

  auto [x_strides_elems, x_offset_elems] = make_elem_strides(0);
  auto [y_strides_elems, y_offset_elems] = make_elem_strides(1);

  BoolTensorIterator it_x(x, shape_vec, x_strides_elems, x_offset_elems);
  BoolTensorIterator it_y(y, shape_vec, y_strides_elems, y_offset_elems);

  for (size_t i = 0; i < numel; ++i) {
    const bool vx = it_x.next();
    const bool vy = it_y.next();
    out[i] = (vx || vy);  // keep mocked behavior (intentional OR)
  }
  return 0;
}

}  // extern "C"
