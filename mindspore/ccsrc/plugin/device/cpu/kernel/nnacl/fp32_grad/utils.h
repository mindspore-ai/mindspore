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
#ifndef NNACL_FP32_GRAD_UTILS_H_
#define NNACL_FP32_GRAD_UTILS_H_

#include "nnacl/op_base.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline size_t GetInputOffset(int num_dims, const int *dims, const int *iter) {
  size_t offset = 0;
  for (int idx = 0; idx < num_dims; ++idx) {
    offset = offset * (size_t)(dims[idx]) + (size_t)(iter[idx]);
  }

  return offset;
}

static inline size_t GetOutputOffset(int num_dims, const int *dims, const int *iter, int num_axis, const int *axes) {
  size_t offset = 0;
  for (int idx = 0; idx < num_dims; ++idx) {
    // if we need to skip this axis
    int is_axis = 0;
    for (int axis_idx = 0; axis_idx < num_axis; ++axis_idx) {
      if (idx == axes[axis_idx]) {
        is_axis = 1;
        break;
      }
    }

    if (is_axis == 0) {
      offset = offset * (size_t)(dims[idx]) + (size_t)(iter[idx]);
    }
  }
  return offset;
}

static inline int NextIndex(int num_dims, const int *dims, int *current) {
  int carry = 1;
  for (int idx = num_dims - 1; idx >= 0; --idx) {
    int current_val = current[idx] + carry;
    if (dims[idx] == current_val) {
      current[idx] = 0;
    } else {
      current[idx] = current_val;
      carry = 0;
      break;
    }
  }
  return (carry == 0);
}

#ifdef __cplusplus
}
#endif

#endif  // NNACL_FP32_GRAD_UTILS_H_
