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

#include "nnacl/fp32/stack.h"
#include "nnacl/arithmetic_common.h"

void DoStack(const float *const *inputs, size_t input_num, int *in_shape, size_t shape_size, int axis, float *output) {
  size_t one_input_size = 1;
  for (size_t i = 0; i < shape_size; ++i) {
    one_input_size *= in_shape[i];
  }
  int in_strides[4];
  ComputeStrides(in_shape, in_strides, shape_size);

  size_t copy_num = axis > 0 ? in_strides[axis - 1] : one_input_size;
  size_t copy_size = copy_num * sizeof(float);
  size_t pre_axis_count = 1;
  for (size_t i = 0; i < axis; ++i) {
    pre_axis_count *= in_shape[i];
  }
  size_t in_offset = 0;
  size_t out_offset = 0;
  for (size_t i = 0; i < pre_axis_count; ++i) {
    for (size_t j = 0; j < input_num; ++j) {
      memcpy(output + out_offset, inputs[j] + in_offset, copy_size);
      out_offset += copy_num;
    }
    in_offset += copy_num;
  }
}
