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

#include "nnacl/fp32/concat.h"
#include <string.h>

void Concat(void **input, int input_num, int axis, int **inputs_output_shape, size_t shape_size, void *output) {
  int before_axis_size = 1;
  for (int i = 0; i < axis; ++i) {
    before_axis_size *= inputs_output_shape[0][i];
  }
  // sizeof float/int32
  int after_axis_size = 4;
  for (size_t i = axis + 1; i < shape_size; ++i) {
    after_axis_size *= inputs_output_shape[0][i];
  }
  int axis_offset = 0;
  uint8_t *dst_base = (output);
  size_t output_stride = after_axis_size * inputs_output_shape[input_num][axis];
  for (int i = 0; i < input_num; ++i) {
    uint8_t *src_base = (input[i]);
    size_t input_stride = after_axis_size * inputs_output_shape[i][axis];
    for (int j = 0; j < before_axis_size; ++j) {
      uint8_t *src = src_base + j * input_stride;
      uint8_t *dst = dst_base + j * output_stride + axis_offset * after_axis_size;
      memcpy(dst, src, input_stride);
    }
    axis_offset += inputs_output_shape[i][axis];
  }
}
