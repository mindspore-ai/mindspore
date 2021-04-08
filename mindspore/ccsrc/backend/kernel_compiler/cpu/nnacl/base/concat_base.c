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

#include "nnacl/base/concat_base.h"

void Concat(void **input, int input_num, int axis, int **inputs_output_shape, size_t shape_size, void *output,
            int task_id, int thread_num, int data_size) {
  int before_axis_size = 1;
  for (int i = 0; i < axis; ++i) {
    before_axis_size *= inputs_output_shape[0][i];
  }

  int after_axis_size = data_size;
  for (size_t i = axis + 1; i < shape_size; ++i) {
    after_axis_size *= inputs_output_shape[0][i];
  }
  int axis_offset = 0;
  uint8_t *dst_base = (output);
  size_t output_stride = after_axis_size * inputs_output_shape[input_num][axis];
  for (int i = 0; i < input_num; ++i) {
    const uint8_t *src_base = (input[i]);
    size_t input_stride = after_axis_size * inputs_output_shape[i][axis];
    int offset = UP_DIV(input_stride, thread_num);
    int count = input_stride - offset * task_id;
    if (count <= 0) {
      axis_offset += inputs_output_shape[i][axis];
      continue;
    }
    count = MSMIN(offset, count);
    for (int j = 0; j < before_axis_size; j++) {
      const uint8_t *src = src_base + j * input_stride + task_id * offset;
      uint8_t *dst = dst_base + j * output_stride + axis_offset * after_axis_size + task_id * offset;
      memcpy(dst, src, count);
    }
    axis_offset += inputs_output_shape[i][axis];
  }
}
