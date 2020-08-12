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

#include "nnacl/fp32_grad/reduce_grad.h"

static inline bool NextIndex(const int num_dims, const int *dims, int *current) {
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

static inline size_t GetInputOffset(const int num_dims, const int *dims, const int *iter) {
  size_t offset = 0;
  for (int idx = 0; idx < num_dims; ++idx) {
    offset = offset * (size_t)(dims[idx]) + (size_t)(iter[idx]);
  }

  return offset;
}

static inline size_t GetOutputOffset(const int num_dims, const int *dims, const int *iter, const int num_axis,
                                     const int *axes) {
  size_t offset = 0;
  for (int idx = 0; idx < num_dims; ++idx) {
    // if we need to skip this axis
    bool is_axis = false;
    for (int axis_idx = 0; axis_idx < num_axis; ++axis_idx) {
      if (idx == axes[axis_idx]) {
        is_axis = true;
        break;
      }
    }

    if (!is_axis) {
      offset = offset * (size_t)(dims[idx]) + (size_t)(iter[idx]);
    }
  }
  return offset;
}

void ReduceMeanByAxes(const float *input_data, int *input_iter, const int *input_dims, int input_num_dims,
                      const int *axes, int num_axes, float *output_data, const int *output_dims, int output_num_dims) {
  size_t num_outputs = 1;
  for (int idx = 0; idx < output_num_dims; ++idx) {
    size_t current = (size_t)(output_dims[idx]);
    num_outputs *= current;
  }

  // Reset input iterator.
  for (int idx = 0; idx < input_num_dims; ++idx) {
    input_iter[idx] = 0;
  }
  // Iterate through input_data.
  do {
    size_t input_offset = GetInputOffset(input_num_dims, input_dims, input_iter);
    size_t output_offset = GetOutputOffset(input_num_dims, input_dims, input_iter, num_axes, axes);
    output_data[output_offset] += input_data[input_offset];
  } while (NextIndex(input_num_dims, input_dims, input_iter));

  // Calculate mean by dividing output_data by num of aggregated element.
  size_t num_elements_in_axis = 1;
  for (int idx = 0; idx < num_axes; ++idx) {
    size_t current = (size_t)(input_dims[axes[idx]]);
    num_elements_in_axis *= current;
  }

  for (size_t idx = 0; idx < num_outputs; ++idx) {
    output_data[idx] = output_data[idx] / (float)(num_elements_in_axis);
  }
}

float ReduceMeanAll(const float *src, int size) {
  float sum = 0;
  for (int i = 0; i < size; ++i) {
    sum += src[i];
  }
  return sum / size;
}

void ReduceSumByAxes(const float *input, const int *input_dims, float *output, const int *output_dims, int num_dims) {
  int num_outputs = 1;
  int same_shape = true;
  for (int idx = 0; idx < num_dims; ++idx) {
    num_outputs *= output_dims[idx];
    if (output_dims[idx] != input_dims[idx]) same_shape = false;
  }
  if (same_shape) {
    memcpy(output, input, num_outputs * sizeof(float));
    return;
  }

  for (int idx = 0; idx < num_outputs; ++idx) output[idx] = 0;  // zero output

  int input_iter[8] = {0};
  int axes[5] = {0};
  int num_axes = 0;
  for (int i = 0; i < num_dims; i++)
    if (output_dims[i] == 1) axes[num_axes++] = i;

  // Iterate through input_data.
  do {
    size_t input_offset = GetInputOffset(num_dims, input_dims, input_iter);
    size_t output_offset = GetOutputOffset(num_dims, input_dims, input_iter, num_axes, axes);
    output[output_offset] += input[input_offset];
  } while (NextIndex(num_dims, input_dims, input_iter));
}
