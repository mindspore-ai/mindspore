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

#include "nnacl/fp32/softmax.h"
#include <math.h>
#include <float.h>

// output = exp(input) / reduce_sum(exp(input), axis)
void Softmax(const float *input_ptr, float *output_ptr, float *sum_data, SoftmaxParameter *parameter) {
  int32_t axis = parameter->axis_;
  int n_dim = parameter->n_dim_;
  int ele_size = parameter->element_size_;
  int *input_shape = parameter->input_shape_;

  float max_data = -FLT_MAX;
  for (int i = 0; i < ele_size; i++) {
    max_data = max_data > input_ptr[i] ? max_data : input_ptr[i];
  }

  for (int i = 0; i < ele_size; i++) {
    output_ptr[i] = exp(input_ptr[i] - max_data);
  }
  int inner_size = 1, outter_size = 1;
  for (int i = 0; i < axis; i++) {
    outter_size *= input_shape[i];
  }
  for (int i = axis + 1; i < n_dim; i++) {
    inner_size *= input_shape[i];
  }

  for (int i = 0; i < outter_size; i++) {
    int outter_offset = i * input_shape[axis] * inner_size;
    int sum_outter_offset = i * inner_size;
    for (int k = 0; k < inner_size; k++) {
      int inner_offset = outter_offset + k;
      for (int j = 0; j < input_shape[axis]; j++) {
        int axis_offset = inner_offset + j * inner_size;
        sum_data[k + sum_outter_offset] += output_ptr[axis_offset];
      }
    }
  }

  for (int i = 0; i < outter_size; i++) {
    int outter_offset = i * input_shape[axis] * inner_size;
    int sum_outter_offset = i * inner_size;
    for (int j = 0; j < input_shape[axis]; j++) {
      int axis_offset = outter_offset + j * inner_size;
      for (int k = 0; k < inner_size; k++) {
        int inner_offset = axis_offset + k;
        output_ptr[inner_offset] = output_ptr[inner_offset] / sum_data[k + sum_outter_offset];
      }
    }
  }
}
