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

#include "nnacl/int8/softmax_int8.h"
#include <math.h>

int Int8Softmax(const int8_t *input_ptr, int8_t *output_ptr, int count, float *exp_data, float *sum_data,
                SoftmaxQuantArg quant_param, SoftmaxParameter *parameter) {
  int32_t axis = parameter->axis_;
  int n_dim = parameter->n_dim_;
  int *input_shape = parameter->input_shape_;
  int axis_shape_size = input_shape[axis];

  double output_scale = quant_param.out_quant_arg_.scale_;
  int32_t output_zp = quant_param.out_quant_arg_.zp_;

  int inner_size = 1;
  for (int i = axis + 1; i < n_dim; i++) {
    inner_size *= input_shape[i];
  }

  for (int o = 0; o < count; o++) {
    int outter_offset = o * axis_shape_size * inner_size;
    for (int i = 0; i < inner_size; i++) {
      float sum = 0;
      for (int j = 0; j < axis_shape_size; j++) {
        int axis_offset = outter_offset + i + j * inner_size;
        sum += exp_data[axis_offset];
      }
      sum_data[i] = sum;
    }
    for (int j = 0; j < axis_shape_size; j++) {
      int axis_offset = outter_offset + j * inner_size;
      for (int i = 0; i < inner_size; i++) {
        int inner_offset = axis_offset + i;
        float real_output = exp_data[inner_offset] / sum_data[i];
        int32_t output_scaled = round(real_output / output_scale) + output_zp;
        output_ptr[inner_offset] = MSMAX(CHAR_MIN, MSMIN(CHAR_MAX, output_scaled));
      }
    }
  }
  return 0;
}
