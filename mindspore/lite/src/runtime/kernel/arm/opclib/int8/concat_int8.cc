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

#include "src/runtime/kernel/arm/opclib/int8/concat_int8.h"
#include <string.h>

void Concat(int8_t **inputs, int8_t *output_ptr, ConcatQuantArg *quant_concat_parm, int axis) {
  float output_scale = quant_concat_parm->out_quant_args_.scale_;
  float output_inverse_scale = 1.f / output_scale;
  int input_num = quant_concat_parm->input_num_;
  int *output_shape = quant_concat_parm->output_shape_;
  int output_dim = quant_concat_parm->output_dim_;
  QuantArg *input_quant = quant_concat_parm->in_quant_args_;
  int output_zp = quant_concat_parm->out_quant_args_.zp_;

  int before_axis_size = 1;
  for (int i = 0; i < axis; i++) {
    before_axis_size *= output_shape[i];
  }

  int after_axis_size = 1;
  for (size_t i = axis + 1; i < output_dim; i++) {
    after_axis_size *= output_shape[i];
  }

  for (int k = 0; k < before_axis_size; k++) {
    for (int i = 0; i < input_num; i++) {
      int *input_shape = quant_concat_parm->input_shapes_[i];
      int copy_size = input_shape[axis] * after_axis_size;
      int8_t *input_ptr = inputs[i] + k * copy_size;
      if (input_quant[i].scale_ == output_scale && input_quant[i].zp_ == output_zp) {
        memcpy(output_ptr, input_ptr, copy_size);
      } else {
        float scale = input_quant[i].scale_ * output_inverse_scale;
        float bias = -input_quant[i].zp_ * scale;
        for (int j = 0; j < copy_size; j++) {
          int32_t output_tmp = round(input_ptr[j] * scale + bias) + output_zp;
          if (output_tmp > 127) {
            output_ptr[j] = 127;
          } else if (output_tmp < -128) {
            output_ptr[j] = -128;
          } else {
            output_ptr[j] = (int8_t)output_tmp;
          }
        }
      }
      output_ptr += copy_size;
    }
  }
}

