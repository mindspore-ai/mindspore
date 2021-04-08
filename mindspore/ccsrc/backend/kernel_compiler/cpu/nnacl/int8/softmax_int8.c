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

int SoftmaxInt8(const int8_t *input_ptr, int8_t *output_ptr, int count, int *exp_data, int *sum_data,
                SoftmaxQuantArg quant_param, SoftmaxParameter *parameter) {
  int32_t axis = parameter->axis_;
  int n_dim = parameter->n_dim_;
  int *input_shape = parameter->input_shape_;
  int axis_shape_size = input_shape[axis];

  int inner_size = 1;
  for (int i = axis + 1; i < n_dim; i++) {
    inner_size *= input_shape[i];
  }

  for (int o = 0; o < count; o++) {
    int outter_offset = o * axis_shape_size * inner_size;

    for (int c = 0; c < inner_size; c++) {
      int8_t max_row = quant_param.output_activation_min_;
      for (int i = 0; i < axis_shape_size; ++i) {
        int axis_offset = outter_offset + c + i * inner_size;
        max_row = MSMAX(max_row, input_ptr[axis_offset]);
      }

      int32_t exp_sum = 0;
      for (int i = 0; i < axis_shape_size; ++i) {
        int axis_offset = outter_offset + c + i * inner_size;
        const int32_t input_val = input_ptr[axis_offset] - max_row;
        const int32_t input_scaled = SaturatingRoundingDoublingHighMul(
          input_val * (1 << (unsigned int)quant_param.shift_left_), quant_param.output_multiplier_);
        int exp_val = exp_on_negative_values(input_scaled, 5);
        exp_data[axis_offset] = exp_val;
        exp_sum = exp_sum + Rescale(exp_val, 0, 12);
      }
      sum_data[c] = exp_sum;
    }
    for (int i = 0; i < axis_shape_size; ++i) {
      int axis_offset = outter_offset + i * inner_size;
      for (int c = 0; c < inner_size; ++c) {
        int num_bits_over_unit;
        int shifted_scale = ComputerReciprocal(sum_data[c], 12, &num_bits_over_unit);
        int unsat_output = RoundingDivideByPOT(
          SaturatingRoundingDoublingHighMul(shifted_scale, exp_data[axis_offset + c]), num_bits_over_unit + 31 - 8);

        int raw_output = unsat_output + quant_param.output_activation_min_;
        output_ptr[axis_offset + c] =
          (int8_t)MSMAX(quant_param.output_activation_min_, MSMIN(raw_output, quant_param.output_activation_max_));
      }
    }
  }
  return 0;
}
