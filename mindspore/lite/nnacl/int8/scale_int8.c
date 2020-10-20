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

#include "nnacl/int8/scale_int8.h"
#include "nnacl/quantization/fixed_point.h"

void ScaleInnerInt8(const int8_t *in_data, int8_t *out_data, const int8_t *scale, int outer_start, int outer_end,
                    int axis_size, int inner_size, const ScaleParameter *scale_param, int max, int min) {
  for (int out = outer_start; out < outer_end; out++) {
    int out_offset = out * axis_size * inner_size;
    for (int i = 0; i < axis_size; i++) {
      int axis_offset = out_offset + i * inner_size;
      int in_index = 0;

      for (; in_index < inner_size; in_index++) {
        int in_offset = axis_offset + in_index;
        int tmp_input_scale = (in_data[in_offset] - scale_param->input_zp_) * (scale[i] - scale_param->scale_zp_);
        int input_mul_scale =
          RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(
                                tmp_input_scale * (1 << (unsigned int)scale_param->scale_mul_arg_.left_shift_),
                                scale_param->scale_mul_arg_.multiplier_),
                              scale_param->scale_mul_arg_.right_shift_);
        int tmp = input_mul_scale + scale_param->output_zp_;
        tmp = tmp > max ? max : tmp;
        tmp = tmp < min ? min : tmp;
        out_data[in_offset] = tmp;
      }
    }
  }
}

void ScaleInnerWithBiasInt8(const int8_t *in_data, int8_t *out_data, const int8_t *scale, const int8_t *offset,
                            int outer_start, int outer_end, int axis_size, int inner_size,
                            const ScaleParameter *scale_param, int max, int min) {
  for (int out = outer_start; out < outer_end; out++) {
    int out_offset = out * axis_size * inner_size;
    for (int i = 0; i < axis_size; i++) {
      int axis_offset = out_offset + i * inner_size;
      int in_index = 0;

      for (; in_index < inner_size; in_index++) {
        int in_offset = axis_offset + in_index;
        int tmp_input_scale = (in_data[in_offset] - scale_param->input_zp_) * (scale[i] - scale_param->scale_zp_);
        int input_mul_scale =
          RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(
                                tmp_input_scale * (1 << (unsigned int)scale_param->scale_mul_arg_.left_shift_),
                                scale_param->scale_mul_arg_.multiplier_),
                              scale_param->scale_mul_arg_.right_shift_);
        int tmp_bias = offset[i] - scale_param->offset_zp_;
        int bias = RoundingDivideByPOT(
          SaturatingRoundingDoublingHighMul(tmp_bias * (1 << (unsigned int)scale_param->offset_mul_arg_.left_shift_),
                                            scale_param->offset_mul_arg_.multiplier_),
          scale_param->offset_mul_arg_.right_shift_);
        int tmp = input_mul_scale + bias + scale_param->output_zp_;
        tmp = tmp > max ? max : tmp;
        tmp = tmp < min ? min : tmp;
        out_data[in_offset] = tmp;
      }
    }
  }
}

void DoScaleInt8(const int8_t *in_data, int8_t *out_data, const int8_t *scale, int task_id,
                 const ScaleParameter *scale_param, int max, int min) {
  int outer_step = UP_DIV(scale_param->outer_size_, scale_param->op_parameter_.thread_num_);
  int outer_start = task_id * outer_step;
  int outer_end = MSMIN(outer_start + outer_step, scale_param->outer_size_);

  ScaleInnerInt8(in_data, out_data, scale, outer_start, outer_end, scale_param->axis_size_, scale_param->inner_size_,
                 scale_param, max, min);
}

void DoScaleWithBiasInt8(const int8_t *in_data, int8_t *out_data, const int8_t *scale, const int8_t *offset,
                         int task_id, const ScaleParameter *scale_param, int max, int min) {
  int outer_step = UP_DIV(scale_param->outer_size_, scale_param->op_parameter_.thread_num_);
  int outer_start = task_id * outer_step;
  int outer_end = MSMIN(outer_start + outer_step, scale_param->outer_size_);

  ScaleInnerWithBiasInt8(in_data, out_data, scale, offset, outer_start, outer_end, scale_param->axis_size_,
                         scale_param->inner_size_, scale_param, max, min);
}
