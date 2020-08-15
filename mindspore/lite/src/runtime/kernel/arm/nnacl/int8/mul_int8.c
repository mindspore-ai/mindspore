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

#include "nnacl/int8/mul_int8.h"
#include "nnacl/mul_parameter.h"
#ifdef ENABLE_NEON
#include <arm_neon.h>
#include "nnacl/int8/add_int8.h"
#endif
#include "nnacl/quantization/fixed_point.h"

#ifdef ENABLE_NEON

int16x4_t ClacSumHalfWordMul(int32x4_t scaled_input0, int32x4_t scaled_input1, int32x4_t left_shift_out_vec,
                          int32x4_t output_multiplier_vec, MulQuantArg para) {
  int32x4_t input_scale = vmulq_s32(scaled_input0, scaled_input1);
  int32x4_t raw_sum = RoundingDivideByPOTInt32x4(
    SaturatingRoundingDoublingHighMulInt32x4(vmulq_s32(input_scale, left_shift_out_vec), output_multiplier_vec),
    para.shift_right_);
  raw_sum = vaddq_s32(raw_sum, vdupq_n_s32(para.out_quant_arg_.zp_));
  raw_sum = vmaxq_s32(raw_sum, vdupq_n_s32(para.output_activation_min_));
  raw_sum = vminq_s32(raw_sum, vdupq_n_s32(para.output_activation_max_));
  return vqmovn_s32(raw_sum);
}

void MulInt8NEON(int8_t *input0_data, int8_t *input1_data, int8_t *output_data, int64_t real_dst_count,
                 MulQuantArg para, int *index) {
  int32x4_t output_multiplier_vec = vdupq_n_s32(para.output_multiplier_);
  int32x4_t left_shift_out_vec = vdupq_n_s32(1 << para.shift_left_);

  for (; (*index) <= real_dst_count - 8; (*index) += 8) {
    int16x8_t input0_val = LoadAndAddOffset(input0_data, *index, para.in_quant_args_[0].zp_);
    int16x8_t input1_val = LoadAndAddOffset(input1_data, *index, para.in_quant_args_[1].zp_);

    int32x4_t input0_low = vmovl_s16(vget_low_s16(input0_val));
    int32x4_t input0_high = vmovl_s16(vget_high_s16(input0_val));
    int32x4_t input1_low = vmovl_s16(vget_low_s16(input1_val));
    int32x4_t input1_high = vmovl_s16(vget_high_s16(input1_val));

    int16x4_t sum_low = ClacSumHalfWordMul(input0_low, input1_low, left_shift_out_vec, output_multiplier_vec, para);
    int16x4_t sum_high = ClacSumHalfWordMul(input0_high, input1_high, left_shift_out_vec, output_multiplier_vec, para);

    int16x8_t res_s16 = vcombine_s16(sum_low, sum_high);
    int8x8_t res_u8_n0 = vqmovn_s16(res_s16);
    vst1_s8(output_data, res_u8_n0);
    output_data += 8;
  }
}
#endif

void Mul(int8_t *input0_data, int8_t *input1_data, int8_t *output_data, int64_t real_dst_count, MulQuantArg para) {
  int index = 0;
#ifdef ENABLE_NEON
  MulInt8NEON(input0_data, input1_data, output_data, real_dst_count, para, &index);
#endif
  for (; index < real_dst_count; ++index) {
    const int32_t input0_val = para.in_quant_args_[0].zp_ + input0_data[index];
    const int32_t input1_val = para.in_quant_args_[1].zp_ + input1_data[index];
    int32_t mul_result = RoundingDivideByPOT(
      SaturatingRoundingDoublingHighMul(input0_val * input1_val * (1 << para.shift_left_), para.output_multiplier_),
      para.shift_right_);

    mul_result += para.out_quant_arg_.zp_;

    if (mul_result > para.output_activation_max_) {
      output_data[index] = para.output_activation_max_;
    } else if (mul_result < para.output_activation_min_) {
      output_data[index] = para.output_activation_min_;
    } else {
      output_data[index] = (mul_result);
    }
  }
  return;
}
