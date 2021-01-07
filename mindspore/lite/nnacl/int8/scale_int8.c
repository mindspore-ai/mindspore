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
#include "nnacl/int8/fixed_point.h"

#ifdef ENABLE_NEON
int16x4_t ClacSumHalfWordMul2(int32x4_t scaled_input0, int32x4_t scaled_input1, int32x4_t left_shift_out_vec,
                              int32x4_t output_multiplier_vec, const ScaleParameter *scale_param) {
  int32x4_t input_scale = vmulq_s32(scaled_input0, scaled_input1);
  int32x4_t raw_sum = RoundingDivideByPOTInt32x4(
    SaturatingRoundingDoublingHighMulInt32x4(vmulq_s32(input_scale, left_shift_out_vec), output_multiplier_vec),
    scale_param->scale_mul_arg_.right_shift_);
  raw_sum = vaddq_s32(raw_sum, vdupq_n_s32(scale_param->output_zp_));
  raw_sum = vmaxq_s32(raw_sum, vdupq_n_s32(scale_param->output_activation_min_));
  raw_sum = vminq_s32(raw_sum, vdupq_n_s32(scale_param->output_activation_max_));
  return vqmovn_s32(raw_sum);
}

int16x4_t ClacSumHalfWordMul3(int32x4_t scaled_input0, int32x4_t scaled_input1, int32x4_t scaled_input2,
                              const ScaleParameter *scale_param) {
  int32x4_t output_multiplier_vec = vdupq_n_s32(scale_param->scale_mul_arg_.multiplier_);
  int32x4_t output_multiplier_vec2 = vdupq_n_s32(scale_param->offset_mul_arg_.multiplier_);
  int32x4_t left_shift_out_vec = vdupq_n_s32(1 << scale_param->scale_mul_arg_.left_shift_);
  int32x4_t left_shift_out_vec2 = vdupq_n_s32(1 << scale_param->offset_mul_arg_.left_shift_);
  int32x4_t input_scale = vmulq_s32(scaled_input0, scaled_input1);
  int32x4_t raw_sum = RoundingDivideByPOTInt32x4(
    SaturatingRoundingDoublingHighMulInt32x4(vmulq_s32(input_scale, left_shift_out_vec), output_multiplier_vec),
    scale_param->scale_mul_arg_.right_shift_);
  int32x4_t raw_sum2 = RoundingDivideByPOTInt32x4(
    SaturatingRoundingDoublingHighMulInt32x4(vmulq_s32(scaled_input2, left_shift_out_vec2), output_multiplier_vec2),
    scale_param->offset_mul_arg_.right_shift_);
  raw_sum = vaddq_s32(raw_sum, vdupq_n_s32(scale_param->output_zp_));
  raw_sum = vaddq_s32(raw_sum, raw_sum2);
  raw_sum = vmaxq_s32(raw_sum, vdupq_n_s32(scale_param->output_activation_min_));
  raw_sum = vminq_s32(raw_sum, vdupq_n_s32(scale_param->output_activation_max_));
  return vqmovn_s32(raw_sum);
}
#endif

void DoScaleInt8(const int8_t *in_data, int8_t *out_data, const int8_t *scale, const ScaleParameter *scale_param,
                 int real_dst_count) {
  int index = 0;
#ifdef ENABLE_NEON
  int32x4_t output_multiplier_vec = vdupq_n_s32(scale_param->scale_mul_arg_.multiplier_);
  int32x4_t left_shift_out_vec = vdupq_n_s32(1 << scale_param->scale_mul_arg_.left_shift_);

  for (; index <= real_dst_count - 8; index += 8) {
    int8x8_t input_s8 = vld1_s8(in_data + index);
    int16x8_t input_s16 = vmovl_s8(input_s8);
    int16x8_t input0_val = vaddq_s16(input_s16, vdupq_n_s16(scale_param->input_zp_));

    int8x8_t input1_s8 = vld1_s8(scale + index);
    int16x8_t input1_s16 = vmovl_s8(input1_s8);
    int16x8_t input1_val = vaddq_s16(input1_s16, vdupq_n_s16(scale_param->scale_zp_));

    int32x4_t input0_low = vmovl_s16(vget_low_s16(input0_val));
    int32x4_t input0_high = vmovl_s16(vget_high_s16(input0_val));
    int32x4_t input1_low = vmovl_s16(vget_low_s16(input1_val));
    int32x4_t input1_high = vmovl_s16(vget_high_s16(input1_val));

    int16x4_t sum_low =
      ClacSumHalfWordMul2(input0_low, input1_low, left_shift_out_vec, output_multiplier_vec, scale_param);
    int16x4_t sum_high =
      ClacSumHalfWordMul2(input0_high, input1_high, left_shift_out_vec, output_multiplier_vec, scale_param);

    int16x8_t res_s16 = vcombine_s16(sum_low, sum_high);
    int8x8_t res_u8_n0 = vqmovn_s16(res_s16);
    vst1_s8(out_data, res_u8_n0);
    out_data += 8;
  }
#endif
  for (; index < real_dst_count; ++index) {
    const int32_t input0_val = scale_param->input_zp_ + in_data[index];
    const int32_t input1_val = scale_param->scale_zp_ + scale[index];
    int32_t mul_result = RoundingDivideByPOT(
      SaturatingRoundingDoublingHighMul(input0_val * input1_val * (1 << scale_param->scale_mul_arg_.left_shift_),
                                        scale_param->scale_mul_arg_.multiplier_),
      scale_param->scale_mul_arg_.right_shift_);

    mul_result += scale_param->output_zp_;

    if (mul_result > scale_param->output_activation_max_) {
      out_data[index] = scale_param->output_activation_max_;
    } else if (mul_result < scale_param->output_activation_min_) {
      out_data[index] = scale_param->output_activation_min_;
    } else {
      out_data[index] = (int8_t)mul_result;
    }
  }
  return;
}

void DoScaleWithBiasInt8(const int8_t *in_data, int8_t *out_data, const int8_t *scale, const int8_t *offset,
                         const ScaleParameter *scale_param, int real_dst_count) {
  int index = 0;
#ifdef ENABLE_NEON
  for (; index <= real_dst_count - 8; index += 8) {
    int8x8_t input_s8 = vld1_s8(in_data + index);
    int16x8_t input_s16 = vmovl_s8(input_s8);
    int16x8_t input0_val = vaddq_s16(input_s16, vdupq_n_s16(scale_param->input_zp_));

    int8x8_t input1_s8 = vld1_s8(scale + index);
    int16x8_t input1_s16 = vmovl_s8(input1_s8);
    int16x8_t input1_val = vaddq_s16(input1_s16, vdupq_n_s16(scale_param->scale_zp_));

    int8x8_t input2_s8 = vld1_s8(offset + index);
    int16x8_t input2_s16 = vmovl_s8(input2_s8);
    int16x8_t input2_val = vaddq_s16(input2_s16, vdupq_n_s16(scale_param->offset_zp_));

    int32x4_t input0_low = vmovl_s16(vget_low_s16(input0_val));
    int32x4_t input0_high = vmovl_s16(vget_high_s16(input0_val));
    int32x4_t input1_low = vmovl_s16(vget_low_s16(input1_val));
    int32x4_t input1_high = vmovl_s16(vget_high_s16(input1_val));
    int32x4_t input2_low = vmovl_s16(vget_low_s16(input2_val));
    int32x4_t input2_high = vmovl_s16(vget_high_s16(input2_val));

    int16x4_t sum_low = ClacSumHalfWordMul3(input0_low, input1_low, input2_low, scale_param);
    int16x4_t sum_high = ClacSumHalfWordMul3(input0_high, input1_high, input2_high, scale_param);

    int16x8_t res_s16 = vcombine_s16(sum_low, sum_high);
    int8x8_t res_u8_n0 = vqmovn_s16(res_s16);
    vst1_s8(out_data, res_u8_n0);
    out_data += 8;
  }
#endif
  for (; index < real_dst_count; ++index) {
    const int32_t input0_val = in_data[index] - scale_param->input_zp_;
    const int32_t input1_val = scale[index] - scale_param->scale_zp_;
    int32_t mul_result = RoundingDivideByPOT(
      SaturatingRoundingDoublingHighMul(input0_val * input1_val * (1 << scale_param->scale_mul_arg_.left_shift_),
                                        scale_param->scale_mul_arg_.multiplier_),
      scale_param->scale_mul_arg_.right_shift_);
    int tmp_bias = offset[index] - scale_param->offset_zp_;
    int bias = RoundingDivideByPOT(
      SaturatingRoundingDoublingHighMul(tmp_bias * (1 << (unsigned int)scale_param->offset_mul_arg_.left_shift_),
                                        scale_param->offset_mul_arg_.multiplier_),
      scale_param->offset_mul_arg_.right_shift_);

    mul_result += bias + scale_param->output_zp_;

    if (mul_result > scale_param->output_activation_max_) {
      out_data[index] = scale_param->output_activation_max_;
    } else if (mul_result < scale_param->output_activation_min_) {
      out_data[index] = scale_param->output_activation_min_;
    } else {
      out_data[index] = (int8_t)mul_result;
    }
  }
  return;
}
