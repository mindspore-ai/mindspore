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

#ifdef ENABLE_NEON
int16x4_t ClacSumHalfWordMul(int16x4_t scaled_input0, int16x4_t scaled_input1, int32x4_t left_shift_out_vec,
                             int32x4_t right_shift_out_vec, int32x4_t output_multiplier_vec) {
  int32x4_t input_scale = vmull_s16(scaled_input0, scaled_input1);
  int32x4_t raw_sum = vqrdmulhq_s32(vmulq_s32(input_scale, left_shift_out_vec), output_multiplier_vec);
  const int32x4_t fixup = vshrq_n_s32(vandq_s32(raw_sum, right_shift_out_vec), 31);
  const int32x4_t fixed_up_x = vqaddq_s32(raw_sum, fixup);
  raw_sum = vrshlq_s32(fixed_up_x, right_shift_out_vec);
  return vqmovn_s32(raw_sum);
}

void MulInt8NEON(int8_t *input0_data, int8_t *input1_data, int8_t *output_data, int64_t real_dst_count,
                 MulQuantArg para, int *index) {
  int32x4_t output_multiplier_vec = vdupq_n_s32(para.output_multiplier_);
  int32x4_t left_shift_out_vec = vdupq_n_s32(1 << para.shift_left_);
  int32x4_t right_shift_out_vec = vdupq_n_s32(-para.shift_right_);
  int16x8_t out_zp_vec = vdupq_n_s16(para.out_quant_arg_.zp_);
  int8x16_t out_min_vec = vdupq_n_s8(para.output_activation_min_);
  int8x16_t out_max_vec = vdupq_n_s8(para.output_activation_max_);
  int8x8_t out_min_vec_s8 = vdup_n_s8(para.output_activation_min_);
  int8x8_t out_max_vec_s8 = vdup_n_s8(para.output_activation_max_);

  for (; (*index) <= real_dst_count - 16; (*index) += 16) {
    int16x8_t zp1_vec = vdupq_n_s16(para.in_quant_args_[0].zp_);
    int16x8_t zp2_vec = vdupq_n_s16(para.in_quant_args_[1].zp_);
    int8x16_t input0_vec = vld1q_s8(input0_data + *index);
    int8x16_t input1_vec = vld1q_s8(input1_data + *index);
    int16x8_t input0_low = vmovl_s8(vget_low_s8(input0_vec));
    int16x8_t input0_high = vmovl_s8(vget_high_s8(input0_vec));
    int16x8_t input1_low = vmovl_s8(vget_low_s8(input1_vec));
    int16x8_t input1_high = vmovl_s8(vget_high_s8(input1_vec));
    input0_low = vaddq_s16(input0_low, zp1_vec);
    input0_high = vaddq_s16(input0_high, zp1_vec);
    input1_low = vaddq_s16(input1_low, zp2_vec);
    input1_high = vaddq_s16(input1_high, zp2_vec);

    int16x4_t input0_low_low = vget_low_s16(input0_low);
    int16x4_t input0_low_high = vget_high_s16(input0_low);
    int16x4_t input0_high_low = vget_low_s16(input0_high);
    int16x4_t input0_high_high = vget_high_s16(input0_high);
    int16x4_t input1_low_low = vget_low_s16(input1_low);
    int16x4_t input1_low_high = vget_high_s16(input1_low);
    int16x4_t input1_high_low = vget_low_s16(input1_high);
    int16x4_t input1_high_high = vget_high_s16(input1_high);

    int16x4_t sum_low_low = ClacSumHalfWordMul(input0_low_low, input1_low_low, left_shift_out_vec, right_shift_out_vec,
                                               output_multiplier_vec);
    int16x4_t sum_low_high = ClacSumHalfWordMul(input0_low_high, input1_low_high, left_shift_out_vec,
                                                right_shift_out_vec, output_multiplier_vec);
    int16x4_t sum_high_low = ClacSumHalfWordMul(input0_high_low, input1_high_low, left_shift_out_vec,
                                                right_shift_out_vec, output_multiplier_vec);
    int16x4_t sum_high_high = ClacSumHalfWordMul(input0_high_high, input1_high_high, left_shift_out_vec,
                                                 right_shift_out_vec, output_multiplier_vec);

    int16x8_t res_s16 = vaddq_s16(vcombine_s16(sum_low_low, sum_low_high), out_zp_vec);
    int16x8_t res_s162 = vaddq_s16(vcombine_s16(sum_high_low, sum_high_high), out_zp_vec);
    int8x8_t res_u8_n0 = vqmovn_s16(res_s16);
    int8x8_t res_u8_n1 = vqmovn_s16(res_s162);
    int8x16_t res_s8 = vcombine_s8(res_u8_n0, res_u8_n1);
    res_s8 = vminq_s8(res_s8, out_max_vec);
    res_s8 = vmaxq_s8(res_s8, out_min_vec);
    vst1q_s8(output_data, res_s8);
    output_data += 16;
  }
  for (; (*index) <= real_dst_count - 8; (*index) += 8) {
    int16x8_t input0_val = LoadAndAddOffset(input0_data, *index, para.in_quant_args_[0].zp_);
    int16x8_t input1_val = LoadAndAddOffset(input1_data, *index, para.in_quant_args_[1].zp_);

    int16x4_t input0_low = vget_low_s16(input0_val);
    int16x4_t input0_high = vget_high_s16(input0_val);
    int16x4_t input1_low = vget_low_s16(input1_val);
    int16x4_t input1_high = vget_high_s16(input1_val);

    int16x4_t sum_low =
      ClacSumHalfWordMul(input0_low, input1_low, left_shift_out_vec, right_shift_out_vec, output_multiplier_vec);
    int16x4_t sum_high =
      ClacSumHalfWordMul(input0_high, input1_high, left_shift_out_vec, right_shift_out_vec, output_multiplier_vec);

    int16x8_t res_s16 = vaddq_s16(vcombine_s16(sum_low, sum_high), out_zp_vec);
    int8x8_t res_u8_n0 = vqmovn_s16(res_s16);
    res_u8_n0 = vmin_s8(res_u8_n0, out_max_vec_s8);
    res_u8_n0 = vmax_s8(res_u8_n0, out_min_vec_s8);
    vst1_s8(output_data, res_u8_n0);
    output_data += 8;
  }
}
#endif

void FastMul(int8_t *input0_data, int8_t *input1_data, int8_t *output_data, int depth, int64_t real_dst_count,
             bool input1_broad, MulQuantArg para) {
  // input0 need broadcast
  int32_t zp1 = para.in_quant_args_[0].zp_;
  int32_t zp2 = para.in_quant_args_[1].zp_;
  if (input1_broad) {
    zp1 = para.in_quant_args_[1].zp_;
    zp2 = para.in_quant_args_[0].zp_;
  }
#ifdef ENABLE_ARM
  int32x4_t output_multiplier_vec = vdupq_n_s32(para.output_multiplier_);
  int32x4_t left_shift_out_vec = vdupq_n_s32(1 << para.shift_left_);
  int32x4_t right_shift_out_vec = vdupq_n_s32(-para.shift_right_);
  int16x8_t out_zp_vec = vdupq_n_s16(para.out_quant_arg_.zp_);
  int8x16_t out_min_vec = vdupq_n_s8(para.output_activation_min_);
  int8x16_t out_max_vec = vdupq_n_s8(para.output_activation_max_);
  int8x8_t out_min_vec_s8 = vdup_n_s8(para.output_activation_min_);
  int8x8_t out_max_vec_s8 = vdup_n_s8(para.output_activation_max_);
  int16x8_t zp1_vec = vdupq_n_s16(zp1);
  int16x8_t zp2_vec = vdupq_n_s16(zp2);
#endif
  for (int index = 0; index < real_dst_count; ++index) {
    int j = 0;
#ifdef ENABLE_ARM
    for (; j <= depth - 16; j += 16) {
      int8x16_t input0_vec = vld1q_s8(input0_data + j);
      int8x16_t input1_vec = vld1q_s8(input1_data);
      int16x8_t input0_low = vmovl_s8(vget_low_s8(input0_vec));
      int16x8_t input0_high = vmovl_s8(vget_high_s8(input0_vec));
      int16x8_t input1_low = vmovl_s8(vget_low_s8(input1_vec));
      int16x8_t input1_high = vmovl_s8(vget_high_s8(input1_vec));
      input0_low = vaddq_s16(input0_low, zp1_vec);
      input0_high = vaddq_s16(input0_high, zp1_vec);
      input1_low = vaddq_s16(input1_low, zp2_vec);
      input1_high = vaddq_s16(input1_high, zp2_vec);

      int16x4_t input0_low_low = vget_low_s16(input0_low);
      int16x4_t input0_low_high = vget_high_s16(input0_low);
      int16x4_t input0_high_low = vget_low_s16(input0_high);
      int16x4_t input0_high_high = vget_high_s16(input0_high);
      int16x4_t input1_low_low = vget_low_s16(input1_low);
      int16x4_t input1_low_high = vget_high_s16(input1_low);
      int16x4_t input1_high_low = vget_low_s16(input1_high);
      int16x4_t input1_high_high = vget_high_s16(input1_high);

      int16x4_t sum_low_low = ClacSumHalfWordMul(input0_low_low, input1_low_low, left_shift_out_vec,
                                                 right_shift_out_vec, output_multiplier_vec);
      int16x4_t sum_low_high = ClacSumHalfWordMul(input0_low_high, input1_low_high, left_shift_out_vec,
                                                  right_shift_out_vec, output_multiplier_vec);
      int16x4_t sum_high_low = ClacSumHalfWordMul(input0_high_low, input1_high_low, left_shift_out_vec,
                                                  right_shift_out_vec, output_multiplier_vec);
      int16x4_t sum_high_high = ClacSumHalfWordMul(input0_high_high, input1_high_high, left_shift_out_vec,
                                                   right_shift_out_vec, output_multiplier_vec);

      int16x8_t res_s16 = vaddq_s16(vcombine_s16(sum_low_low, sum_low_high), out_zp_vec);
      int16x8_t res_s162 = vaddq_s16(vcombine_s16(sum_high_low, sum_high_high), out_zp_vec);
      int8x8_t res_u8_n0 = vqmovn_s16(res_s16);
      int8x8_t res_u8_n1 = vqmovn_s16(res_s162);
      int8x16_t res_s8 = vcombine_s8(res_u8_n0, res_u8_n1);
      res_s8 = vminq_s8(res_s8, out_max_vec);
      res_s8 = vmaxq_s8(res_s8, out_min_vec);
      vst1q_s8(output_data, res_s8);
      input1_data += 16;
      output_data += 16;
    }
    for (; j <= depth - 8; j += 8) {
      int8x8_t input0_vec = vld1_s8(input0_data + j);
      int8x8_t input1_vec = vld1_s8(input1_data);
      int16x8_t input0_val = vmovl_s8(input0_vec);
      int16x8_t input1_val = vmovl_s8(input1_vec);
      input0_val = vaddq_s16(input0_val, zp1_vec);
      input1_val = vaddq_s16(input1_val, zp2_vec);

      int16x4_t input0_low = vget_low_s16(input0_val);
      int16x4_t input0_high = vget_high_s16(input0_val);
      int16x4_t input1_low = vget_low_s16(input1_val);
      int16x4_t input1_high = vget_high_s16(input1_val);

      int16x4_t sum_low =
        ClacSumHalfWordMul(input0_low, input1_low, left_shift_out_vec, right_shift_out_vec, output_multiplier_vec);
      int16x4_t sum_high =
        ClacSumHalfWordMul(input0_high, input1_high, left_shift_out_vec, right_shift_out_vec, output_multiplier_vec);

      int16x8_t res_s16 = vaddq_s16(vcombine_s16(sum_low, sum_high), out_zp_vec);
      int8x8_t res_u8_n0 = vqmovn_s16(res_s16);
      res_u8_n0 = vmin_s8(res_u8_n0, out_max_vec_s8);
      res_u8_n0 = vmax_s8(res_u8_n0, out_min_vec_s8);
      vst1_s8(output_data, res_u8_n0);
      input1_data += 8;
      output_data += 8;
    }
#endif
    for (; j < depth; ++j) {
      const int32_t input0_val = zp1 + input0_data[j];
      const int32_t input1_val = zp2 + input1_data[0];
      int32_t mul_result = RoundingDivideByPOT(
        SaturatingRoundingDoublingHighMul(input0_val * input1_val * (1 << para.shift_left_), para.output_multiplier_),
        para.shift_right_);

      mul_result += para.out_quant_arg_.zp_;
      mul_result = mul_result < para.output_activation_max_ ? mul_result : para.output_activation_max_;
      mul_result = mul_result > para.output_activation_min_ ? mul_result : para.output_activation_min_;
      output_data[0] = (int8_t)mul_result;
      input1_data++;
      output_data++;
    }
  }
  return;
}

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
    mul_result = mul_result < para.output_activation_max_ ? mul_result : para.output_activation_max_;
    mul_result = mul_result > para.output_activation_min_ ? mul_result : para.output_activation_min_;
    output_data[index] = (int8_t)mul_result;
  }
  return;
}
