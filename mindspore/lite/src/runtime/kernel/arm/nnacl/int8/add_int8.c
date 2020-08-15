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

#include "nnacl/int8/add_int8.h"
#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include "nnacl/quantization/fixed_point.h"

#ifdef ENABLE_NEON
int16x8_t LoadAndAddOffset(int8_t *data, int index, int offset) {
  int8x8_t input_s8 = vld1_s8(data + index);
  int16x8_t input_s16 = vmovl_s8(input_s8);
  return vaddq_s16(input_s16, vdupq_n_s16(offset));
}

int32x4_t ClacScaledInput(int32x4_t input, int32x4_t left_shift_result_vec, int32x4_t input_multiplier_vec,
                          int32x4_t right_shift_vec) {
  int32x4_t shifted_input = vmulq_s32(input, left_shift_result_vec);
  shifted_input = vqrdmulhq_s32(shifted_input, input_multiplier_vec);
  const int32x4_t fixup = vshrq_n_s32(vandq_s32(shifted_input, right_shift_vec), 31);
  return vrshlq_s32(vqaddq_s32(shifted_input, fixup), right_shift_vec);
}

int16x4_t AddClacSumHalfWord(int32x4_t scaled_input0, int32x4_t scaled_input1, int32x4_t left_shift_out_vec,
                          int32x4_t output_multiplier_vec, AddQuantParameter *para) {
  int32x4_t raw_sum = vaddq_s32(scaled_input0, scaled_input1);

  raw_sum = RoundingDivideByPOTInt32x4(vqrdmulhq_s32(vmulq_s32(raw_sum, left_shift_out_vec), output_multiplier_vec),
                                       para->right_shift_out_);
  raw_sum = vaddq_s32(raw_sum, vdupq_n_s32(para->output_offset_));
  raw_sum = vmaxq_s32(raw_sum, vdupq_n_s32(para->output_activation_min_));
  raw_sum = vminq_s32(raw_sum, vdupq_n_s32(para->output_activation_max_));
  return vqmovn_s32(raw_sum);
}

void AddInt8NEON(int8_t *input0_data, int8_t *input1_data, int8_t *output_data, int64_t real_dst_count,
                 AddQuantParameter *para, int *index) {
  int32x4_t left_shift_result0_vec = vdupq_n_s32(para->left_shift_result0_);
  int32x4_t left_shift_result1_vec = vdupq_n_s32(para->left_shift_result1_);
  int32x4_t input0_multiplier_vec = vdupq_n_s32(para->input0_multiplier_);
  int32x4_t input1_multiplier_vec = vdupq_n_s32(para->input1_multiplier_);
  int32x4_t output_multiplier_vec = vdupq_n_s32(para->output_multiplier_);
  int32x4_t left_shift_out_vec = vdupq_n_s32((1 << para->left_shift_out_));
  int32x4_t right_shift0_vec = vdupq_n_s32(-para->right_shift0_);
  int32x4_t right_shift1_vec = vdupq_n_s32(-para->right_shift1_);

  for (; (*index) <= real_dst_count - 8; (*index) += 8) {
    int16x8_t input0_val = LoadAndAddOffset(input0_data, *index, para->input0_offset_);
    int16x8_t input1_val = LoadAndAddOffset(input1_data, *index, para->input1_offset_);

    int32x4_t input0_low = vmovl_s16(vget_low_s16(input0_val));
    int32x4_t input0_high = vmovl_s16(vget_high_s16(input0_val));
    int32x4_t input1_low = vmovl_s16(vget_low_s16(input1_val));
    int32x4_t input1_high = vmovl_s16(vget_high_s16(input1_val));

    int32x4_t scaled_input0_low =
      ClacScaledInput(input0_low, left_shift_result0_vec, input0_multiplier_vec, right_shift0_vec);
    int32x4_t scaled_input0_high =
      ClacScaledInput(input0_high, left_shift_result0_vec, input0_multiplier_vec, right_shift0_vec);
    int32x4_t scaled_input1_low =
      ClacScaledInput(input1_low, left_shift_result1_vec, input1_multiplier_vec, right_shift1_vec);
    int32x4_t scaled_input1_high =
      ClacScaledInput(input1_high, left_shift_result1_vec, input1_multiplier_vec, right_shift1_vec);

    int16x4_t sum_low =
      AddClacSumHalfWord(scaled_input0_low, scaled_input1_low, left_shift_out_vec, output_multiplier_vec, para);
    int16x4_t sum_high =
      AddClacSumHalfWord(scaled_input0_high, scaled_input1_high, left_shift_out_vec, output_multiplier_vec, para);

    int16x8_t res_s16 = vcombine_s16(sum_low, sum_high);
    int8x8_t res_u8_n0 = vqmovn_s16(res_s16);
    vst1_s8(output_data + *index, res_u8_n0);
  }
}
#endif

void AddInt8(int8_t *input0_data, int8_t *input1_data, int8_t *output_data, int64_t real_dst_count,
             AddQuantParameter *para) {
  int index = 0;
#ifdef ENABLE_NEON
  AddInt8NEON(input0_data, input1_data, output_data, real_dst_count, para, &index);
#endif
  for (; index < real_dst_count; ++index) {
    const int32_t input0_val = para->input0_offset_ + input0_data[index];
    const int32_t input1_val = para->input1_offset_ + input1_data[index];
    const int32_t shifted_input0_val = input0_val * para->left_shift_result0_;
    const int32_t shifted_input1_val = input1_val * para->left_shift_result1_;
    const int32_t scaled_input0_val = RoundingDivideByPOT(
      SaturatingRoundingDoublingHighMul(shifted_input0_val, para->input0_multiplier_), para->right_shift0_);
    const int32_t scaled_input1_val = RoundingDivideByPOT(
      SaturatingRoundingDoublingHighMul(shifted_input1_val, para->input1_multiplier_), para->right_shift1_);

    const int32_t raw_sum = scaled_input0_val + scaled_input1_val;
    const int32_t raw_output =
      RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(raw_sum * (1 << (unsigned int)para->left_shift_out_),
                                                            para->output_multiplier_),
                          para->right_shift_out_) +
      para->output_offset_;

    output_data[index] = (int8_t)MSMAX(para->output_activation_min_, MSMIN(raw_output, para->output_activation_max_));
  }
  return;
}
