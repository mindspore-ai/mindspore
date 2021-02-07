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
#ifdef ENABLE_AVX
#include <x86intrin.h>
#include "nnacl/intrinsics/avx/common_utils.h"
#endif
#include "nnacl/int8/fixed_point.h"

#ifdef ENABLE_ARM
void AddInt8InputRounding(int32x4_t *in1, int32x4_t *in2, int32x4_t *in3, int32x4_t *in4, const int32x4_t left_vec,
                          const int32x4_t right_vec, const int32_t multiplier) {
  // Apply left shift
  *in1 = vmulq_s32(*in1, left_vec);
  *in2 = vmulq_s32(*in2, left_vec);
  *in3 = vmulq_s32(*in3, left_vec);
  *in4 = vmulq_s32(*in4, left_vec);

  // Apply the fixed-point part of the multiplier.
  *in1 = vqrdmulhq_n_s32(*in1, multiplier);
  *in2 = vqrdmulhq_n_s32(*in2, multiplier);
  *in3 = vqrdmulhq_n_s32(*in3, multiplier);
  *in4 = vqrdmulhq_n_s32(*in4, multiplier);

  // Apply right shift
  *in1 = vqaddq_s32(*in1, vshrq_n_s32(vandq_s32(*in1, right_vec), 31));
  *in2 = vqaddq_s32(*in2, vshrq_n_s32(vandq_s32(*in2, right_vec), 31));
  *in3 = vqaddq_s32(*in3, vshrq_n_s32(vandq_s32(*in3, right_vec), 31));
  *in4 = vqaddq_s32(*in4, vshrq_n_s32(vandq_s32(*in4, right_vec), 31));

  *in1 = vrshlq_s32(*in1, right_vec);
  *in2 = vrshlq_s32(*in2, right_vec);
  *in3 = vrshlq_s32(*in3, right_vec);
  *in4 = vrshlq_s32(*in4, right_vec);
}

void AddInt8OutputRounding(int32x4_t *out1, int32x4_t *out2, int32x4_t *out3, int32x4_t *out4, const int32x4_t left_vec,
                           const int32x4_t right_vec, const int32_t multiplier) {
  // Apply left shift
  *out1 = vshlq_s32(*out1, left_vec);
  *out2 = vshlq_s32(*out2, left_vec);
  *out3 = vshlq_s32(*out3, left_vec);
  *out4 = vshlq_s32(*out4, left_vec);

  // Apply the fixed-point part of the multiplier.
  *out1 = vqrdmulhq_n_s32(*out1, multiplier);
  *out2 = vqrdmulhq_n_s32(*out2, multiplier);
  *out3 = vqrdmulhq_n_s32(*out3, multiplier);
  *out4 = vqrdmulhq_n_s32(*out4, multiplier);

  // Apply right shift
  *out1 = vqaddq_s32(*out1, vshrq_n_s32(vandq_s32(*out1, right_vec), 31));
  *out2 = vqaddq_s32(*out2, vshrq_n_s32(vandq_s32(*out2, right_vec), 31));
  *out3 = vqaddq_s32(*out3, vshrq_n_s32(vandq_s32(*out3, right_vec), 31));
  *out4 = vqaddq_s32(*out4, vshrq_n_s32(vandq_s32(*out4, right_vec), 31));

  *out1 = vrshlq_s32(*out1, right_vec);
  *out2 = vrshlq_s32(*out2, right_vec);
  *out3 = vrshlq_s32(*out3, right_vec);
  *out4 = vrshlq_s32(*out4, right_vec);
}
#endif

void AddInt8(const int8_t *input0, const int8_t *input1, int8_t *output, int size, AddQuantParameter *params) {
  int in0_left_shift = (1 << params->left_shift_) * (1 << params->in0_args_.left_shift_);
  int in1_left_shift = (1 << params->left_shift_) * (1 << params->in1_args_.left_shift_);
  int index = 0;
#ifdef ENABLE_ARM
  const int8x16_t min_vec = vdupq_n_s8(params->min_);
  const int8x16_t max_vec = vdupq_n_s8(params->max_);

  const int16x8_t in0_zp_vec = vdupq_n_s16(params->in0_args_.zp_);
  const int16x8_t in1_zp_vec = vdupq_n_s16(params->in1_args_.zp_);
  const int16x8_t out_zp_vec = vdupq_n_s16(params->out_zp_);

  const int32x4_t in0_left_vec = vdupq_n_s32(in0_left_shift);
  const int32x4_t in1_left_vec = vdupq_n_s32(in1_left_shift);

  const int32x4_t in0_right_vec = vdupq_n_s32(-params->in0_args_.right_shift_);
  const int32x4_t in1_right_vec = vdupq_n_s32(-params->in1_args_.right_shift_);

  const int32x4_t out_left_vec = vdupq_n_s32(params->out_left_shift_);
  const int32x4_t out_right_vec = vdupq_n_s32(-params->out_right_shift_);

  for (; index <= size - 16; index += 16) {
    const int8x16_t in0_src = vld1q_s8(input0 + index);
    const int8x16_t in1_src = vld1q_s8(input1 + index);

    const int16x8_t in0_s16_low = vmovl_s8(vget_low_s8(in0_src));
    const int16x8_t in0_s16_high = vmovl_s8(vget_high_s8(in0_src));
    const int16x8_t in1_s16_low = vmovl_s8(vget_low_s8(in1_src));
    const int16x8_t in1_s16_high = vmovl_s8(vget_high_s8(in1_src));

    const int16x8_t in0_zp_low = vaddq_s16(in0_s16_low, in0_zp_vec);
    const int16x8_t in0_zp_high = vaddq_s16(in0_s16_high, in0_zp_vec);
    const int16x8_t in1_zp_low = vaddq_s16(in1_s16_low, in1_zp_vec);
    const int16x8_t in1_zp_high = vaddq_s16(in1_s16_high, in1_zp_vec);

    int32x4_t in0_1 = vmovl_s16(vget_low_s16(in0_zp_low));
    int32x4_t in0_2 = vmovl_s16(vget_high_s16(in0_zp_low));
    int32x4_t in0_3 = vmovl_s16(vget_low_s16(in0_zp_high));
    int32x4_t in0_4 = vmovl_s16(vget_high_s16(in0_zp_high));
    int32x4_t in1_1 = vmovl_s16(vget_low_s16(in1_zp_low));
    int32x4_t in1_2 = vmovl_s16(vget_high_s16(in1_zp_low));
    int32x4_t in1_3 = vmovl_s16(vget_low_s16(in1_zp_high));
    int32x4_t in1_4 = vmovl_s16(vget_high_s16(in1_zp_high));

    AddInt8InputRounding(&in0_1, &in0_2, &in0_3, &in0_4, in0_left_vec, in0_right_vec, params->in0_args_.multiplier_);
    AddInt8InputRounding(&in1_1, &in1_2, &in1_3, &in1_4, in1_left_vec, in1_right_vec, params->in1_args_.multiplier_);

    /* calculate output */
    int32x4_t out1 = vaddq_s32(in0_1, in1_1);
    int32x4_t out2 = vaddq_s32(in0_2, in1_2);
    int32x4_t out3 = vaddq_s32(in0_3, in1_3);
    int32x4_t out4 = vaddq_s32(in0_4, in1_4);

    AddInt8OutputRounding(&out1, &out2, &out3, &out4, out_left_vec, out_right_vec, params->out_multiplier_);

    const int16x4_t out1_s16 = vmovn_s32(out1);
    const int16x4_t out2_s16 = vmovn_s32(out2);
    const int16x4_t out3_s16 = vmovn_s32(out3);
    const int16x4_t out4_s16 = vmovn_s32(out4);

    const int16x8_t out_s16_1 = vaddq_s16(vcombine_s16(out1_s16, out2_s16), out_zp_vec);
    const int16x8_t out_s16_2 = vaddq_s16(vcombine_s16(out3_s16, out4_s16), out_zp_vec);

    const int8x16_t out = vcombine_s8(vqmovn_s16(out_s16_1), vqmovn_s16(out_s16_2));
    const int8x16_t int8_out = vmaxq_s8(min_vec, vminq_s8(max_vec, out));

    vst1q_s8(output + index, int8_out);
  }
#endif
  for (; index < size; index++) {
    const int32_t in0_left = (input0[index] + params->in0_args_.zp_) * in0_left_shift;
    const int32_t in1_left = (input1[index] + params->in1_args_.zp_) * in1_left_shift;
    const int32_t in0 =
      MultiplyByMultiplierAndRightShift(in0_left, params->in0_args_.multiplier_, params->in0_args_.right_shift_);
    const int32_t in1 =
      MultiplyByMultiplierAndRightShift(in1_left, params->in1_args_.multiplier_, params->in1_args_.right_shift_);

    int32_t out = MultiplyByQuantizedMultiplier(in0 + in1, params->out_multiplier_, params->out_left_shift_,
                                                -params->out_right_shift_);
    out += params->out_zp_;
    output[index] = (int8_t)MSMAX(params->min_, MSMIN(out, params->max_));
  }
  return;
}

void AddOptInt8(const int8_t *ptr_in, const int8_t element_in, int8_t *output, int size, AddQuantParameter *params,
                AddQuantQrgs *ptr_args, AddQuantQrgs *ele_args) {
  int ptr_left_shift = (1 << params->left_shift_) * (1 << ptr_args->left_shift_);
  int ele_left_shift = (1 << params->left_shift_) * (1 << ele_args->left_shift_);
  int index = 0;

#ifdef ENABLE_ARM
  /* const value init */
  const int8x16_t min_vec = vdupq_n_s8(params->min_);
  const int8x16_t max_vec = vdupq_n_s8(params->max_);

  const int16x8_t ptr_zp_vec = vdupq_n_s16(ptr_args->zp_);
  const int16x8_t ele_zp_vec = vdupq_n_s16(ele_args->zp_);
  const int16x8_t out_zp_vec = vdupq_n_s16(params->out_zp_);

  const int32x4_t ptr_left_vec = vdupq_n_s32(ptr_left_shift);
  const int32x4_t ele_left_vec = vdupq_n_s32(ele_left_shift);

  const int32x4_t ptr_right_vec = vdupq_n_s32(-ptr_args->right_shift_);
  const int32x4_t ele_right_vec = vdupq_n_s32(-ele_args->right_shift_);

  const int32x4_t out_left_vec = vdupq_n_s32(params->out_left_shift_);
  const int32x4_t out_right_vec = vdupq_n_s32(-params->out_right_shift_);

  /* deal with const node */
  const int8x16_t ele_src = vdupq_n_s8(element_in);
  const int16x8_t ele_s16_low = vmovl_s8(vget_low_s8(ele_src));
  const int16x8_t ele_s16_high = vmovl_s8(vget_high_s8(ele_src));
  const int16x8_t ele_zp_low = vaddq_s16(ele_s16_low, ele_zp_vec);
  const int16x8_t ele_zp_high = vaddq_s16(ele_s16_high, ele_zp_vec);
  int32x4_t ele1 = vmovl_s16(vget_low_s16(ele_zp_low));
  int32x4_t ele2 = vmovl_s16(vget_high_s16(ele_zp_low));
  int32x4_t ele3 = vmovl_s16(vget_low_s16(ele_zp_high));
  int32x4_t ele4 = vmovl_s16(vget_high_s16(ele_zp_high));

  AddInt8InputRounding(&ele1, &ele2, &ele3, &ele4, ele_left_vec, ele_right_vec, ele_args->multiplier_);

  for (; index <= size - 16; index += 16) {
    const int8x16_t ptr_src = vld1q_s8(ptr_in + index);

    const int16x8_t ptr_s16_low = vmovl_s8(vget_low_s8(ptr_src));
    const int16x8_t ptr_s16_high = vmovl_s8(vget_high_s8(ptr_src));

    const int16x8_t ptr_zp_low = vaddq_s16(ptr_s16_low, ptr_zp_vec);
    const int16x8_t ptr_zp_high = vaddq_s16(ptr_s16_high, ptr_zp_vec);

    int32x4_t ptr1 = vmovl_s16(vget_low_s16(ptr_zp_low));
    int32x4_t ptr2 = vmovl_s16(vget_high_s16(ptr_zp_low));
    int32x4_t ptr3 = vmovl_s16(vget_low_s16(ptr_zp_high));
    int32x4_t ptr4 = vmovl_s16(vget_high_s16(ptr_zp_high));

    AddInt8InputRounding(&ptr1, &ptr2, &ptr3, &ptr4, ptr_left_vec, ptr_right_vec, ptr_args->multiplier_);

    /* calculate output */
    int32x4_t out1 = vaddq_s32(ptr1, ele1);
    int32x4_t out2 = vaddq_s32(ptr2, ele2);
    int32x4_t out3 = vaddq_s32(ptr3, ele3);
    int32x4_t out4 = vaddq_s32(ptr4, ele4);

    AddInt8OutputRounding(&out1, &out2, &out3, &out4, out_left_vec, out_right_vec, params->out_multiplier_);

    const int16x4_t out1_s16 = vmovn_s32(out1);
    const int16x4_t out2_s16 = vmovn_s32(out2);
    const int16x4_t out3_s16 = vmovn_s32(out3);
    const int16x4_t out4_s16 = vmovn_s32(out4);

    const int16x8_t out_s16_1 = vaddq_s16(vcombine_s16(out1_s16, out2_s16), out_zp_vec);
    const int16x8_t out_s16_2 = vaddq_s16(vcombine_s16(out3_s16, out4_s16), out_zp_vec);

    const int8x16_t out = vcombine_s8(vqmovn_s16(out_s16_1), vqmovn_s16(out_s16_2));
    const int8x16_t int8_out = vmaxq_s8(min_vec, vminq_s8(max_vec, out));

    vst1q_s8(output + index, int8_out);
  }
#endif
  for (; index < size; index++) {
    const int32_t ptr_left = (ptr_in[index] + ptr_args->zp_) * ptr_left_shift;
    const int32_t ele_left = (element_in + ele_args->zp_) * ele_left_shift;
    const int32_t ptr = MultiplyByMultiplierAndRightShift(ptr_left, ptr_args->multiplier_, ptr_args->right_shift_);
    const int32_t ele = MultiplyByMultiplierAndRightShift(ele_left, ele_args->multiplier_, ele_args->right_shift_);

    int32_t out = MultiplyByQuantizedMultiplier(ptr + ele, params->out_multiplier_, params->out_left_shift_,
                                                -params->out_right_shift_);
    out += params->out_zp_;
    output[index] = (int8_t)MSMAX(params->min_, MSMIN(out, params->max_));
  }
  return;
}

int ElementAddInt8(const int8_t *in0, const int8_t *in1, int8_t *out, int size) {
  for (int i = 0; i < size; i++) {
    out[i] = in0[i] + in1[i];
  }
  return NNACL_OK;
}

int BroadcastAddInt8(const int8_t *in0, const int8_t *in1, int8_t *tile_in0, int8_t *tile_in1, int8_t *out, int size,
                     ArithmeticParameter *param) {
  TileDimensionsInt8(in0, in1, tile_in0, tile_in1, param);
  return ElementAddInt8(tile_in0, tile_in1, out, size);
}

#ifdef ENABLE_AVX
void AddInt8Rounding(__m128i *in1, __m128i *in2, __m128i *in3, __m128i *in4, const __m128i left_vec,
                     const int32_t right_shift, const __m128i multiplier) {
  // Apply left shift
  *in1 = _mm_mullo_epi32(*in1, left_vec);
  *in2 = _mm_mullo_epi32(*in2, left_vec);
  *in3 = _mm_mullo_epi32(*in3, left_vec);
  *in4 = _mm_mullo_epi32(*in4, left_vec);

  // Apply the fixed-point part of the multiplier.
  *in1 = _mm_qrdmulh_epi32(*in1, multiplier);
  *in2 = _mm_qrdmulh_epi32(*in2, multiplier);
  *in3 = _mm_qrdmulh_epi32(*in3, multiplier);
  *in4 = _mm_qrdmulh_epi32(*in4, multiplier);

  // Apply right shift
  int32_t in1_remainder_mask = (1ll << (right_shift)) - 1;
  int32_t in1_remainder_threshold = in1_remainder_mask >> 1;
  const __m128i vin1_remainder_mask = _mm_set1_epi32(in1_remainder_mask);
  const __m128i vin1_remainder_threshold = _mm_set1_epi32(in1_remainder_threshold);

  const __m128i in1_remainder =
    _mm_add_epi32(_mm_and_si128(*in1, vin1_remainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), *in1));
  *in1 = _mm_sub_epi32(_mm_rshr_epi32(*in1, right_shift), _mm_cmpgt_epi32(in1_remainder, vin1_remainder_threshold));

  const __m128i in2_remainder =
    _mm_add_epi32(_mm_and_si128(*in2, vin1_remainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), *in2));
  *in2 = _mm_sub_epi32(_mm_rshr_epi32(*in2, right_shift), _mm_cmpgt_epi32(in2_remainder, vin1_remainder_threshold));

  const __m128i in3_remainder =
    _mm_add_epi32(_mm_and_si128(*in3, vin1_remainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), *in3));
  *in3 = _mm_sub_epi32(_mm_rshr_epi32(*in3, right_shift), _mm_cmpgt_epi32(in3_remainder, vin1_remainder_threshold));

  const __m128i in4_remainder =
    _mm_add_epi32(_mm_and_si128(*in4, vin1_remainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), *in4));
  *in4 = _mm_sub_epi32(_mm_rshr_epi32(*in4, right_shift), _mm_cmpgt_epi32(in4_remainder, vin1_remainder_threshold));
}

void AddInt8_AVX2(const int8_t *input0, const int8_t *input1, int8_t *output, int size, AddQuantParameter *params) {
  const int in0_left_shift = (1 << params->left_shift_) * (1 << params->in0_args_.left_shift_);
  const int in1_left_shift = (1 << params->left_shift_) * (1 << params->in1_args_.left_shift_);
  const __m128i min_vec = _mm_set1_epi8(params->min_);
  const __m128i max_vec = _mm_set1_epi8(params->max_);
  const __m128i in0_zp_vec = _mm_set1_epi16(params->in0_args_.zp_);
  const __m128i in1_zp_vec = _mm_set1_epi16(params->in1_args_.zp_);
  const __m128i out_zp_vec = _mm_set1_epi16(params->out_zp_);
  const __m128i in0_left_vec = _mm_set1_epi32(in0_left_shift);
  const __m128i in1_left_vec = _mm_set1_epi32(in1_left_shift);
  const __m128i in0_multiplier = _mm_set1_epi32(params->in0_args_.multiplier_);
  const __m128i in1_multiplier = _mm_set1_epi32(params->in1_args_.multiplier_);
  const __m128i out_multiplier = _mm_set1_epi32(params->out_multiplier_);
  int index = 0;
  for (; index <= size - 16; index += 16) {
    const __m128i in0_src = _mm_loadu_si128((__m128i_u *)(input0 + index));
    const __m128i in1_src = _mm_loadu_si128((__m128i_u *)(input1 + index));

    const __m256i in0_s16 = _mm256_cvtepi8_epi16(in0_src);
    const __m128i in0_s16_low = _mm256_extractf128_si256(in0_s16, 0);
    const __m128i in0_s16_high = _mm256_extractf128_si256(in0_s16, 1);
    const __m256i in1_s16 = _mm256_cvtepi8_epi16(in1_src);
    const __m128i in1_s16_low = _mm256_extractf128_si256(in1_s16, 0);
    const __m128i in1_s16_high = _mm256_extractf128_si256(in1_s16, 1);

    const __m128i in0_zp_low = _mm_add_epi16(in0_s16_low, in0_zp_vec);
    const __m128i in0_zp_high = _mm_add_epi16(in0_s16_high, in0_zp_vec);
    const __m128i in1_zp_low = _mm_add_epi16(in1_s16_low, in1_zp_vec);
    const __m128i in1_zp_high = _mm_add_epi16(in1_s16_high, in1_zp_vec);

    __m256i tmp_in0 = _mm256_cvtepi16_epi32(in0_zp_low);
    __m128i in0_1 = _mm256_extractf128_si256(tmp_in0, 0);
    __m128i in0_2 = _mm256_extractf128_si256(tmp_in0, 1);
    tmp_in0 = _mm256_cvtepi16_epi32(in0_zp_high);
    __m128i in0_3 = _mm256_extractf128_si256(tmp_in0, 0);
    __m128i in0_4 = _mm256_extractf128_si256(tmp_in0, 1);
    __m256i tmp_in1 = _mm256_cvtepi16_epi32(in1_zp_low);
    __m128i in1_1 = _mm256_extractf128_si256(tmp_in1, 0);
    __m128i in1_2 = _mm256_extractf128_si256(tmp_in1, 1);
    tmp_in1 = _mm256_cvtepi16_epi32(in1_zp_high);
    __m128i in1_3 = _mm256_extractf128_si256(tmp_in1, 0);
    __m128i in1_4 = _mm256_extractf128_si256(tmp_in1, 1);

    AddInt8Rounding(&in0_1, &in0_2, &in0_3, &in0_4, in0_left_vec, params->in0_args_.right_shift_, in0_multiplier);
    AddInt8Rounding(&in1_1, &in1_2, &in1_3, &in1_4, in1_left_vec, params->in1_args_.right_shift_, in1_multiplier);

    /* calculate output */
    __m128i out1 = _mm_add_epi32(in0_1, in1_1);
    __m128i out2 = _mm_add_epi32(in0_2, in1_2);
    __m128i out3 = _mm_add_epi32(in0_3, in1_3);
    __m128i out4 = _mm_add_epi32(in0_4, in1_4);

    // Apply left shift
    out1 = _mm_slli_epi32(out1, params->out_left_shift_);
    out2 = _mm_slli_epi32(out2, params->out_left_shift_);
    out3 = _mm_slli_epi32(out3, params->out_left_shift_);
    out4 = _mm_slli_epi32(out4, params->out_left_shift_);

    // Apply the fixed-point part of the multiplier.
    out1 = _mm_qrdmulh_epi32(out1, out_multiplier);
    out2 = _mm_qrdmulh_epi32(out2, out_multiplier);
    out3 = _mm_qrdmulh_epi32(out3, out_multiplier);
    out4 = _mm_qrdmulh_epi32(out4, out_multiplier);

    // Apply right shift
    int32_t out_remainder_mask = (1ll << (params->out_right_shift_)) - 1;
    int32_t out_remainder_threshold = out_remainder_mask >> 1;
    const __m128i vout_remainder_mask = _mm_set1_epi32(out_remainder_mask);
    const __m128i vout_remainder_threshold = _mm_set1_epi32(out_remainder_threshold);
    const __m128i out1_remainder =
      _mm_add_epi32(_mm_and_si128(out1, vout_remainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), out1));
    out1 = _mm_sub_epi32(_mm_rshr_epi32(out1, params->out_right_shift_),
                         _mm_cmpgt_epi32(out1_remainder, vout_remainder_threshold));
    const __m128i out2_remainder =
      _mm_add_epi32(_mm_and_si128(out2, vout_remainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), out2));
    out2 = _mm_sub_epi32(_mm_rshr_epi32(out2, params->out_right_shift_),
                         _mm_cmpgt_epi32(out2_remainder, vout_remainder_threshold));
    const __m128i out3_remainder =
      _mm_add_epi32(_mm_and_si128(out3, vout_remainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), out3));
    out3 = _mm_sub_epi32(_mm_rshr_epi32(out3, params->out_right_shift_),
                         _mm_cmpgt_epi32(out3_remainder, vout_remainder_threshold));
    const __m128i out4_remainder =
      _mm_add_epi32(_mm_and_si128(out4, vout_remainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), out4));
    out4 = _mm_sub_epi32(_mm_rshr_epi32(out4, params->out_right_shift_),
                         _mm_cmpgt_epi32(out4_remainder, vout_remainder_threshold));

    __m128i out1_s16 = _mm_packs_epi32(out1, out2);
    __m128i out2_s16 = _mm_packs_epi32(out3, out4);

    __m128i out_s16_1 = _mm_add_epi16(out1_s16, out_zp_vec);
    __m128i out_s16_2 = _mm_add_epi16(out2_s16, out_zp_vec);
    __m128i out = _mm_packs_epi16(out_s16_1, out_s16_2);
    __m128i int8_out = _mm_max_epi8(min_vec, _mm_min_epi8(max_vec, out));

    _mm_storeu_si128((__m128i_u *)(output + index), int8_out);
  }
  for (; index < size; index++) {
    const int32_t in0_left = (input0[index] + params->in0_args_.zp_) * in0_left_shift;
    const int32_t in1_left = (input1[index] + params->in1_args_.zp_) * in1_left_shift;
    const int32_t in0 =
      MultiplyByMultiplierAndRightShift(in0_left, params->in0_args_.multiplier_, params->in0_args_.right_shift_);
    const int32_t in1 =
      MultiplyByMultiplierAndRightShift(in1_left, params->in1_args_.multiplier_, params->in1_args_.right_shift_);

    int32_t out = MultiplyByQuantizedMultiplier(in0 + in1, params->out_multiplier_, params->out_left_shift_,
                                                -params->out_right_shift_);
    out += params->out_zp_;
    output[index] = (int8_t)MSMAX(params->min_, MSMIN(out, params->max_));
  }
  return;
}

void AddOptInt8_AVX2(const int8_t *ptr_in, const int8_t element_in, int8_t *output, int size, AddQuantParameter *params,
                     AddQuantQrgs *ptr_args, AddQuantQrgs *ele_args) {
  // input0: ptr_in
  // input1: element_in
  // load quant parameters of input0 and input1
  const int in0_left_shift = (1 << params->left_shift_) * (1 << ptr_args->left_shift_);
  const int in1_left_shift = (1 << params->left_shift_) * (1 << ele_args->left_shift_);
  const __m128i min_vec = _mm_set1_epi8(params->min_);
  const __m128i max_vec = _mm_set1_epi8(params->max_);
  const __m128i in0_zp_vec = _mm_set1_epi16(ptr_args->zp_);
  const __m128i in1_zp_vec = _mm_set1_epi16(ele_args->zp_);
  const __m128i out_zp_vec = _mm_set1_epi16(params->out_zp_);
  const __m128i in0_left_vec = _mm_set1_epi32(in0_left_shift);
  const __m128i in1_left_vec = _mm_set1_epi32(in1_left_shift);
  const __m128i in0_multiplier = _mm_set1_epi32(params->in0_args_.multiplier_);
  const __m128i in1_multiplier = _mm_set1_epi32(params->in1_args_.multiplier_);
  const __m128i out_multiplier = _mm_set1_epi32(params->out_multiplier_);

  // input1 can be processed once because it is const
  const __m128i in1_src = _mm_set1_epi8(element_in);
  const __m256i in1_s16 = _mm256_cvtepi8_epi16(in1_src);
  const __m128i in1_s16_low = _mm256_extractf128_si256(in1_s16, 0);
  const __m128i in1_s16_high = _mm256_extractf128_si256(in1_s16, 1);
  const __m128i in1_zp_low = _mm_add_epi16(in1_s16_low, in1_zp_vec);
  const __m128i in1_zp_high = _mm_add_epi16(in1_s16_high, in1_zp_vec);
  __m256i tmp_in1 = _mm256_cvtepi16_epi32(in1_zp_low);
  __m128i in1_1 = _mm256_extractf128_si256(tmp_in1, 0);
  __m128i in1_2 = _mm256_extractf128_si256(tmp_in1, 1);
  tmp_in1 = _mm256_cvtepi16_epi32(in1_zp_high);
  __m128i in1_3 = _mm256_extractf128_si256(tmp_in1, 0);
  __m128i in1_4 = _mm256_extractf128_si256(tmp_in1, 1);

  AddInt8Rounding(&in1_1, &in1_2, &in1_3, &in1_4, in1_left_vec, params->in1_args_.right_shift_, in1_multiplier);

  int index = 0;
  for (; index <= size - 16; index += 16) {
    const __m128i in0_src = _mm_loadu_si128((__m128i_u *)(ptr_in + index));
    const __m256i in0_s16 = _mm256_cvtepi8_epi16(in0_src);
    const __m128i in0_s16_low = _mm256_extractf128_si256(in0_s16, 0);
    const __m128i in0_s16_high = _mm256_extractf128_si256(in0_s16, 1);
    const __m128i in0_zp_low = _mm_add_epi16(in0_s16_low, in0_zp_vec);
    const __m128i in0_zp_high = _mm_add_epi16(in0_s16_high, in0_zp_vec);

    __m256i tmp_in0 = _mm256_cvtepi16_epi32(in0_zp_low);
    __m128i in0_1 = _mm256_extractf128_si256(tmp_in0, 0);
    __m128i in0_2 = _mm256_extractf128_si256(tmp_in0, 1);
    tmp_in0 = _mm256_cvtepi16_epi32(in0_zp_high);
    __m128i in0_3 = _mm256_extractf128_si256(tmp_in0, 0);
    __m128i in0_4 = _mm256_extractf128_si256(tmp_in0, 1);

    AddInt8Rounding(&in0_1, &in0_2, &in0_3, &in0_4, in0_left_vec, params->in0_args_.right_shift_, in0_multiplier);

    /* calculate output */
    __m128i out1 = _mm_add_epi32(in0_1, in1_1);
    __m128i out2 = _mm_add_epi32(in0_2, in1_2);
    __m128i out3 = _mm_add_epi32(in0_3, in1_3);
    __m128i out4 = _mm_add_epi32(in0_4, in1_4);

    // Apply left shift
    out1 = _mm_slli_epi32(out1, params->out_left_shift_);
    out2 = _mm_slli_epi32(out2, params->out_left_shift_);
    out3 = _mm_slli_epi32(out3, params->out_left_shift_);
    out4 = _mm_slli_epi32(out4, params->out_left_shift_);

    // Apply the fixed-point part of the multiplier.
    out1 = _mm_qrdmulh_epi32(out1, out_multiplier);
    out2 = _mm_qrdmulh_epi32(out2, out_multiplier);
    out3 = _mm_qrdmulh_epi32(out3, out_multiplier);
    out4 = _mm_qrdmulh_epi32(out4, out_multiplier);

    // Apply right shift
    int32_t out_remainder_mask = (1ll << (params->out_right_shift_)) - 1;
    int32_t out_remainder_threshold = out_remainder_mask >> 1;
    const __m128i vout_remainder_mask = _mm_set1_epi32(out_remainder_mask);
    const __m128i vout_remainder_threshold = _mm_set1_epi32(out_remainder_threshold);
    const __m128i out1_remainder =
      _mm_add_epi32(_mm_and_si128(out1, vout_remainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), out1));
    out1 = _mm_sub_epi32(_mm_rshr_epi32(out1, params->out_right_shift_),
                         _mm_cmpgt_epi32(out1_remainder, vout_remainder_threshold));
    const __m128i out2_remainder =
      _mm_add_epi32(_mm_and_si128(out2, vout_remainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), out2));
    out2 = _mm_sub_epi32(_mm_rshr_epi32(out2, params->out_right_shift_),
                         _mm_cmpgt_epi32(out2_remainder, vout_remainder_threshold));
    const __m128i out3_remainder =
      _mm_add_epi32(_mm_and_si128(out3, vout_remainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), out3));
    out3 = _mm_sub_epi32(_mm_rshr_epi32(out3, params->out_right_shift_),
                         _mm_cmpgt_epi32(out3_remainder, vout_remainder_threshold));
    const __m128i out4_remainder =
      _mm_add_epi32(_mm_and_si128(out4, vout_remainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), out4));
    out4 = _mm_sub_epi32(_mm_rshr_epi32(out4, params->out_right_shift_),
                         _mm_cmpgt_epi32(out4_remainder, vout_remainder_threshold));

    __m128i out1_s16 = _mm_packs_epi32(out1, out2);
    __m128i out2_s16 = _mm_packs_epi32(out3, out4);

    __m128i out_s16_1 = _mm_add_epi16(out1_s16, out_zp_vec);
    __m128i out_s16_2 = _mm_add_epi16(out2_s16, out_zp_vec);
    __m128i out = _mm_packs_epi16(out_s16_1, out_s16_2);
    __m128i int8_out = _mm_max_epi8(min_vec, _mm_min_epi8(max_vec, out));

    _mm_storeu_si128((__m128i_u *)(output + index), int8_out);
  }
  for (; index < size; index++) {
    const int32_t in0_left = (ptr_in[index] + ptr_args->zp_) * in0_left_shift;
    const int32_t in1_left = (element_in + ele_args->zp_) * in1_left_shift;
    const int32_t in0 = MultiplyByMultiplierAndRightShift(in0_left, ptr_args->multiplier_, ptr_args->right_shift_);
    const int32_t in1 = MultiplyByMultiplierAndRightShift(in1_left, ele_args->multiplier_, ele_args->right_shift_);

    int32_t out = MultiplyByQuantizedMultiplier(in0 + in1, params->out_multiplier_, params->out_left_shift_,
                                                -params->out_right_shift_);
    out += params->out_zp_;
    output[index] = (int8_t)MSMAX(params->min_, MSMIN(out, params->max_));
  }
  return;
}
#endif
