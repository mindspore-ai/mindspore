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

#include <math.h>
#include "nnacl/int8/arithmetic_self_int8.h"
#ifdef ENABLE_NEON
#include <arm_neon.h>
#include "nnacl/int8/common_func_int8.h"
#endif
#include "nnacl/int8/fixed_point.h"

int Int8ElementFloor(int8_t *input, int8_t *output, int element_size, ArithSelfQuantArg para) {
  float in_scale = para.in_args_.scale_;
  int32_t in_zp = para.in_args_.zp_;
  float out_scale = para.out_args_.scale_;
  int32_t out_zp = para.out_args_.zp_;
  float bias = in_zp * in_scale;
  for (int i = 0; i < element_size; i++) {
    int32_t output_tmp = round(floorf(input[i] * in_scale + bias) / out_scale) + out_zp;
    if (output_tmp > para.output_activation_max_) {
      output[i] = para.output_activation_max_;
    } else if (output_tmp < para.output_activation_min_) {
      output[i] = para.output_activation_min_;
    } else {
      output[i] = (int8_t)output_tmp;
    }
  }
  return NNACL_OK;
}

int Int8ElementRound(int8_t *input, int8_t *output, int element_size, ArithSelfQuantArg para) {
  float in_scale = para.in_args_.scale_;
  int32_t in_zp = para.in_args_.zp_;
  float out_scale = para.out_args_.scale_;
  int32_t out_zp = para.out_args_.zp_;
  float bias = in_zp * in_scale;
  for (int i = 0; i < element_size; i++) {
    int32_t output_tmp = round(round(input[i] * in_scale + bias) / out_scale) + out_zp;
    if (output_tmp > para.output_activation_max_) {
      output[i] = para.output_activation_max_;
    } else if (output_tmp < para.output_activation_min_) {
      output[i] = para.output_activation_min_;
    } else {
      output[i] = (int8_t)output_tmp;
    }
  }
  return NNACL_OK;
}

int Int8ElementCeil(int8_t *input, int8_t *output, int element_size, ArithSelfQuantArg para) {
  float in_scale = para.in_args_.scale_;
  int32_t in_zp = para.in_args_.zp_;
  float out_scale = para.out_args_.scale_;
  int32_t out_zp = para.out_args_.zp_;
  float bias = in_zp * in_scale;
  for (int i = 0; i < element_size; i++) {
    int32_t output_tmp = round(ceil(input[i] * in_scale + bias) / out_scale) + out_zp;
    if (output_tmp > para.output_activation_max_) {
      output[i] = para.output_activation_max_;
    } else if (output_tmp < para.output_activation_min_) {
      output[i] = para.output_activation_min_;
    } else {
      output[i] = (int8_t)output_tmp;
    }
  }
  return NNACL_OK;
}

int Int8ElementAbs(int8_t *input, int8_t *output, int element_size, ArithSelfQuantArg para) {
  float in_scale = para.in_args_.scale_;
  int32_t in_zp = para.in_args_.zp_;
  float out_scale = para.out_args_.scale_;
  int32_t out_zp = para.out_args_.zp_;
  float bias = in_zp * in_scale;
  for (int i = 0; i < element_size; i++) {
    int32_t output_tmp = round(fabsf(input[i] * in_scale + bias) / out_scale) + out_zp;
    if (output_tmp > para.output_activation_max_) {
      output[i] = para.output_activation_max_;
    } else if (output_tmp < para.output_activation_min_) {
      output[i] = para.output_activation_min_;
    } else {
      output[i] = (int8_t)output_tmp;
    }
  }
  return NNACL_OK;
}

int Int8ElementSin(int8_t *input, int8_t *output, int element_size, ArithSelfQuantArg para) {
  float in_scale = para.in_args_.scale_;
  int32_t in_zp = para.in_args_.zp_;
  float out_scale = para.out_args_.scale_;
  int32_t out_zp = para.out_args_.zp_;
  float bias = in_zp * in_scale;
  for (int i = 0; i < element_size; i++) {
    int32_t output_tmp = round(sinf(input[i] * in_scale + bias) / out_scale) + out_zp;
    if (output_tmp > para.output_activation_max_) {
      output[i] = para.output_activation_max_;
    } else if (output_tmp < para.output_activation_min_) {
      output[i] = para.output_activation_min_;
    } else {
      output[i] = (int8_t)output_tmp;
    }
  }
  return NNACL_OK;
}

int Int8ElementCos(int8_t *input, int8_t *output, int element_size, ArithSelfQuantArg para) {
  float in_scale = para.in_args_.scale_;
  int32_t in_zp = para.in_args_.zp_;
  float out_scale = para.out_args_.scale_;
  int32_t out_zp = para.out_args_.zp_;
  float bias = in_zp * in_scale;
  for (int i = 0; i < element_size; i++) {
    int32_t output_tmp = round(cosf(input[i] * in_scale + bias) / out_scale) + out_zp;
    if (output_tmp > para.output_activation_max_) {
      output[i] = para.output_activation_max_;
    } else if (output_tmp < para.output_activation_min_) {
      output[i] = para.output_activation_min_;
    } else {
      output[i] = (int8_t)output_tmp;
    }
  }
  return NNACL_OK;
}

int Int8ElementLog(int8_t *input, int8_t *output, int element_size, ArithSelfQuantArg para) {
  float in_scale = para.in_args_.scale_;
  int32_t in_zp = para.in_args_.zp_;
  float out_scale = para.out_args_.scale_;
  int32_t out_zp = para.out_args_.zp_;
  float bias = in_zp * in_scale;
  for (int i = 0; i < element_size; i++) {
    int32_t output_tmp = round(logf(input[i] * in_scale + bias) / out_scale) + out_zp;
    if (output_tmp > para.output_activation_max_) {
      output[i] = para.output_activation_max_;
    } else if (output_tmp < para.output_activation_min_) {
      output[i] = para.output_activation_min_;
    } else {
      output[i] = (int8_t)output_tmp;
    }
  }
  return NNACL_OK;
}

int Int8ElementSqrt(int8_t *input, int8_t *output, int element_size, ArithSelfQuantArg para) {
  float in_scale = para.in_args_.scale_;
  int32_t in_zp = para.in_args_.zp_;
  float out_scale = para.out_args_.scale_;
  int32_t out_zp = para.out_args_.zp_;
  float bias = in_zp * in_scale;
  for (int i = 0; i < element_size; i++) {
    float input_f32 = input[i] * in_scale + bias;
    if (input_f32 < 0) {
      return NNACL_ERRCODE_SQRT_NEGATIVE;
    }
    int32_t output_tmp = round(sqrtf(input_f32) / out_scale) + out_zp;
    if (output_tmp > para.output_activation_max_) {
      output[i] = para.output_activation_max_;
    } else if (output_tmp < para.output_activation_min_) {
      output[i] = para.output_activation_min_;
    } else {
      output[i] = (int8_t)output_tmp;
    }
  }
  return NNACL_OK;
}

int Int8ElementRsqrt(int8_t *input, int8_t *output, int element_size, ArithSelfQuantArg para) {
  float in_scale = para.in_args_.scale_;
  int32_t in_zp = para.in_args_.zp_;
  float out_scale = para.out_args_.scale_;
  int32_t out_zp = para.out_args_.zp_;
  float bias = in_zp * in_scale;
  for (int i = 0; i < element_size; i++) {
    float input_f32 = input[i] * in_scale + bias;
    if (input_f32 <= 0) {
      return NNACL_ERRCODE_RSQRT_NEGATIVE_OR_ZERO;
    }
    int32_t output_tmp = round(1.f / (sqrtf(input_f32) * out_scale)) + out_zp;
    if (output_tmp > para.output_activation_max_) {
      output[i] = para.output_activation_max_;
    } else if (output_tmp < para.output_activation_min_) {
      output[i] = para.output_activation_min_;
    } else {
      output[i] = (int8_t)output_tmp;
    }
  }
  return NNACL_OK;
}

#ifdef ENABLE_NEON

int16x4_t ClacSumHalfWord(int32x4_t scaled_input, int32x4_t left_shift_out_vec, int32x4_t output_multiplier_vec,
                          ArithSelfQuantArg para) {
  int32x4_t input_scale = vmulq_s32(scaled_input, scaled_input);
  int32x4_t raw_sum = RoundingDivideByPOTInt32x4(
    SaturatingRoundingDoublingHighMulInt32x4(vmulq_s32(input_scale, left_shift_out_vec), output_multiplier_vec),
    para.shift_right_);
  raw_sum = vaddq_s32(raw_sum, vdupq_n_s32(para.out_args_.zp_));
  raw_sum = vmaxq_s32(raw_sum, vdupq_n_s32(para.output_activation_min_));
  raw_sum = vminq_s32(raw_sum, vdupq_n_s32(para.output_activation_max_));
  return vqmovn_s32(raw_sum);
}

void SquareInt8NEON(int8_t *input_data, int8_t *output_data, int64_t element_size, ArithSelfQuantArg para, int *index) {
  int32x4_t output_multiplier_vec = vdupq_n_s32(para.output_multiplier_);
  int32x4_t left_shift_out_vec = vdupq_n_s32(1 << para.shift_left_);

  for (; (*index) <= element_size - 8; (*index) += 8) {
    int16x8_t input_val = LoadAndAddOffset(input_data, *index, para.in_args_.zp_);
    int32x4_t input_low = vmovl_s16(vget_low_s16(input_val));
    int32x4_t input_high = vmovl_s16(vget_high_s16(input_val));

    int16x4_t sum_low = ClacSumHalfWord(input_low, left_shift_out_vec, output_multiplier_vec, para);
    int16x4_t sum_high = ClacSumHalfWord(input_high, left_shift_out_vec, output_multiplier_vec, para);

    int16x8_t res_s16 = vcombine_s16(sum_low, sum_high);
    int8x8_t res_u8_n0 = vqmovn_s16(res_s16);
    vst1_s8(output_data, res_u8_n0);
    output_data += 8;
  }
}
#endif

int Int8ElementSquare(int8_t *input, int8_t *output, int element_size, ArithSelfQuantArg para) {
  int32_t in_zp = para.in_args_.zp_;
  int32_t out_zp = para.out_args_.zp_;

  int index = 0;
#ifdef ENABLE_NEON
  SquareInt8NEON(input, output, element_size, para, &index);
#endif
  for (; index < element_size; index++) {
    const int32_t input_val = input[index] + in_zp;
    int32_t output_tmp = RoundingDivideByPOT(
      SaturatingRoundingDoublingHighMul(input_val * input_val * (1 << para.shift_left_), para.output_multiplier_),
      para.shift_right_);
    output_tmp += out_zp;
    if (output_tmp > para.output_activation_max_) {
      output[index] = para.output_activation_max_;
    } else if (output_tmp < para.output_activation_min_) {
      output[index] = para.output_activation_min_;
    } else {
      output[index] = (int8_t)output_tmp;
    }
  }
  return NNACL_OK;
}

int Int8ElementLogicalNot(int8_t *input, int8_t *output, int element_size, ArithSelfQuantArg para) {
  float in_scale = para.in_args_.scale_;
  int32_t in_zp = para.in_args_.zp_;
  float out_scale = para.out_args_.scale_;
  int32_t out_zp = para.out_args_.zp_;
  float bias = in_zp * in_scale;
  for (int i = 0; i < element_size; i++) {
    int32_t output_tmp = round(((float)(!(bool)(input[i] * in_scale + bias))) / out_scale) + out_zp;
    if (output_tmp > para.output_activation_max_) {
      output[i] = para.output_activation_max_;
    } else if (output_tmp < para.output_activation_min_) {
      output[i] = para.output_activation_min_;
    } else {
      output[i] = (output_tmp);
    }
  }
  return NNACL_OK;
}

int Int8ElementReciprocal(int8_t *input, int8_t *output, int element_size, ArithSelfQuantArg para) {
  float in_scale = para.in_args_.scale_;
  int32_t in_zp = para.in_args_.zp_;
  float out_scale = para.out_args_.scale_;
  int32_t out_zp = para.out_args_.zp_;
  float bias = in_zp * in_scale;
  for (int i = 0; i < element_size; i++) {
    float input_f32 = input[i] * in_scale + bias;
    if (input_f32 == 0.0f) {
      return NNACL_ERR;
    }
    int32_t output_tmp = round(1.f / (input_f32 * out_scale)) + out_zp;
    if (output_tmp > para.output_activation_max_) {
      output[i] = para.output_activation_max_;
    } else if (output_tmp < para.output_activation_min_) {
      output[i] = para.output_activation_min_;
    } else {
      output[i] = (int8_t)output_tmp;
    }
  }
  return NNACL_OK;
}
