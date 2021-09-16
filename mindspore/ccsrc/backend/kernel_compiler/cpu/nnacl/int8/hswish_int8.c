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

#include "nnacl/int8/hswish_int8.h"

int16_t SaturatingLeftShift(int16_t value, int shift_num) {
  int32_t result = (int32_t)value * (1 << shift_num);
  return MSMAX(MSMIN(result, SHRT_MAX), SHRT_MIN);
}

int HSwishInt8(const int8_t *src, int length, int8_t *dst, const HswishQuantArg *arg) {
  for (int i = 0; i < length; i++) {
    const int16_t input_value = src[i] - arg->input_zp;
    const int16_t input_value_scale = input_value * (1 << 7);
    const int16_t input_value_on_preshift_output_scale =
      SaturatingRoundingDoublingHighMulInt16(input_value_scale, arg->output_multiplier_fixedpoint_int16);
    int16_t relu6_value = input_value_scale;
    if (arg->relu6_multiplier_exponent > 0) {
      relu6_value = SaturatingLeftShift(relu6_value, arg->relu6_multiplier_exponent - 1);
    }
    relu6_value = SaturatingRoundingDoublingHighMulInt16(relu6_value, arg->relu6_multiplier_fixedpoint_int16);

    if (arg->relu6_multiplier_exponent > 0) {
      relu6_value = SaturatingLeftShift(relu6_value, 1);
    }
    if (arg->relu6_multiplier_exponent < 0) {
      relu6_value = RoundingDivideByPOT(relu6_value, -arg->relu6_multiplier_exponent);
    }
    relu6_value = (size_t)(relu6_value + (1 << 15)) >> 1;
    const int16_t preshift_output_value =
      SaturatingRoundingDoublingHighMulInt16(relu6_value, input_value_on_preshift_output_scale);

    int16_t output = RoundingDivideByPOT(preshift_output_value, -arg->output_multiplier_exponent);
    output += arg->output_zp;
    output = MSMIN(output, 127);
    output = MSMAX(output, -128);
    dst[i] = (int8_t)output;
  }
  return NNACL_OK;
}
