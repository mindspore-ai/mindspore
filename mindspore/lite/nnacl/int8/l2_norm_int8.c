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
#include <limits.h>
#include "nnacl/int8/l2_norm_int8.h"
#include "nnacl/quantization/fixed_point.h"
#include "nnacl/errorcode.h"

void GetSqrtQuantMultiplierExp(int32_t input, int reverse_shift, int32_t *multiplier, int32_t *shift) {
  if (input <= 1) {
    *multiplier = INT_MAX;
    *shift = 0;
  }
  *shift = 11;
  while (input >= (1 << 29)) {
    input /= 4;
    ++*shift;
  }
  int max_left_shift_bits = CountLeadingSignBits(input);
  int left_shift_bit_pairs = max_left_shift_bits / 2 - 1;
  *shift -= left_shift_bit_pairs;
  input <<= 2 * left_shift_bit_pairs;
  int32_t fixedpoint_f3_input = input >> 1;  // sign: 1 bit, integer: 3 bit, fractional: 28 bit
  int32_t fp_f3_half_input = SaturatingRoundingMultiplyByPOT(fixedpoint_f3_input, -1);
  int32_t fp_f3_half_three = (1 << 28) + (1 << 27);
  int32_t tmp = (1 << 28);  // one
  for (int i = 0; i < 5; i++) {
    int32_t tmp3 = Rescale(SaturatingRoundingDoublingHighMul(tmp, SaturatingRoundingDoublingHighMul(tmp, tmp)), 9, 3);
    tmp = Rescale(SaturatingRoundingDoublingHighMul(fp_f3_half_three, tmp) -
                    SaturatingRoundingDoublingHighMul(fp_f3_half_input, tmp3),
                  6, 3);
  }
  int32_t fp_f0_half_sqrt_2 = 1518500250;  // sqrt(2) / 2
  tmp = SaturatingRoundingDoublingHighMul(tmp, fp_f0_half_sqrt_2);
  *multiplier = tmp;
  if (*shift < 0) {
    *multiplier <<= -*shift;
    *shift = 0;
  }
  *shift *= reverse_shift;
}

int32_t MultiplyByQuantizedMultiplier2(int32_t input, int32_t multiplier, int shift) {
  int left_shift = MSMAX(shift, 0);
  int right_shift = MSMAX(-shift, 0);
  return RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(input * (1 << left_shift), multiplier), right_shift);
}

int L2NormalizationInt8(const int8_t *input_data, int8_t *output_data, const L2NormParameter *param,
                        const L2NormQuantArg *quant_param, const int begin, const int end) {
  const int inner_size = param->shape_[param->shape_num_ - 1];

  for (int i = begin; i < end; ++i) {
    int32_t square_sum = 0.0f;
    for (int j = 0; j < inner_size; ++j) {
      int32_t in = input_data[i * inner_size + j] - quant_param->in_.zp_;
      square_sum += in * in;
    }
    int32_t multiplier;
    int32_t shift;
    GetSqrtQuantMultiplierExp(square_sum, -1, &multiplier, &shift);
    for (int k = 0; k < inner_size; ++k) {
      int32_t in = input_data[i * inner_size + k] - quant_param->in_.zp_;
      int32_t out = MultiplyByQuantizedMultiplier2(in, multiplier, shift + 7);
      output_data[i * inner_size + k] = MSMIN(127, MSMAX(-128, out));
    }
  }
  return NNACL_OK;
}
