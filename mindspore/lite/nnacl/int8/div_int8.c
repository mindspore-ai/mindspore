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

#include "nnacl/int8/div_int8.h"

int DivInt8(int8_t *input0_data, int8_t *input1_data, int8_t *output_data, int64_t real_dst_count, DivQuantArg *para) {
  int index = 0;
  for (; index < real_dst_count; ++index) {
    const int32_t input0_val = para->in0_args_.zp_ + input0_data[index];
    const int32_t input1_val = para->in1_args_.zp_ + input1_data[index];
    if (input1_val == 0) {
      return NNACL_ERRCODE_DIVISOR_ZERO;
    }

    int recip_shift;
    const int32_t input1_inv = (input1_val > 0) ? ComputerReciprocal(input1_val, 31, &recip_shift)
                                                : -ComputerReciprocal(-input1_val, 31, &recip_shift);
    const int leading_bits = CountLeadingSignBits(input0_val);
    const int32_t raw_data =
      SaturatingRoundingDoublingHighMul(input0_val * (1 << (unsigned int)leading_bits), input1_inv);
    const int total_shift = para->output_shift_ - recip_shift - leading_bits;
    const int32_t raw_output =
      RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(raw_data, para->output_multiplier_), -total_shift) +
      para->out_args_.zp_;
    output_data[index] = (int8_t)MSMAX(para->output_activation_min_, MSMIN(raw_output, para->output_activation_max_));
  }
  return NNACL_OK;
}
