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
#include "nnacl/int8/relux_int8.h"

void ReluXInt8(const int8_t *src, int length, int8_t *dst, const ReluXQuantArg *arg) {
  for (int i = 0; i < length; ++i) {
    if (src[i] <= arg->input_arg.zp_) {
      dst[i] = arg->output_arg.zp_;
      continue;
    }
    const int32_t input_val = src[i] - arg->input_arg.zp_;
    const int32_t scaled_input = SaturatingRoundingDoublingHighMul(input_val, arg->input_multiplier_);
    const int32_t shifted_input = RoundingDivideByPOT(scaled_input * (1U << arg->left_shift_), -arg->right_shift_);
    const int32_t output = shifted_input + arg->output_arg.zp_;
    dst[i] = (int8_t)MSMIN(output, arg->quantized_output_max);
  }
}
