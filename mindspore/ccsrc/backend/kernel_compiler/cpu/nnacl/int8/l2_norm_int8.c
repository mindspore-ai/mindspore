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
#include "nnacl/int8/l2_norm_int8.h"
#include <limits.h>
#include "nnacl/int8/fixed_point.h"
#include "nnacl/errorcode.h"

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
      int32_t out = RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(in * (1 << 7), multiplier), -shift);
      output_data[i * inner_size + k] = MSMIN(127, MSMAX(-128, out));
    }
  }
  return NNACL_OK;
}
