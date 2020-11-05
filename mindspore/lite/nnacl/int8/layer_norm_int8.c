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

#include "nnacl/int8/layer_norm_int8.h"

/*
 * origin : (x-mean) / sqrt(variance + epsilon)  * gamma + beta
 * quant  : (x-mean) / sqrt(sum(x * x) - mean * mean) * gamma + beta
 *
 * */
int LayerNormInt8(const int8_t *src_data, const int8_t *gamma_data, const int32_t *beta_data, int8_t *dst_data,
                  bool affine, int outer_size, int inner_size, LayerNormQuantArg *quant_) {
  if (src_data == NULL || dst_data == NULL) {
    return NNACL_NULL_PTR;
  }

  if (affine && (gamma_data == NULL || beta_data == NULL)) {
    return NNACL_NULL_PTR;
  }

  for (int out_index = 0; out_index < outer_size; out_index++) {
    const int8_t *src = src_data + out_index * inner_size;
    int8_t *dst = dst_data + out_index * inner_size;
    int32_t mean = 0;
    int32_t square_mean = 0;
    for (int in_index = 0; in_index < inner_size; in_index++) {
      int32_t tmp_src = src[in_index] - quant_->in_quant_arg_.zp_;
      mean += tmp_src;
      square_mean += tmp_src * tmp_src;
    }
    mean = round(mean / inner_size);
    square_mean = round(square_mean / inner_size);

    int32_t variance_value = square_mean - mean * mean;

    int32_t multiplier;
    int32_t shift;
    GetSqrtQuantMultiplierExp(variance_value, -1, &multiplier, &shift);

    for (int in_index = 0; in_index < inner_size; in_index++) {
      int32_t in = src[in_index] - quant_->in_quant_arg_.zp_ - mean;
      int32_t tmp = RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(in * (1 << 7), multiplier), -shift);
      if (affine) {
        tmp = tmp * (gamma_data[in_index] - quant_->gamma_quant_arg_.zp_) + beta_data[in_index];
      }
      int32_t out = MultiplyByQuantizedMultiplier(tmp, quant_->multiplier_, quant_->shift_left_, quant_->shift_right_);
      dst[in_index] = (int8_t)MSMIN(quant_->output_activation_max_, MSMAX(quant_->output_activation_max_, out));
    }
  }
  return NNACL_OK;
}
