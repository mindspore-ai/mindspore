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
#include "src/runtime/kernel/arm/opclib/int8/arithmetic_self_int8.h"

int ElementFloor(int8_t *input, int8_t *output, int element_size, ArithSelfQuantArg para) {
  if (para.in_args_.scale_ == para.out_args_.scale_ && para.in_args_.zp_ == para.out_args_.zp_) {
    for (int i = 0; i < element_size; i++) {
      output[i] = floorf(input[i]);
    }
  } else {
    float in_scale = para.in_args_.scale_;
    int32_t in_zp = para.in_args_.zp_;
    float out_scale = para.out_args_.scale_;
    int32_t out_zp = para.out_args_.zp_;
    float bias = -in_zp * in_scale;
    for (int i = 0; i < element_size; i++) {
      int32_t output_tmp = round(floorf(input[i] * in_scale + bias) / out_scale) + out_zp;
      if (output_tmp > para.output_activation_max_) {
        output[i] = para.output_activation_max_;
      } else if (output_tmp < para.output_activation_min_) {
        output[i] = para.output_activation_min_;
      } else {
        output[i] = static_cast<int8_t>(output_tmp);
      }
    }
  }
  return OPCLIB_OK;
}

int ElementRound(int8_t *input, int8_t *output, int element_size, ArithSelfQuantArg para) {
  if (para.in_args_.scale_ == para.out_args_.scale_ && para.in_args_.zp_ == para.out_args_.zp_) {
    for (int i = 0; i < element_size; i++) {
      output[i] = round(input[i]);
    }
  } else {
    float in_scale = para.in_args_.scale_;
    int32_t in_zp = para.in_args_.zp_;
    float out_scale = para.out_args_.scale_;
    int32_t out_zp = para.out_args_.zp_;
    float bias = -in_zp * in_scale;
    for (int i = 0; i < element_size; i++) {
      int32_t output_tmp = round(round(input[i] * in_scale + bias) / out_scale) + out_zp;
      if (output_tmp > para.output_activation_max_) {
        output[i] = para.output_activation_max_;
      } else if (output_tmp < para.output_activation_min_) {
        output[i] = para.output_activation_min_;
      } else {
        output[i] = static_cast<int8_t>(output_tmp);
      }
    }
  }
  return OPCLIB_OK;
}

int ElementCeil(int8_t *input, int8_t *output, int element_size, ArithSelfQuantArg para) {
  if (para.in_args_.scale_ == para.out_args_.scale_ && para.in_args_.zp_ == para.out_args_.zp_) {
    for (int i = 0; i < element_size; i++) {
      output[i] = ceil(input[i]);
    }
  } else {
    float in_scale = para.in_args_.scale_;
    int32_t in_zp = para.in_args_.zp_;
    float out_scale = para.out_args_.scale_;
    int32_t out_zp = para.out_args_.zp_;
    float bias = -in_zp * in_scale;
    for (int i = 0; i < element_size; i++) {
      int32_t output_tmp = round(ceil(input[i] * in_scale + bias) / out_scale) + out_zp;
      if (output_tmp > para.output_activation_max_) {
        output[i] = para.output_activation_max_;
      } else if (output_tmp < para.output_activation_min_) {
        output[i] = para.output_activation_min_;
      } else {
        output[i] = static_cast<int8_t>(output_tmp);
      }
    }
  }
  return OPCLIB_OK;
}
