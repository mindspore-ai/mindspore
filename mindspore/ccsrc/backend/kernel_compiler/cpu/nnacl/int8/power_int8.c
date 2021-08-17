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

#include "nnacl/int8/power_int8.h"

int PowerInt8(const int8_t *input, const int8_t *exp_ptr, int8_t *output, int count, const PowerParameter *param) {
  double input_scale = param->quant_arg_.in_args_.scale_;
  int input_zp = param->quant_arg_.in_args_.zp_;
  double output_scale = param->quant_arg_.out_args_.scale_;
  int output_zp = param->quant_arg_.out_args_.zp_;
  int act_min = param->quant_arg_.output_activation_min_;
  int act_max = param->quant_arg_.output_activation_max_;

  if (param->broadcast_) {
    for (int i = 0; i < count; ++i) {
      float input_val = input_scale * (input[i] - input_zp);
      float output_val = pow(param->scale_ * input_val + param->shift_, param->power_);
      int32_t output_scaled = round(output_val / output_scale) + output_zp;
      output[i] = (int8_t)MSMAX(act_min, MSMIN(output_scaled, act_max));
    }
  } else {
    double exp_scale = param->quant_arg_.exp_args_.scale_;
    int exp_zp = param->quant_arg_.exp_args_.zp_;
    for (int i = 0; i < count; ++i) {
      float input_val = input_scale * (input[i] - input_zp);
      float exp_val = exp_scale * (exp_ptr[i] - exp_zp);
      float output_val = pow(param->scale_ * input_val + param->shift_, exp_val);
      int32_t output_scaled = round(output_val / output_scale) + output_zp;
      output[i] = (int8_t)MSMAX(act_min, MSMIN(output_scaled, act_max));
    }
  }
  return 0;
}
