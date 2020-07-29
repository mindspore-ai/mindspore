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

#include "src/runtime/kernel/arm/opclib/int8/reshape_int8.h"
#include <string.h>

void Reshape(int8_t *input_ptr, int8_t *output_ptr, size_t data_size, int input_num, QuantArg in_quant_arg,
             QuantArg out_quant_arg) {
  if (in_quant_arg.scale_ == out_quant_arg.scale_ && in_quant_arg.zp_ == out_quant_arg.zp_) {
    memcpy(output_ptr, input_ptr, data_size);
  } else {
    float output_inverse_scale = 1.f / out_quant_arg.scale_;
    float scale = in_quant_arg.scale_ * output_inverse_scale;
    float bias = -in_quant_arg.zp_ * scale;
    int32_t output_zp = out_quant_arg.zp_;
    for (int i = 0; i < input_num; i++) {
      int32_t output_tmp = round(input_ptr[i] * scale + bias) + output_zp;
      if (output_tmp > 127) {
        output_ptr[i] = 127;
      } else if (output_tmp < -128) {
        output_ptr[i] = -128;
      } else {
        output_ptr[i] = (int8_t)output_tmp;
      }
    }
  }
}

