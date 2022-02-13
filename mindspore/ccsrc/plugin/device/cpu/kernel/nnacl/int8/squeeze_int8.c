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

#include "nnacl/int8/squeeze_int8.h"

void SqueezeInt8(const int8_t *input_ptr, int8_t *output_ptr, const SqueezeQuantArg *quant_Squeeze_parm, int num,
                 int task_id, int thread_count) {
  float output_scale = quant_Squeeze_parm->out_quant_args_->scale_;
  const float output_inverse_scale = 1.f / output_scale;
  QuantArg *input_quant = quant_Squeeze_parm->in_quant_args_;
  int output_zp = quant_Squeeze_parm->out_quant_args_->zp_;

  const int i = 0;
  for (int j = task_id; j < num; j += thread_count) {
    float scale = input_quant[i].scale_ * output_inverse_scale;
    float bias = -input_quant[i].zp_ * scale;
    int32_t output_tmp = round(input_ptr[j] * scale + bias) + output_zp;
    if (output_tmp > 127) {
      output_ptr[j] = 127;
    } else if (output_tmp < -128) {
      output_ptr[j] = -128;
    } else {
      output_ptr[j] = (int8_t)output_tmp;
    }
  }
}
