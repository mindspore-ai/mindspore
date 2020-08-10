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

#include "nnacl/int8/sigmoid_int8.h"

int SigmoidInt8(const int8_t *src, int length, int8_t *dst, SigmoidQuantArg *arg) {
  for (int i = 0; i < length; i++) {
    const int16_t input_value = src[i] - arg->input_zp;
    int16_t output;
    output = round(1 / arg->output_scale * (1 / (1 + exp(-arg->input_scale * input_value))));
    output += arg->output_zp;
    output = MSMIN(output, 127);
    output = MSMAX(output, -128);
    dst[i] = (int8_t)output;
  }
  return 0;
}
