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

#include "nnacl/int8/reshape_int8.h"
#include "nnacl/reshape_parameter.h"
#include <string.h>

void Int8Reshape(const int8_t *input_ptr, int8_t *output_ptr, int64_t real_dst_count, ReshapeQuantArg para) {
  if (para.in_args_.scale_ == para.out_args_.scale_ && para.in_args_.zp_ == para.out_args_.zp_) {
    memcpy(output_ptr, input_ptr, real_dst_count);
  } else {
    const float output_inverse_scale = 1.f / para.out_args_.scale_;
    float scale = para.in_args_.scale_ * output_inverse_scale;
    float bias = -para.in_args_.zp_ * scale;
    int32_t output_zp = para.out_args_.zp_;
    for (int i = 0; i < real_dst_count; i++) {
      int32_t output_tmp = round(input_ptr[i] * scale + bias) + output_zp;
      if (output_tmp > para.output_activation_max_) {
        output_ptr[i] = para.output_activation_max_;
      } else if (output_tmp < para.output_activation_min_) {
        output_ptr[i] = para.output_activation_min_;
      } else {
        output_ptr[i] = (int8_t)output_tmp;
      }
    }
  }
}
