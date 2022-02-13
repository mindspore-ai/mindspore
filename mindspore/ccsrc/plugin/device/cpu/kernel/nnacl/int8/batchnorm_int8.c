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

#include "nnacl/int8/batchnorm_int8.h"
#include <math.h>
#include "nnacl/batchnorm_parameter.h"

void BatchNormInt8(int8_t *output_ptr, const int8_t *input_ptr, const float *alpha_ptr, const float *beta_ptr,
                   int task_id, BatchNormParameter *param) {
  int unit_st = task_id * param->unit_;
  int unit_end = MSMIN((task_id + 1) * param->unit_, param->units_);
  for (int u = unit_st; u < unit_end; u++) {
    for (int c = 0; c < param->channel_; c++) {
      int32_t output_tmp = round(input_ptr[u * param->channel_ + c] * alpha_ptr[c] + beta_ptr[c]);
      output_tmp = output_tmp > 127 ? 127 : output_tmp;
      output_tmp = output_tmp < -128 ? -128 : output_tmp;
      output_ptr[u * param->channel_ + c] = (int8_t)output_tmp;
    }
  }
}
