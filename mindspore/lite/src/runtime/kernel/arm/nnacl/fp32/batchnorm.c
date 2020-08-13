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

#include "nnacl/fp32/batchnorm.h"
#include <math.h>

void BatchNorm(float *output_ptr, const float *input_ptr, const float *mean_ptr, const float *variance_ptr, int task_id,
               BatchNormParameter *param) {
  for (int c = task_id; c < param->channel_; c += param->op_parameter_.thread_num_) {
    float variance_sqrt = sqrt(variance_ptr[c] + param->epsilon_);
    for (int u = 0; u < param->unit_; u++) {
      output_ptr[u * param->channel_ + c] = (input_ptr[u * param->channel_ + c] - mean_ptr[c]) / variance_sqrt;
    }
  }
}

void FusedBatchNorm(float *output_ptr, const float *input_ptr, const float *scale_ptr, const float *offest_ptr,
                    const float *mean_ptr, const float *variance_ptr, int task_id, BatchNormParameter *param) {
  for (int c = task_id; c < param->channel_; c += param->op_parameter_.thread_num_) {
    float variance_sqrt = sqrt(variance_ptr[c] + param->epsilon_);
    for (int u = 0; u < param->unit_; u++) {
      output_ptr[u * param->channel_ + c] =
        (input_ptr[u * param->channel_ + c] - mean_ptr[c]) / variance_sqrt * scale_ptr[c] + offest_ptr[c];
    }
  }
}
