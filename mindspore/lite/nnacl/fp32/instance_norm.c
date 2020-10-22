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

#include "nnacl/fp32/instance_norm.h"
#include <math.h>
#include "nnacl/instance_norm_parameter.h"
#include "nnacl/op_base.h"

void InstanceNormFp32(const void *input, const void *mean, const void *variance, InstanceNormParameter *param,
                      int task_id, void *output) {
  int units_per_thread = UP_DIV(param->unit_, param->op_parameter_.thread_num_);
  int completed_units = task_id * units_per_thread;
  if (completed_units >= param->unit_) {
    return;
  }
  int cur_unit = MSMIN(units_per_thread, param->unit_ - completed_units);
  int cur_offset = completed_units * param->channel_;
  for (int n = 0; n < param->batch_; n++) {
    for (int hw = 0; hw < cur_unit; hw++) {
      for (int c = 0; c < param->channel_; c++) {
        float variance_sqrt = sqrt(((const float *)variance)[n * param->channel_ + c] + param->epsilon_);
        ((float *)output)[cur_offset + c] =
          (((const float *)input)[cur_offset + c] - ((const float *)mean)[n * param->channel_ + c]) / variance_sqrt;
      }
      cur_offset += param->channel_;
    }
    cur_offset += (param->unit_ - cur_unit) * param->channel_;
  }
}
