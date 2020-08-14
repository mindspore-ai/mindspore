/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
// * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "src/runtime/kernel/arm/nnacl/caffeprelu.h"

void CaffePRelu(float *input, float *output, CaffePReluParameter *prelu_param_, int task_id) {
  int block = (int)(prelu_param_->input_num_ / prelu_param_->op_parameter_.thread_num_);
  int start = task_id * block;
  int end = start + block;
  if (task_id == prelu_param_->op_parameter_.thread_num_ - 1) {
    end = prelu_param_->input_num_;
  }
  for (int i = start; i < end; i++) {
    if (input[i] > 0) {
      output[i] = input[i];
    } else {
      if (!prelu_param_->channeShared) {
        int temp = i / (prelu_param_->input_num_ / prelu_param_->channel_num_);
        output[i] = input[i] * prelu_param_->negtive_slope_[temp];
      } else {
        output[i] = input[i] * prelu_param_->negtive_slope_[0];
      }
    }
  }
}
