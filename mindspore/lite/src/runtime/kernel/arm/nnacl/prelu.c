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
#include "nnacl/prelu.h"

void PRelu(float *input, float *output, PReluParameter *prelu_param_, int task_id) {
  for (int i = task_id; i < prelu_param_->input_num_; i += prelu_param_->op_parameter_.thread_num_) {
    if (input[i] <= 0) {
      output[i] = input[i] * prelu_param_->negtive_slope_[0];
    } else {
      output[i] = input[i];
    }
  }
}
