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
#include "nnacl/fp32/leaky_relu.h"

void DoLeakyRelu(float *input, float *output, LeakyReluParameter *param, int task_id) {
  for (int i = task_id; i < param->input_num_; i += param->op_parameter_.thread_num_) {
    if (input[i] <= 0) {
      output[i] = input[i] * param->slope_[0];
    } else {
      output[i] = input[i];
    }
  }
}
