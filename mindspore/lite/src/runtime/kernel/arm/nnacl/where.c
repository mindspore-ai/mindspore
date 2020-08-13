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
#include "nnacl/where.h"

void Where(bool *input, float *input1, float *input2, float *output, WhereParameter *where_param_, int task_id) {
  for (int i = task_id; i < where_param_->number_; i += where_param_->op_parameter_.thread_num_) {
    if (input[where_param_->num_ > 1 ? i : 0] == true) {
      output[i] = input1[where_param_->num1_ > 1 ? i : 0];
    } else {
      output[i] = input2[where_param_->num2_ > 1 ? i : 0];
    }
  }
}
