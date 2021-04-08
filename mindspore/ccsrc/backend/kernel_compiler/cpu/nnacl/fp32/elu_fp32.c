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

#include "nnacl/fp32/elu_fp32.h"
#include <math.h>
#include "nnacl/errorcode.h"

void Calculate_Data(const float *input_data, float *output_data, int num, const EluParameter *parameter) {
  output_data[num] = input_data[num] < 0 ? parameter->alpha_ * expm1(input_data[num]) : input_data[num];
}

int Elu(const float *input_data, float *output_data, const EluParameter *parameter, int task_id) {
  for (size_t i = task_id; i < parameter->in_size_; i += parameter->op_parameter_.thread_num_) {
    Calculate_Data(input_data, output_data, i, parameter);
  }
  return NNACL_OK;
}
