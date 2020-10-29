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

#include "nnacl/fp32/exp.h"
#include <math.h>
#include "nnacl/errorcode.h"

int Exp(const float *input_data, float *output_data, ExpParameter *parameter, int task_id) {
  if (parameter->scale_ == 1) {
    for (size_t i = task_id; i < parameter->element_num_; i += parameter->thread_num_) {
      output_data[i] = expf(input_data[i]);
    }
  } else {
    for (size_t i = task_id; i < parameter->element_num_; i += parameter->thread_num_) {
      output_data[i] = expf(input_data[i] * parameter->in_scale_);
    }
  }
  if (parameter->out_scale_ != 1) {
    for (size_t i = task_id; i < parameter->element_num_; i += parameter->thread_num_) {
      output_data[i] = output_data[i] * parameter->out_scale_;
    }
  }
  return NNACL_OK;
}
