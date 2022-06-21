/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "wrapper/fp32/activation_fp32_wrapper.h"
#include "nnacl/fp32/activation_fp32.h"
#include "nnacl/errorcode.h"

int DoSigmoid(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  ActivationFp32Args *args = (ActivationFp32Args *)cdata;
  const ActivationParameter *activation_param = args->activation_param_;
  int length = args->length_;
  int stride = UP_DIV(length, activation_param->op_parameter_.thread_num_);
  int count = MSMIN(stride, length - stride * task_id);
  if (count <= 0) {
    return NNACL_OK;
  }
  Sigmoid(args->input_ + stride * task_id, count, args->output_ + stride * task_id);
  return NNACL_OK;
}
