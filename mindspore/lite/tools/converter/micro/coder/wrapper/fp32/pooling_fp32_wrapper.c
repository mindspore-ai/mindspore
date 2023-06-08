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

#include "wrapper/fp32/pooling_fp32_wrapper.h"
#include "nnacl/fp32/pooling_fp32.h"
#include "nnacl/errorcode.h"

int DoMaxPooling(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  PoolingFp32Args *args = (PoolingFp32Args *)cdata;
  return MaxPooling(args->input_, args->output_, args->pooling_param_, args->pooling_args_, task_id,
                    args->pooling_param_->op_parameter_.thread_num_);
}
