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

#include "wrapper/fp32/split_fp32_wrapper.h"
#include "nnacl/errorcode.h"

int DoSplitRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  SplitFp32Args *args = (SplitFp32Args *)cdata;
  int thread_n_stride = UP_DIV(args->num_unit_, args->param->op_parameter_.thread_num_);
  int num_unit_thread = MSMIN(thread_n_stride, args->num_unit_ - task_id * thread_n_stride);
  if (num_unit_thread <= 0) {
    return NNACL_OK;
  }

  int thread_offset = task_id * thread_n_stride;
  return DoSplit(args->input_ptr_, args->output_ptr_, args->in_tensor_shape, thread_offset, num_unit_thread,
                 args->param, args->data_type_size);
}
