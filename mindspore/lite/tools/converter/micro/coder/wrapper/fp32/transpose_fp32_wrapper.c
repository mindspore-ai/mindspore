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

#include "wrapper/fp32/transpose_fp32_wrapper.h"
#include <string.h>
#include "nnacl/fp32/pack_fp32.h"
#include "nnacl/errorcode.h"

int DoTransposeNCHWToNHWC(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  TransposeFp32Args *args = (TransposeFp32Args *)cdata;
  const TransposeParameter *trans_param = args->transpose_param_;
  PackNCHWToNHWCFp32(args->input_, args->output_, args->batches_, args->plane_, args->channel_, task_id,
                     trans_param->op_parameter_.thread_num_);
  return NNACL_OK;
}

int DoTransposeNHWCToNCHW(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  TransposeFp32Args *args = (TransposeFp32Args *)cdata;
  const TransposeParameter *trans_param = args->transpose_param_;
  PackNHWCToNCHWFp32(args->input_, args->output_, args->batches_, args->plane_, args->channel_, task_id,
                     trans_param->op_parameter_.thread_num_);
  return NNACL_OK;
}
