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

#include "wrapper/fp32/conv_winograd_fp32_wrapper.h"
#include "nnacl/errorcode.h"

int ConvWinogradFp32Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  ConvWinogradFp32Args *args = (ConvWinogradFp32Args *)cdata;
  ConvWinogardFp32(args->input_data_, args->trans_weight_, args->bias_data_, args->output_data_, args->buffer_list_,
                   task_id, args->conv_param_, args->trans_func_);
  return NNACL_OK;
}
