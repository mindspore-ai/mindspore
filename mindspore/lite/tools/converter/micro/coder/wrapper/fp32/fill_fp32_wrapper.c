/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "wrapper/fp32/fill_fp32_wrapper.h"
#include "nnacl/errorcode.h"
#include "nnacl/base/fill_base.h"

int DoFillFp32(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  FillFp32Args *args = (FillFp32Args *)cdata;
  FillFp32(args->output_, args->size_, args->data_);
  return NNACL_OK;
}

int DoFillInt32(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  FillInt32Args *args = (FillInt32Args *)cdata;
  FillInt32(args->output_, args->size_, args->data_);
  return NNACL_OK;
}

int DoFillBool(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  FillBoolArgs *args = (FillBoolArgs *)cdata;
  FillBool(args->output_, args->size_, args->data_);
  return NNACL_OK;
}
