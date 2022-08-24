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

#include "wrapper/fp32/scale_fp32_wrapper.h"
#include "nnacl/fp32/scale_fp32.h"
#include "nnacl/errorcode.h"

int DoScaleRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  ScaleFp32Args *args = (ScaleFp32Args *)cdata;
  const ScaleParameter *scale_param = args->scale_param_;
  int block[C2NUM] = {args->split_points_[task_id], args->split_points_[task_id + 1]};
  DoScaleFp32(args->input_, args->scale_, args->offset_, args->output_, scale_param, block);
  return NNACL_OK;
}
