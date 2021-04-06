/*
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "wrapper/int8/concat_int8_wrapper.h"

int ConcatInt8Run(void *cdata, int task_id) {
  ConcatInt8Args *args = (ConcatInt8Args *)cdata;
  int64_t real_dst_count = MSMIN(args->before_axis_size_ - task_id * args->count_unit_, args->count_unit_);
  if (real_dst_count <= 0) {
    return NNACL_OK;
  }
  Int8Concat(args->inputs_, args->output_, args->para_, args->axis_, real_dst_count, task_id);
  return NNACL_OK;
}
