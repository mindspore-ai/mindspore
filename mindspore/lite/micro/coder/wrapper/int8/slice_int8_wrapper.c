/**
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

#include "wrapper/int8/slice_int8_wrapper.h"
#include "nnacl/int8/slice_int8.h"

int SliceInt8Run(void *cdata, int task_id) {
  SliceArgs *args = (SliceArgs *)(cdata);
  int ret = SliceInt8(args->input_data_, args->output_data_, args->param_, task_id);
  return ret;
}
