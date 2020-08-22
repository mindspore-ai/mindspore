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

#include <memory.h>
#include "nnacl/errorcode.h"
#include "nnacl/prior_box.h"

int PriorBox(const float *input_data, float *output_data, const size_t size, const int tid, const int thread_num) {
  if (thread_num == 0) {
    return NNACL_ERR;
  }
  size_t unit_size = size / thread_num;
  if (tid == thread_num - 1) {
    size_t tail_size = size - unit_size * tid;
    (void)memcpy(output_data + tid * unit_size, input_data + tid * unit_size, tail_size * sizeof(float));
  } else {
    (void)memcpy(output_data + tid * unit_size, input_data + tid * unit_size, unit_size * sizeof(float));
  }
  return NNACL_OK;
}
