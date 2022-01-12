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

#include "nnacl/fp32/scatter_nd_fp32.h"
#include <string.h>
#include "nnacl/errorcode.h"

int DoScatterND(void *output, const void *update, int *output_unit_offsets, const ScatterNDParameter *param,
                int task_id) {
  if (param->op_parameter.thread_num_ == 0) {
    return NNACL_ERR;
  }
  int unit_per_thread = UP_DIV(param->num_unit, param->op_parameter.thread_num_);
  int begin = unit_per_thread * task_id;
  int end = MSMIN(begin + unit_per_thread, param->num_unit);

  int data_type_len = param->data_type_len;
  for (int i = begin; i < end; i++) {
    (void)memcpy((int8_t *)output + output_unit_offsets[i] * data_type_len,
                 (int8_t *)update + i * param->unit_size * data_type_len, param->unit_size * data_type_len);
  }
  return NNACL_OK;
}
