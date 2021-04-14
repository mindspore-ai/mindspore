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

#include "wrapper/int8/conv3x3_run_int8_wrapper.h"

int Conv3x3Int8Run(void *cdata, int task_id) {
  Conv3x3Int8Args *args = (Conv3x3Int8Args *)cdata;
  Conv3x3Int8(args->input_data, args->transed_weight, args->bias_data, args->output_data, args->tile_buffer,
              args->block_unit_buffer, args->tmp_dst_buffer, args->tmp_out, task_id, args->conv_param);
  return NNACL_OK;
}
