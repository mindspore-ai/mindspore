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

#ifndef MINDSPORE_LITE_MICRO_INT8_CONV3x3_WRAPPER_INT8_WRAPPER_H_
#define MINDSPORE_LITE_MICRO_INT8_CONV3x3_WRAPPER_INT8_WRAPPER_H_

#include "nnacl/errorcode.h"
#include "nnacl/conv_parameter.h"
#include "nnacl/int8/conv3x3_int8.h"

typedef struct {
  int16_t *input_data;
  int16_t *transed_weight;
  int32_t *bias_data;
  int8_t *output_data;
  int16_t *tile_buffer;
  int16_t *block_unit_buffer;
  int32_t *tmp_dst_buffer;
  int8_t *tmp_out;
  ConvParameter *conv_param;
} Conv3x3Int8Args;

int Conv3x3Int8Run(void *cdata, int task_id);

#endif  // MINDSPORE_LITE_MICRO_INT8_CONV3x3_WRAPPER_INT8_WRAPPER_H_
