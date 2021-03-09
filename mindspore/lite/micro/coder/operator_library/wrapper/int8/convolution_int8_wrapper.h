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

#ifndef MINDSPORE_LITE_MICRO_INT8_CONVOLUTION_WRAPPER_INT8_WRAPPER_H_
#define MINDSPORE_LITE_MICRO_INT8_CONVOLUTION_WRAPPER_INT8_WRAPPER_H_

#include "nnacl/errorcode.h"
#include "nnacl/conv_parameter.h"
#include "nnacl/matmul_parameter.h"
#include "nnacl/int8/conv_int8.h"

typedef struct {
  int8_t *input_data_;
  int8_t *packed_input_;
  int8_t *matmul_input_;
  int8_t *packed_weight_;
  const int32_t *bias_data_;
  int8_t *output_data_;
  int32_t *filter_zp_;
  int32_t *input_sum_;
  ConvParameter *conv_param_;
  MATMUL_OPT_R_FUNC matmul_func_;
  bool is_optimize_;
} ConvolutionInt8Args;

int ConvolutionInt8Run(void *cdata, int task_id);

#endif  // MINDSPORE_LITE_MICRO_INT8_CONVOLUTION_WRAPPER_INT8_WRAPPER_H_
