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

#ifndef MINDSPORE_LITE_MICRO_CODER_OPERATOR_LIBRARY_WRAPPER_INT8_BATCHNORM_INT8_WRAPPER_H_
#define MINDSPORE_LITE_MICRO_CODER_OPERATOR_LIBRARY_WRAPPER_INT8_BATCHNORM_INT8_WRAPPER_H_

#include <stdint.h>
#include "nnacl/batchnorm_parameter.h"
typedef struct BatchNormArgs {
  int8_t *in_addr_;
  int8_t *out_addr_;
  float *alpha_addr_;
  float *beta_addr_;
  BatchNormParameter *batchnorm_param_;
} BatchNormArgs;

int BatchNormInt8Run(void *cdata, int task_id);

#endif  // MINDSPORE_LITE_MICRO_CODER_OPERATOR_LIBRARY_WRAPPER_INT8_BATCHNORM_INT8_WRAPPER_H_
