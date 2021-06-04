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

#ifndef MINDSPORE_LITE_MICRO_CODER_WRAPPER_DECONVOLUTION_FP32_WRAPPER_H_
#define MINDSPORE_LITE_MICRO_CODER_WRAPPER_DECONVOLUTION_FP32_WRAPPER_H_

#include "nnacl/errorcode.h"
#include "nnacl/conv_parameter.h"
#include "nnacl/matmul_parameter.h"

typedef struct {
  const float *packed_input_;
  const float *packed_weight_;
  const float *packed_bias_;
  float *packed_output_;
  float *output_;
  float *tmp_buffer_;
  const MatMulParameter *matmul_param_;
  const ConvParameter *conv_param_;
} DeConvFp32Args;

#ifdef __cplusplus
extern "C" {
#endif

int DoDeconvFp32(const float *packed_input, const float *packed_weight, const float *packed_bias, float *packed_output,
                 float *output, float *tmp_ori_buffer, const MatMulParameter *matmul_param,
                 const ConvParameter *conv_param, int task_id);

int DeConvFp32Run(void *cdata, int task_id, float lhs_scale, float rhs_scale);

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_LITE_MICRO_CODER_WRAPPER_DECONVOLUTION_FP32_WRAPPER_H_
