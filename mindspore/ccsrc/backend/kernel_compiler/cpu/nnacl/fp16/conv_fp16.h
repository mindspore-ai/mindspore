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
#ifndef MINDSPORE_NNACL_FP16_CONV_FP16_H_
#define MINDSPORE_NNACL_FP16_CONV_FP16_H_

#include <arm_neon.h>
#include "nnacl/conv_parameter.h"
#include "nnacl/matmul_parameter.h"
#include "nnacl/fp16/winograd_utils_fp16.h"
#include "nnacl/fp16/winograd_transform_fp16.h"

typedef float16_t *TmpBufferAddressFp16;
typedef float16_t *MatricesFp16;

#ifdef __cplusplus
extern "C" {
#endif

// fp16 convolution common (im2col+gemm)
void ConvFp16(const float16_t *input_data, float16_t *packed_input, const float16_t *packed_weight,
              const float16_t *bias_data, float16_t *col_major_input, float16_t *output_data, int task_id,
              const ConvParameter *conv_param);

void ConvOutNc8hw8Fp16(const float16_t *input_data, float16_t *packed_input, const float16_t *packed_weight,
                       const float16_t *bias_data, float16_t *col_major_input, float16_t *output_data, int task_id,
                       const ConvParameter *conv_param);

void Conv1x1OutNc8hw8MultiThreadByInputFp16(const float16_t *input, float16_t *pack_input, const float16_t *weight,
                                            const float16_t *bias, float16_t *output, int task_id,
                                            const MatMulParameter *param);

void Conv1x1OutNc8hw8MultiThreadByWeightFp16(const float16_t *input, float16_t *pack_input, const float16_t *weight,
                                             const float16_t *bias, float16_t *output, int task_id,
                                             const MatMulParameter *param);

// fp16 convolution winograd
void ConvWinogardFp16(const float16_t *input_data, const float16_t *trans_weight, const float16_t *bias_data,
                      float16_t *output_data, TmpBufferAddressFp16 *buffer_list, int task_id,
                      const ConvParameter *conv_param, InputTransFp16Func in_func, OutputTransFp16Func out_func);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP16_CONV_FP16_H_
