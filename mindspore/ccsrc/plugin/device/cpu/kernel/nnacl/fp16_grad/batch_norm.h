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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NNACL_FP16_GRAD_BATCH_NORM_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NNACL_FP16_GRAD_BATCH_NORM_H_

#include "nnacl/op_base.h"

#ifdef __cplusplus
extern "C" {
#endif

void var2InvarFp16(float16_t *save_var, int size, float eps);
void backwardAllFp16(const float16_t *in, const float16_t *yt, const float16_t *mean, const float16_t *invar,
                     const float16_t *scale, int size, int ch, float *dxhat_sum, float *dxhathat_sum, float16_t *dbias,
                     float16_t *dscale, float16_t *dx);
void backwardP1Fp16(const float16_t *in, const float16_t *yt, const float16_t *mean, const float16_t *invar,
                    const float16_t *scale, int size, int ch, float *dxhat_sum, float *dxhathat_sum, float16_t *dbias,
                    float16_t *dscale);
void backwardP2Fp16(const float16_t *in, const float16_t *yt, const float16_t *mean, const float16_t *invar,
                    const float16_t *scale, int size, int total_size, int ch, const float *dxhat_sum,
                    const float *dxhathat_sum, float16_t *dx);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NNACL_FP16_GRAD_BATCH_NORM_H_
