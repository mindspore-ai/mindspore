/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_NNACL_ADAM_FP32_H
#define MINDSPORE_NNACL_ADAM_FP32_H
#include <math.h>
#include "nnacl/op_base.h"
#include "nnacl/errorcode.h"

#ifdef __cplusplus
extern "C" {
#endif
int AdamFp32(float *var, float *m, float *v, float lr, float beta1, float beta2, float epsilon, const float *gradient,
             size_t start, size_t end, bool use_nesterov);
int AdamDeltaFp32(float *delta, float *m, float *v, float lr, float beta1, float beta2, float epsilon,
                  const float *gradient, size_t start, size_t end, bool use_nesterov);
int AdamWeightDecayFp32(float *var, float *m, float *v, float lr, float beta1, float beta2, float epsilon, float decay,
                        const float *gradient, size_t start, size_t end);
size_t FusedCastAdamFp32Fp16(float *var, const int16_t *gradient16, float *m, float *v, float lr, float beta1,
                             float beta2, float epsilon, float decay, float global_norm_reciprocal, size_t start,
                             size_t end);
size_t FusedCastAdamFp32Fp32(float *var, const float *gradient32, float *m, float *v, float lr, float beta1,
                             float beta2, float epsilon, float decay, float global_norm_reciprocal, size_t start,
                             size_t end);
size_t FusedCastAdamFp16Fp16(int16_t *var16, const int16_t *gradient16, float *m, float *v, float lr, float beta1,
                             float beta2, float epsilon, float decay, float global_norm_reciprocal, size_t start,
                             size_t end);
size_t FusedCastAdamFp16Fp32(int16_t *var16, const float *gradient32, float *m, float *v, float lr, float beta1,
                             float beta2, float epsilon, float decay, float global_norm_reciprocal, size_t start,
                             size_t end);

int DoAdam(float *m, float *v, const float *gradient, float *weight, float beta1, float beta2, float *beta1_power,
           float *beta2_power, float eps, float learning_rate, bool nesterov, int start, int end);
#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_NNACL_ADAM_FP32_H
