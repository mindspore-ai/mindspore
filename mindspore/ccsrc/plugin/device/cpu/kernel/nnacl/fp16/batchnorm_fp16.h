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
#ifndef NNACL_FP16_BATCHNORM_FP16_H_
#define NNACL_FP16_BATCHNORM_FP16_H_

#include "nnacl/kernel/batch_norm.h"

#ifdef __cplusplus
extern "C" {
#endif

void BatchNormFp16(const float16_t *input, const float16_t *mean, const float16_t *variance,
                   const BatchNormStruct *param, int task_id, int thread_num, float16_t *output);
void FusedBatchNormFp16(const float16_t *input, const float16_t *scale, const float16_t *offset, const float16_t *mean,
                        const float16_t *variance, const BatchNormStruct *param, int task_id, int thread_num,
                        float16_t *output);
void FusedBatchNormFp16MeanVar(const float16_t *input, float16_t *run_mean, float16_t *run_var,
                               const BatchNormStruct *param, float16_t *save_mean, float16_t *save_var);
#ifdef __cplusplus
}
#endif

#endif  // NNACL_FP16_BATCHNORM_FP16_H_
