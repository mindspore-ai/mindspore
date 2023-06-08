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

#ifndef NNACL_FP16_POOLING_FP16_H_
#define NNACL_FP16_POOLING_FP16_H_

#include <math.h>
#include "nnacl/pooling_parameter.h"
#include "nnacl/intrinsics/ms_simd_instructions_fp16.h"
#include "nnacl/kernel/pooling.h"

#ifdef __cplusplus
extern "C" {
#endif
int AvgPoolingFp16(const float16_t *input_ptr, float16_t *output_ptr, const PoolingParameter *pooling_param,
                   const PoolingComputeParam *pooling_args, int task_id, int thread_num);

void MaxPoolingFp16(const float16_t *input_ptr, float16_t *output_ptr, const PoolingParameter *pooling_param,
                    const PoolingComputeParam *pooling_args, int task_id, int thread_num);
#ifdef __cplusplus
}
#endif
#endif  //  NNACL_FP16_POOLING_FP16_H_
