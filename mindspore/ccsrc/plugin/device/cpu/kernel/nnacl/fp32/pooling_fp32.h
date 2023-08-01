/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef NNACL_FP32_POOLING_H_
#define NNACL_FP32_POOLING_H_

#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include "nnacl/op_base.h"
#include "nnacl/pooling_parameter.h"
#include "nnacl/int8/quantize.h"
#include "nnacl/kernel/pooling.h"

#ifdef __cplusplus
extern "C" {
#endif
int AvgPooling(const float *input_ptr, float *output_ptr, const PoolingParameter *pooling_param,
               const PoolingComputeParam *pooling_args, int task_id, int thread_num);
int MaxPooling(const float *input_ptr, float *output_ptr, const PoolingParameter *pooling_param,
               const PoolingComputeParam *pooling_args, int task_id, int thread_num);

int AvgPoolingFromNC4HW4ToNHWC(const float *input_ptr, float *output_ptr, const PoolingParameter *pooling_param,
                               const PoolingComputeParam *pooling_args, int task_id, int thread_num);
int MaxPoolingFromNC4HW4ToNHWC(const float *input_ptr, float *output_ptr, const PoolingParameter *pooling_param,
                               const PoolingComputeParam *pooling_args, int task_id, int thread_num);
void MaxPooling3D_NDHWC(const float *input_ptr, float *output_ptr, const Pooling3DParameter *pooling_param,
                        const Pooling3DComputeParam *pooling_args, int start, int end);
void AvgPooling3D_NDHWC(const float *input_ptr, float *output_ptr, const Pooling3DParameter *pooling_param,
                        const Pooling3DComputeParam *pooling_args, int start, int end);
#ifdef __cplusplus
}
#endif

#endif  // NNACL_FP32_POOLING_H_
