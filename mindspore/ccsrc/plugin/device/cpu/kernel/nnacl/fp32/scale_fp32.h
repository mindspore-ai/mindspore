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

#ifndef NNACL_FP32_SCALE_FP32_H_
#define NNACL_FP32_SCALE_FP32_H_

#include "nnacl/op_base.h"
#include "nnacl/kernel/scale.h"
#ifdef __cplusplus
extern "C" {
#endif
void DoScale(const float *in_data, float *out_data, const float *scale, const float *offset, int task_id,
             const ScaleStruct *scale_param);
void DoScaleRelu(const float *in_data, float *out_data, const float *scale, const float *offset, int task_id,
                 const ScaleStruct *scale_param);
void DoScaleRelu6(const float *in_data, float *out_data, const float *scale, const float *offset, int task_id,
                  const ScaleStruct *scale_param);
#ifdef __cplusplus
}
#endif

#endif  // NNACL_FP32_SCALE_FP32_H_
