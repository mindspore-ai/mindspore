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
#ifndef MINDSPORE_NNACL_FP32_INSTANCE_NORM_H_
#define MINDSPORE_NNACL_FP32_INSTANCE_NORM_H_

#include "nnacl/op_base.h"
#include "nnacl/instance_norm_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MS_ADDQ_F32_VEC(in1, in2, in3, in4, v1, v2, v3, v4) \
  in1 = MS_ADDQ_F32(in1, v1);                               \
  in2 = MS_ADDQ_F32(in2, v2);                               \
  in3 = MS_ADDQ_F32(in3, v3);                               \
  in4 = MS_ADDQ_F32(in4, v4);

#define MS_DIVQ_F32_VEC(in1, in2, in3, in4, v) \
  in1 = MS_DIVQ_F32(in1, v);                   \
  in2 = MS_DIVQ_F32(in2, v);                   \
  in3 = MS_DIVQ_F32(in3, v);                   \
  in4 = MS_DIVQ_F32(in4, v);

int InstanceNorm(const float *src_data, float *dst_data, const float *gamma_data, const float *beta_data,
                 const InstanceNormParameter *param, size_t task_id);
int InstanceNormNC4HW4(const float *src_data, float *dst_data, const float *gamma_data, const float *beta_data,
                       const InstanceNormParameter *param, size_t task_id);
#ifdef ENABLE_AVX
int InstanceNormNC8HW8(const float *src_data, float *dst_data, const float *gamma_data, const float *beta_data,
                       const InstanceNormParameter *param, size_t task_id);
#endif
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_INSTANCE_NORM_H_
