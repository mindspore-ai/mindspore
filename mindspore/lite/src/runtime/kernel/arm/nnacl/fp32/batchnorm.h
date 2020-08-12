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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_FP32_BATCHNORM_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_FP32_BATCHNORM_H_

#include "nnacl/op_base.h"

typedef struct BatchNormParameter {
  OpParameter op_parameter_;
  float epsilon_;
  int unit_;
  int channel_;
} BatchNormParameter;

#ifdef __cplusplus
extern "C" {
#endif

void BatchNorm(float *output_ptr, const float *input_ptr, const float *mean_ptr, const float *variance_ptr, int task_id,
               BatchNormParameter *param);

void FusedBatchNorm(float *output_ptr, const float *input_ptr, const float *scale_ptr, const float *offest_ptr,
                    const float *mean_ptr, const float *variance_ptr, int task_id, BatchNormParameter *param);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_FUSED_BATCHNORM_H_
