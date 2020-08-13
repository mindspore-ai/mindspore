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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_FP32_ROI_POOLING_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_FP32_ROI_POOLING_H_

#include "nnacl/op_base.h"

typedef struct ROIPoolingParameter {
  OpParameter op_parameter_;
  int pooledW_;
  int pooledH_;
  float scale_;
} ROIPoolingParameter;

#ifdef __cplusplus
extern "C" {
#endif
int ROIPooling(float *in_ptr, float *out_ptr, float *roi, const int *in_shape, const int *out_shape, int dim, int tid,
               ROIPoolingParameter *param);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_FP32_ROI_POOLING_H_
