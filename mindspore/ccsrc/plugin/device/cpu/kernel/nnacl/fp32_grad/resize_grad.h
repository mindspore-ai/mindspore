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

#ifndef NNACL_FP32_GRAD_RESIZE_GRAD_H_
#define NNACL_FP32_GRAD_RESIZE_GRAD_H_

#include "nnacl/fp32_grad/resize_grad_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif

int ResizeNearestNeighborGrad(const float *in_addr, float *out_addr, int batch_size, int channel, int format,
                              const ResizeGradParameter *param);
int ResizeBiLinearGrad(const float *in_addr, float *out_addr, int batch_size, int channel, int format,
                       const ResizeGradParameter *param);
#ifdef __cplusplus
}
#endif
#endif  //  NNACL_FP32_GRAD_RESIZE_GRAD_H_
