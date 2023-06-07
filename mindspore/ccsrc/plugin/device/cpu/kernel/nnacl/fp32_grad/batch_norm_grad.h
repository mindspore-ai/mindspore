/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef CCSRC_PLUGIN_DEVICE_CPU_KERNEL_NNACL_FP32_GRAD_BATCH_NORM_H_
#define CCSRC_PLUGIN_DEVICE_CPU_KERNEL_NNACL_FP32_GRAD_BATCH_NORM_H_

#include "nnacl/fp32_grad/batch_norm_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif

void var2Invar(float *save_var, int size, float eps);
void backwardAll(const float *in, const float *yt, const float *mean, const float *invar, const float *scale, int size,
                 int ch, float *dbias, float *dscale, float *dx, bool is_train);
void backwardP1(const float *in, const float *yt, const float *mean, const float *invar, const float *scale, int size,
                int ch, float *dbias, float *dscale);
void backwardP2(const float *in, const float *yt, const float *mean, const float *invar, const float *dscale,
                const float *dbias, const float *scale, int size, int total_size, int ch, float *dx, bool is_train);
#ifdef __cplusplus
}
#endif

#endif  // CCSRC_PLUGIN_DEVICE_CPU_KERNEL_NNACL_FP32_GRAD_BATCH_NORM_H_
