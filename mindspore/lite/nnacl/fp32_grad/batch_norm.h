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

#ifndef MINDSPORE_LITE_NNACL_FP32_GRAD_BATCH_NORM_H_
#define MINDSPORE_LITE_NNACL_FP32_GRAD_BATCH_NORM_H_

#include "nnacl/op_base.h"

typedef struct BNGradParameter {
  OpParameter op_parameter_;
  float epsilon_;
} BNGradParameter;

#ifdef __cplusplus
extern "C" {
#endif

void var2Invar(float *save_var, int size, float eps);
void backwardAll(const float *in, const float *yt, const float *mean, const float *invar, const float *scale, int size,
                 int ch, float *dxhat_sum, float *dxhathat_sum, float *dbias, float *dscale, float *dx);
void backwardP1(const float *in, const float *yt, const float *mean, const float *invar, const float *scale, int size,
                int ch, float *dxhat_sum, float *dxhathat_sum, float *dbias, float *dscale);
void backwardP2(const float *in, const float *yt, const float *mean, const float *invar, const float *scale, int size,
                int total_size, int ch, const float *dxhat_sum, const float *dxhathat_sum, float *dx);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_FP32_GRAD_BATCH_NORM_H_
