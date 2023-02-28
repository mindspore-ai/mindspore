/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_NNACL_FP32_ACTIVATION_H_
#define MINDSPORE_NNACL_FP32_ACTIVATION_H_

#include <math.h>
#include "nnacl/op_base.h"
#include "nnacl/int8/fixed_point.h"
#include "nnacl/activation_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif
int Fp32Relu(const float *src, int length, float *dst);
int Int32Relu(const int32_t *src, int length, int32_t *dst);
int Fp32Relu6(const float *src, int length, float *dst);
int Fp32Clip(const float *src, int length, float *dst, float min, float max);
int Int32Clip(const int *src, int length, int *dst, int min, int max);
int LRelu(const float *src, int length, float *dst, float alpha);
int Sigmoid(const float *src, int length, float *dst);
int Tanh(const float *src, int length, float *dst);
int HSigmoid(const float *src, int length, float *dst);
int Swish(const float *src, int length, float *dst);
int HSwish(const float *src, int length, float *dst);
int HardTanh(const float *src, int length, float *dst, float min_val, float max_val);
int Gelu(const float *src, int length, float *dst, bool approximate);
int Softplus(const float *src, int length, float *dst);
int Elu(const float *src, int length, float *dst, float alpha);
void Celu(const float *src, int length, float *dst, float alpha);
float TanhOpt(float src);
int HShrink(const float *src, int length, float *dst, float lambd);
int SoftShrink(const float *src, int length, float *dst, float lambd);
int SoftsignFp32Opt(const float *src, int length, float *dst);
#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_NNACL_FP32_ACTIVATION_H_
