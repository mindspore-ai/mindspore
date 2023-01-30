/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_NNACL_FP32_GRAD_ACTIVATION_GRAD_FP32_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_NNACL_FP32_GRAD_ACTIVATION_GRAD_FP32_H_

#include <math.h>
#include "nnacl/op_base.h"
#include "nnacl/fp32/arithmetic_fp32.h"
#include "nnacl/errorcode.h"

typedef struct ActivationGradParameter {
  OpParameter op_parameter;
  int type_;
  float alpha_;
} ActivationGradParameter;
#ifdef __cplusplus
extern "C" {
#endif

int ReluGrad(const float *src0, const float *src1, int length, float *dst);
int Relu6Grad(const float *src0, const float *src1, size_t length, float *dst);
int LReluGrad(const float *src0, const float *src1, size_t length, float *dst, float alpha);
int SigmoidGrad(const float *src0, const float *src1, size_t length, float *dst);
int TanhGrad(const float *src0, const float *src1, size_t length, float *dst);
int HSwishGrad(const float *src0, const float *src1, size_t length, float *dst);
int HSigmoidGrad(const float *src0, const float *src1, size_t length, float *dst);
int EluGrad(const float *src0, const float *src1, size_t length, float *dst, float alpha);
int GeluGrad(const float *src0, const float *src1, size_t length, float *dst);
int SoftplusGrad(const float *src, const float *src1, int length, float *dst);
int HardShrinkGrad(const float *src0, const float *src1, int length, float *dst, float lambd);
int SoftShrinkGrad(const float *src0, const float *src1, int length, float *dst, float lambd);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_NNACL_FP32_GRAD_ACTIVATION_GRAD_FP32_H_
