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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_FP32_GRAD_ACTIVATION_GRAD_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_FP32_GRAD_ACTIVATION_GRAD_H_

#include <math.h>
#include "src/runtime/kernel/arm/opclib/op_base.h"
#include "src/runtime/kernel/arm/opclib/fp32/arithmetic.h"
#include "src/runtime/kernel/arm/opclib/errorcode.h"

struct ActivationGradParameter {
  OpParameter op_parameter{};
  int type_;
  float alpha_{0.01};
};

inline int ReluGrad(float *src0, float *src1, int length, float *dst) {
  for (int i = 0; i < length; ++i) {
    dst[i] = src1[i] > 0 ? 1.0f : 0.0f;
  }
  ElementMul(src0, dst, dst, length);
  return OPCLIB_OK;
}

inline int Relu6Grad(float *src0, float *src1, int length, float *dst) {
  for (int i = 0; i < length; ++i) {
    if (src1[i] < 0) {
      dst[i] = 0;
    } else {
      dst[i] = src1[i] > 6.0f ? 0.0f : 1.0f;
    }
  }
  ElementMul(src0, dst, dst, length);
  return OPCLIB_OK;
}

inline int LReluGrad(float *src0, float *src1, int length, float *dst, float alpha) {
  for (int i = 0; i < length; ++i) {
    dst[i] = src1[i] > 0.0f ? 1.0f : alpha;
  }
  ElementMul(src0, dst, dst, length);
  return OPCLIB_OK;
}

inline int SigmoidGrad(float *src0, float *src1, int length, float *dst) {
  for (int i = 0; i < length; ++i) {
    dst[i] = src0[i] * (src1[i] * (1.0f - src1[i]));
  }
  return OPCLIB_OK;
}

inline int TanhGrad(float *src0, float *src1, int length, float *dst) {
  for (int i = 0; i < length; ++i) {
    dst[i] = (1.0f - (src1[i] * src1[i])) * src0[i];
  }
  return OPCLIB_OK;
}

inline int HSwishGrad(float *src0, float *src1, int length, float *dst) {
  for (int i = 0; i < length; ++i) {
    float tmp = (src1[i] > 3.0f ? 1.0f : (src1[i] < -3.0f ? 0.0f : (2.0f * src1[i] + 3.0f) / 6.0f));
    dst[i] = tmp * src0[i];
  }
  return OPCLIB_OK;
}

inline int HSigmoidGrad(float *src0, float *src1, int length, float *dst) {
  for (int i = 0; i < length; ++i) {
    float tmp = (src1[i] > 3.0f ? 1.0f : (src1[i] < -3.0f ? 0.0f : 1.0f / 6.0f));
    dst[i] = tmp * src0[i];
  }
  return OPCLIB_OK;
}

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_NNACL_FP32_GRAD_ACTIVATION_GRAD_H_
