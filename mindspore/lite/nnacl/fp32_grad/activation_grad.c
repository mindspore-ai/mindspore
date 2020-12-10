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

#include <math.h>
#include "nnacl/op_base.h"
#include "nnacl/fp32/arithmetic_fp32.h"
#include "nnacl/fp32_grad/activation_grad.h"
#include "nnacl/errorcode.h"

inline int ReluGrad(float *src0, float *src1, size_t length, float *dst) {
  for (size_t i = 0; i < length; ++i) {
    if (src1[i] > 0) {
      dst[i] = src0[i];
    } else {
      dst[i] = 0;
    }
  }
  return NNACL_OK;
}

int Relu6Grad(float *src0, float *src1, size_t length, float *dst) {
  for (size_t i = 0; i < length; ++i) {
    if (src1[i] > 0.0f && src1[i] <= 6.0f) {
      dst[i] = src0[i];
    } else {
      dst[i] = 0.0f;
    }
  }
  return NNACL_OK;
}

int LReluGrad(float *src0, float *src1, size_t length, float *dst, float alpha) {
  for (size_t i = 0; i < length; ++i) {
    dst[i] = src1[i] > 0.0f ? 1.0f : alpha;
  }
  ElementMul(src0, dst, dst, length);
  return NNACL_OK;
}

int SigmoidGrad(float *src0, float *src1, size_t length, float *dst) {
  for (size_t i = 0; i < length; ++i) {
    dst[i] = src0[i] * (src1[i] * (1.0f - src1[i]));
  }
  return NNACL_OK;
}

int TanhGrad(float *src0, float *src1, size_t length, float *dst) {
  for (size_t i = 0; i < length; ++i) {
    dst[i] = (1.0f - (src1[i] * src1[i])) * src0[i];
  }
  return NNACL_OK;
}

int HSwishGrad(float *src0, float *src1, size_t length, float *dst) {
  for (size_t i = 0; i < length; ++i) {
    float tmp = (src1[i] > 3.0f ? 1.0f : (src1[i] < -3.0f ? 0.0f : (2.0f * src1[i] + 3.0f) / 6.0f));
    dst[i] = tmp * src0[i];
  }
  return NNACL_OK;
}

int HSigmoidGrad(float *src0, float *src1, size_t length, float *dst) {
  for (size_t i = 0; i < length; ++i) {
    float tmp = (src1[i] > 3.0f ? 0.0f : (src1[i] < -3.0f ? 0.0f : 1.0f / 6.0f));
    dst[i] = tmp * src0[i];
  }
  return NNACL_OK;
}
