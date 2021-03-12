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
  int i = 0;
#ifdef ENABLE_ARM
  float32x4_t zero_4 = vdupq_n_f32(0.0f);
  for (; i < length - 4; i += 4) {
    float32x4_t src1_4 = vld1q_f32(src1 + i);
    float32x4_t src0_4 = vld1q_f32(src0 + i);
    uint32x4_t mask_4 = vcgtq_f32(src1_4, zero_4);
    float32x4_t dst_4 = vbslq_f32(mask_4, src0_4, zero_4);
    vst1q_f32(dst + i, dst_4);
  }
#endif
  for (; i < length; ++i) {
    dst[i] = (src1[i] > 0.0f) ? src0[i] : 0.0f;
  }
  return NNACL_OK;
}

int Relu6Grad(float *src0, float *src1, size_t length, float *dst) {
  int i = 0;
#ifdef ENABLE_ARM
  float32x4_t zero_4 = vdupq_n_f32(0.0f);
  float32x4_t six_4 = vdupq_n_f32(6.0f);
  for (; i < length - 4; i += 4) {
    float32x4_t src1_4 = vld1q_f32(src1 + i);
    float32x4_t src0_4 = vld1q_f32(src0 + i);
    float32x4_t max_4 = vmaxq_f32(src1_4, zero_4);
    float32x4_t min_max_4 = vminq_f32(max_4, six_4);
    uint32x4_t mask_4 = vceqq_f32(min_max_4, src1_4);
    float32x4_t dst_4 = vbslq_f32(mask_4, src0_4, zero_4);
    vst1q_f32(dst + i, dst_4);
  }
#endif
  for (; i < length; ++i) {
    dst[i] = (src1[i] > 0.0f && src1[i] <= 6.0f) ? src0[i] : 0.0f;
  }
  return NNACL_OK;
}

int LReluGrad(float *src0, float *src1, size_t length, float *dst, float alpha) {
  for (size_t i = 0; i < length; ++i) {
    dst[i] = src1[i] > 0.0f ? src0[i] : alpha * src0[i];
  }
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

int EluGrad(float *src0, float *src1, size_t length, float *dst, float alpha) {
  for (size_t i = 0; i < length; ++i) {
    dst[i] = (src1[i] > 0.0f ? src0[i] : alpha * expm1(src1[i]) * src0[i]);
  }
  return NNACL_OK;
}

int GeluGrad(float *src0, float *src1, size_t length, float *dst) {
  for (size_t i = 0; i < length; ++i) {
    dst[i] = src0[i] * ((0.5 * (1.0 + erf(src1[i] / 1.4142135623730951))) +
                        (src1[i] * exp(-0.5 * src1[i] * src1[i]) / 2.5066282746));
  }
  return NNACL_OK;
}
