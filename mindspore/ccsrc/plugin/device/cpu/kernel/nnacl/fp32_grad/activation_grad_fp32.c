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

#include <math.h>
#include "nnacl/op_base.h"
#include "nnacl/fp32/arithmetic_fp32.h"
#include "nnacl/fp32/exp_fp32.h"
#include "nnacl/fp32_grad/activation_grad_fp32.h"
#include "nnacl/errorcode.h"
#include "nnacl/intrinsics/ms_simd_instructions.h"
#include "nnacl/activation_grad_simd.h"

int ReluGrad(const float *src0, const float *src1, int length, float *dst) {
  int i = 0;
#ifdef ENABLE_ARM
  float32x4_t zero_4 = vdupq_n_f32(0.0f);
  for (; i < length - C4NUM; i += C4NUM) {
    float32x4_t src1_4 = vld1q_f32(src1 + i);
    float32x4_t src0_4 = vld1q_f32(src0 + i);
    uint32x4_t mask_4 = vcleq_f32(src1_4, zero_4);
    float32x4_t dst_4 = vbslq_f32(mask_4, zero_4, src0_4);
    vst1q_f32(dst + i, dst_4);
  }
#endif
  for (; i < length; ++i) {
    dst[i] = (src1[i] > 0.0f) ? src0[i] : 0.0f;
  }
  return NNACL_OK;
}

int Relu6Grad(const float *src0, const float *src1, size_t length, float *dst) {
  size_t i = 0;
#ifdef ENABLE_ARM
  float32x4_t zero_4 = vdupq_n_f32(0.0f);
  float32x4_t six_4 = vdupq_n_f32(6.0f);
  for (; i < length - C4NUM; i += C4NUM) {
    float32x4_t src1_4 = vld1q_f32(src1 + i);
    float32x4_t src0_4 = vld1q_f32(src0 + i);
    uint32x4_t gt_4 = vcgtq_f32(src1_4, zero_4);
    uint32x4_t le_4 = vcleq_f32(src1_4, six_4);
    uint32x4_t mask_4 = vandq_u32(gt_4, le_4);
    float32x4_t dst_4 = vbslq_f32(mask_4, src0_4, zero_4);
    vst1q_f32(dst + i, dst_4);
  }
#endif
  for (; i < length; ++i) {
    dst[i] = (src1[i] > 0.0f && src1[i] <= 6.0f) ? src0[i] : 0.0f;
  }
  return NNACL_OK;
}

int LReluGrad(const float *src0, const float *src1, size_t length, float *dst, float alpha) {
  for (size_t i = 0; i < length; ++i) {
    dst[i] = src1[i] > 0.0f ? src0[i] : alpha * src0[i];
  }
  return NNACL_OK;
}

int SigmoidGrad(const float *src0, const float *src1, size_t length, float *dst) {
  for (size_t i = 0; i < length; ++i) {
    dst[i] = src0[i] * (src1[i] * (1.0f - src1[i]));
  }
  return NNACL_OK;
}

int TanhGrad(const float *src0, const float *src1, size_t length, float *dst) {
  for (size_t i = 0; i < length; ++i) {
    dst[i] = (1.0f - (src1[i] * src1[i])) * src0[i];
  }
  return NNACL_OK;
}

int HSwishGrad(const float *src0, const float *src1, size_t length, float *dst) {
  for (size_t i = 0; i < length; ++i) {
    float tmp = (src1[i] > 3.0f ? 1.0f : (src1[i] < -3.0f ? 0.0f : (2.0f * src1[i] + 3.0f) / 6.0f));
    dst[i] = tmp * src0[i];
  }
  return NNACL_OK;
}

int HSigmoidGrad(const float *src0, const float *src1, size_t length, float *dst) {
  for (size_t i = 0; i < length; ++i) {
    float tmp = (src1[i] > 3.0f ? 0.0f : (src1[i] < -3.0f ? 0.0f : 1.0f / 6.0f));
    dst[i] = tmp * src0[i];
  }
  return NNACL_OK;
}

int EluGrad(const float *src0, const float *src1, size_t length, float *dst, float alpha) {
  for (size_t i = 0; i < length; ++i) {
    dst[i] = (src1[i] > 0.0f ? src0[i] : alpha * expm1(src1[i]) * src0[i]);
  }
  return NNACL_OK;
}

int GeluGrad(const float *src0, const float *src1, size_t length, float *dst) {
  for (size_t i = 0; i < length; ++i) {
    dst[i] = src0[i] * ((0.5 * (1.0 + erf(src1[i] / 1.4142135623730951))) +
                        (src1[i] * exp(-0.5 * src1[i] * src1[i]) / 2.5066282746));
  }
  return NNACL_OK;
}

int SoftplusGrad(const float *src0, const float *src1, int length, float *dst) {
  int i = 0;
#if defined(ENABLE_AVX)
  for (; i <= length - C8NUM; i += C8NUM) {
    simd_exp256(MS_SUB256_F32(MS_MOV256_F32(0.0f), (MS_LD256_F32(src1 + i))), dst + i);
    MS_ST256_F32(dst + i,
                 MS_DIV256_F32(MS_LD256_F32(src0 + i), MS_ADD256_F32(MS_MOV256_F32(1.0f), MS_LD256_F32(dst + i))));
  }
#endif

#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  for (; i <= length - C4NUM; i += C4NUM) {
    simd_exp128(MS_SUBQ_F32(MS_MOVQ_F32(0.0f), MS_LDQ_F32(src1 + i)), dst + i);
    MS_STQ_F32(dst + i, MS_DIVQ_F32(MS_LDQ_F32(src0 + i), MS_ADDQ_F32(MS_MOVQ_F32(1.0f), MS_LDQ_F32(dst + i))));
  }
#endif

  for (; i < length; ++i) {
    simd_exp32(-src1[i], dst + i);
    dst[i] = src0[i] / (1.0f + dst[i]);
  }
  return NNACL_OK;
}

int HardShrinkGrad(const float *src0, const float *src1, int length, float *dst, float lambd) {
  int i = 0;
  const float neg_lambd = -1 * lambd;
  SIMD_RUN_NO_SCALAR(ShrinkGrad, i, src0, src1, length, dst, lambd);

  for (; i < length; ++i) {
    dst[i] = (src1[i] >= neg_lambd && src1[i] <= lambd) ? 0 : src0[i];
  }
  return NNACL_OK;
}

int SoftShrinkGrad(const float *src0, const float *src1, int length, float *dst, float lambd) {
  int i = 0;
  const float neg_lambd = -1 * lambd;
  SIMD_RUN_NO_SCALAR(ShrinkGrad, i, src0, src1, length, dst, lambd);

  for (; i < length; ++i) {
    dst[i] = (src1[i] >= neg_lambd && src1[i] <= lambd) ? 0 : src0[i];
  }
  return NNACL_OK;
}
