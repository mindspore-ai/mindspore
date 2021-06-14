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

#include "nnacl/fp16_grad/activation_grad.h"
#include <math.h>
#include <float.h>
#ifdef ENABLE_NEON
#include <arm_neon.h>
#include "nnacl/fp32/exp_fp32.h"
#include "nnacl/intrinsics/ms_simd_instructions_fp16.h"
#endif
#include "nnacl/op_base.h"
#include "nnacl/errorcode.h"

int ReluFp16Grad(const float16_t *src0, const float16_t *src1, int length, float16_t *dst) {
  int i = 0;
#ifdef ENABLE_NEON
  float16x8_t zero_v = vdupq_n_f16(0);
  for (; i <= length - C8NUM; i += C8NUM) {
    float16x8_t src0_v = vld1q_f16(src0 + i);
    float16x8_t src1_v = vld1q_f16(src1 + i);
    uint16x8_t mask_v = vcgtq_f16(src1_v, zero_v);
    float16x8_t dst_v = vbslq_f16(mask_v, src0_v, zero_v);
    vst1q_f16(dst + i, dst_v);
  }
#endif
  for (; i < length; i++) {
    dst[i] = (src1[i] > 0.0f) ? src0[i] : 0.0f;
  }
  return NNACL_OK;
}

int Relu6Fp16Grad(const float16_t *src0, const float16_t *src1, int length, float16_t *dst) {
  int i = 0;
#ifdef ENABLE_NEON
  float16x8_t zero_8 = vdupq_n_f16(0);
  float16x8_t six_8 = vdupq_n_f16(6);
  for (; i <= length - 8; i += 8) {
    float16x8_t src1_8 = vld1q_f16(src1 + i);
    float16x8_t src0_8 = vld1q_f16(src0 + i);
    float16x8_t max_8 = vmaxq_f16(src1_8, zero_8);
    float16x8_t min_max_8 = vminq_f16(max_8, six_8);
    uint16x8_t mask_8 = vceqq_f16(min_max_8, src1_8);
    float16x8_t dst_8 = vbslq_f16(mask_8, src0_8, zero_8);
    vst1q_f16(dst + i, dst_8);
  }
#endif
  for (; i < length; ++i) {
    dst[i] = (src1[i] > 0.0f && src1[i] <= 6.0f) ? src0[i] : 0.0f;
  }
  return NNACL_OK;
}

int LReluFp16Grad(const float16_t *src0, const float16_t *src1, int length, float16_t *dst, float16_t alpha) {
  int i = 0;
#ifdef ENABLE_NEON
  const MS_FLOAT16X8 one_8 = vdupq_n_f16(1);
  for (; i <= length - C8NUM; i += C8NUM) {
    MS_FLOAT16X8 src0_8 = MS_LDQ_F16(src0 + i);
    MS_FLOAT16X8 src1_8 = MS_LDQ_F16(src1 + i);
    MS_STQ_F16(dst + i, vmulq_f16(src0_8, vmulq_f16(src1_8, (one_8 - src1_8))));
  }
#endif
  for (; i < length; ++i) {
    dst[i] = src0[i] * (src1[i] * (1.0f - src1[i]));
  }
  return NNACL_OK;
}

int SigmoidFp16Grad(const float16_t *src0, const float16_t *src1, int length, float16_t *dst) {
  int i = 0;
#ifdef ENABLE_NEON
  float16x8_t one_8 = vdupq_n_f16(1);
  for (; i < length - 8; i += 8) {
    float16x8_t src0_8 = vld1q_f16(src0 + i);
    float16x8_t src1_8 = vld1q_f16(src1 + i);
    float16x8_t dst_8 = vmulq_f16(src0_8, vmulq_f16(src1_8, vsubq_f16(one_8, src1_8)));
    vst1q_f16(dst + i, dst_8);
  }
#endif
  for (; i < length; i++) {
    dst[i] = src0[i] * (src1[i] * (1.0f - src1[i]));
  }
  return NNACL_OK;
}

int TanhFp16Grad(const float16_t *src0, const float16_t *src1, int length, float16_t *dst) {
  for (int i = 0; i < length; ++i) {
    dst[i] = (float16_t)((1.0f - ((float)src1[i] * (float)src1[i])) * (float)src0[i]);
  }
  return NNACL_OK;
}

int HSwishFp16Grad(const float16_t *src0, const float16_t *src1, int length, float16_t *dst) {
  for (int i = 0; i < length; ++i) {
    float16_t tmp = (src1[i] > 3.0f ? 1.0f : (src1[i] < -3.0f ? 0.0f : (2.0f * src1[i] + 3.0f) / 6.0f));
    dst[i] = tmp * src0[i];
  }
  return NNACL_OK;
}

int HSigmoidFp16Grad(const float16_t *src0, const float16_t *src1, int length, float16_t *dst) {
  for (int i = 0; i < length; ++i) {
    float16_t tmp = (src1[i] > 3.0f ? 0.0f : (src1[i] < -3.0f ? 0.0f : 1.0f / 6.0f));
    dst[i] = tmp * src0[i];
  }
  return NNACL_OK;
}
int EluFp16Grad(const float16_t *src0, const float16_t *src1, int length, float16_t *dst, float16_t alpha) {
  int i = 0;
#ifdef ENABLE_NEON
  float16x4_t zero_4 = vdup_n_f16(0);
  float16x4_t one_4 = vdup_n_f16(1);
  float16x4_t alpha_4 = vdup_n_f16(alpha);
  for (; i <= length - 4; i += 4) {
    float16x4_t src0_4 = vld1_f16(src0 + i);
    float16x4_t src1_4 = vld1_f16(src1 + i);
    uint16x4_t mask_4 = vcgt_f16(src1_4, zero_4);
    float32x4_t tmp;
    simd_exp(vcvt_f32_f16(src1_4), (float *)&tmp);
    uint16x4_t expm1_4 = vsub_f16(vcvt_f16_f32(tmp), one_4);
    float16x4_t dst_4 = vbsl_f16(mask_4, src0_4, vmul_f16(alpha_4, vmul_f16(expm1_4, src0_4)));
    vst1_f16(dst + i, dst_4);
  }
#endif
  for (; i < length; ++i) {
    dst[i] = (src1[i] > 0.0f ? src0[i] : alpha * expm1(src1[i]) * src0[i]);
  }
  return NNACL_OK;
}

int GeluFp16Grad(const float16_t *src0, const float16_t *src1, int length, float16_t *dst) {
  for (int i = 0; i < length; ++i) {
    dst[i] = src0[i] * ((0.5 * (1.0 + erf(src1[i] / 1.4142135623730951))) +
                        (src1[i] * exp(-0.5 * src1[i] * src1[i]) / 2.5066282746));
  }
  return NNACL_OK;
}
