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

#include <math.h>
#include "nnacl/op_base.h"
#include "nnacl/fp16_grad/activation_grad.h"
#include "nnacl/errorcode.h"

int Fp16ReluGrad(const float16_t *src0, const float16_t *src1, size_t length, float16_t *dst) {
  int i = 0;
#ifdef ENABLE_NEON
  float16x8_t zero_4 = vdupq_n_f16(0);
  for (; i < length - 4; i += 4) {
    float16x8_t src0_4 = vld1q_f16(src0 + i);
    float16x8_t src1_4 = vld1q_f16(src1 + i);
    uint16x8_t mask_4 = vcgtq_f16(src1_4, zero_4);
    float16x8_t dst_4 = vbslq_f16(mask_4, src0_4, zero_4);
    vst1q_f16(dst + i, dst_4);
  }
#endif
  for (; i < length; i++) {
    dst[i] = (src1[i] > 0.0f) ? src0[i] : 0.0f;
  }
  return NNACL_OK;
}

int Fp16SigmoidGrad(const float16_t *src0, const float16_t *src1, size_t length, float16_t *dst) {
  int i = 0;
#ifdef ENABLE_NEON
  float16x8_t one_4 = vdupq_n_f16(1);
  for (; i < length - 4; i += 4) {
    float16x8_t src0_4 = vld1q_f16(src0 + i);
    float16x8_t src1_4 = vld1q_f16(src1 + i);
    float16x8_t dst_4 = vmulq_f16(src0_4, vmulq_f16(src1_4, vsubq_f16(one_4, src1_4)));
    vst1q_f16(dst + i, dst_4);
  }
#endif
  for (; i < length; i++) {
    dst[i] = src0[i] * (src1[i] * (1.0f - src1[i]));
  }
  return NNACL_OK;
}
