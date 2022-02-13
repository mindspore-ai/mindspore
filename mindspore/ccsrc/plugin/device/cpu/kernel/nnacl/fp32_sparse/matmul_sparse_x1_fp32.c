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

#include "nnacl/fp32_sparse/matmul_sparse_x1_fp32.h"
#ifdef ENABLE_ARM64
#include <arm_neon.h>
#endif

void MatMulSparse8x8(const float *a, const float *b, const uint32_t *nnz, const size_t *dmap, float *c,
                     const float *bias, ActType act_type, int out_stride) {
#ifndef ENABLE_ARM64
  return;
#else
  // mul-acc
  for (int oc = 0; oc < 8; oc++) {
    uint32_t cur_nnz = nnz[oc];
    // init 8x1 C with bias
    float32x4_t vacc1 = vld1q_dup_f32(bias + oc);
    float32x4_t vacc2 = vacc1;
    for (uint32_t nz = 0; nz < cur_nnz; nz++) {
      // load w
      float32x4_t vw = vld1q_dup_f32(b++);
      // load 8 inputs
      const float *input = a + (*(dmap++) / sizeof(float));
      float32x4_t vi1 = vld1q_f32(input);
      float32x4_t vi2 = vld1q_f32(input + 4);
      vacc1 = vfmaq_f32(vacc1, vi1, vw);
      vacc2 = vfmaq_f32(vacc2, vi2, vw);
    }
    // save output
    *(c + oc) = vacc1[0];
    *(c + 1 * out_stride + oc) = vacc1[1];
    *(c + 2 * out_stride + oc) = vacc1[2];
    *(c + 3 * out_stride + oc) = vacc1[3];
    *(c + 4 * out_stride + oc) = vacc2[0];
    *(c + 5 * out_stride + oc) = vacc2[1];
    *(c + 6 * out_stride + oc) = vacc2[2];
    *(c + 7 * out_stride + oc) = vacc2[3];
  }
#endif
}
