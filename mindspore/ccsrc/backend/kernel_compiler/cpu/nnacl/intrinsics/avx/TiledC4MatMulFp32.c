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
#ifdef ENABLE_AVX
#ifdef _MSC_VER
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif
#include "nnacl/fp32/common_func_fp32.h"

static inline __m256 padd(__m256 v0, __m256 v1, __m256 v2, __m256 v3) {
  __m256 h0 = _mm256_hadd_ps(v0, v1);
  __m256 h1 = _mm256_hadd_ps(v2, v3);
  __m256 res = _mm256_hadd_ps(h0, h1);
  return res;
}

void TiledC4MatmulFp32(float *dst, const float *src, const float *weight, size_t dst_step, size_t ic4, size_t oc4) {
  for (int oc = 0; oc < oc4; oc++) {
    float *dst_oc = dst + oc * dst_step;
    const float *weight_oc = weight + oc * ic4 * 16;
    for (int cur = 0; cur < 2; cur++) {
      float *cur_dst = dst_oc + cur * 16;
      const float *cur_src = src + cur * 16;

      __m256 in0 = _mm256_loadu_ps(cur_src);
      __m256 in1 = _mm256_loadu_ps(cur_src + 8);

      __m256 w0 = _mm256_broadcast_ps((const __m128 *)(weight_oc));
      __m256 w1 = _mm256_broadcast_ps((const __m128 *)(weight_oc + 4));
      __m256 w2 = _mm256_broadcast_ps((const __m128 *)(weight_oc + 8));
      __m256 w3 = _mm256_broadcast_ps((const __m128 *)(weight_oc + 12));

      __m256 d00 = _mm256_mul_ps(in0, w0);
      __m256 d01 = _mm256_mul_ps(in0, w1);
      __m256 d02 = _mm256_mul_ps(in0, w2);
      __m256 d03 = _mm256_mul_ps(in0, w3);
      __m256 d10 = _mm256_mul_ps(in1, w0);
      __m256 d11 = _mm256_mul_ps(in1, w1);
      __m256 d12 = _mm256_mul_ps(in1, w2);
      __m256 d13 = _mm256_mul_ps(in1, w3);
      for (int ic = 1; ic < ic4; ic++) {
        const float *src_ic = cur_src + ic * 32;
        in0 = _mm256_loadu_ps(src_ic);
        in1 = _mm256_loadu_ps(src_ic + 8);

        const float *weight_ic = weight_oc + ic * 16;
        w0 = _mm256_broadcast_ps((const __m128 *)(weight_ic));
        w1 = _mm256_broadcast_ps((const __m128 *)(weight_ic + 4));
        w2 = _mm256_broadcast_ps((const __m128 *)(weight_ic + 8));
        w3 = _mm256_broadcast_ps((const __m128 *)(weight_ic + 12));

        d00 = _mm256_fmadd_ps(in0, w0, d00);
        d01 = _mm256_fmadd_ps(in0, w1, d01);
        d02 = _mm256_fmadd_ps(in0, w2, d02);
        d03 = _mm256_fmadd_ps(in0, w3, d03);
        d10 = _mm256_fmadd_ps(in1, w0, d10);
        d11 = _mm256_fmadd_ps(in1, w1, d11);
        d12 = _mm256_fmadd_ps(in1, w2, d12);
        d13 = _mm256_fmadd_ps(in1, w3, d13);
      }

      _mm256_storeu_ps(cur_dst, padd(d00, d01, d02, d03));
      _mm256_storeu_ps(cur_dst + 8, padd(d10, d11, d12, d13));
    }
  }
}

#endif
