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

#ifdef ENABLE_SSE
#include <x86intrin.h>
#include "nnacl/pack.h"
#include "nnacl/int8/conv_int8.h"

void PackNHWCToNCHWFp32(const void *src, void *dst, int batches, int plane, int channel) {
  int hw8 = plane / C8NUM * C8NUM;
  int c8 = channel / C8NUM * C8NUM;
  int batch = plane * channel;
  for (int n = 0; n < batches; n++) {
    const float *src_batch = (const float *)src + n * batch;
    float *dst_batch = (float *)dst + n * batch;
    int hw = 0;
    for (; hw < hw8; hw += C8NUM) {
      int c = 0;
      for (; c < c8; c += C8NUM) {
        const float *src_ptr = src_batch + hw * channel + c;
        float *dst_ptr = dst_batch + c * plane + hw;

        // 11-14
        __m128 v0_ma = _mm_loadu_ps(src_ptr);
        __m128 v1_ma = _mm_loadu_ps(src_ptr + channel);
        __m128 v2_ma = _mm_loadu_ps(src_ptr + 2 * channel);
        __m128 v3_ma = _mm_loadu_ps(src_ptr + 3 * channel);

        __m128 v4_ma = _mm_unpacklo_ps(v0_ma, v1_ma);
        __m128 v5_ma = _mm_unpackhi_ps(v0_ma, v1_ma);
        __m128 v6_ma = _mm_unpacklo_ps(v2_ma, v3_ma);
        __m128 v7_ma = _mm_unpackhi_ps(v2_ma, v3_ma);

        __m128 v8_ma = _mm_movelh_ps(v4_ma, v6_ma);
        __m128 v9_ma = _mm_movehl_ps(v6_ma, v4_ma);
        __m128 v10_ma = _mm_movelh_ps(v5_ma, v7_ma);
        __m128 v11_ma = _mm_movehl_ps(v7_ma, v5_ma);

        _mm_storeu_ps(dst_ptr, v8_ma);
        _mm_storeu_ps(dst_ptr + plane, v9_ma);
        _mm_storeu_ps(dst_ptr + 2 * plane, v10_ma);
        _mm_storeu_ps(dst_ptr + 3 * plane, v11_ma);

        // 15-18
        v0_ma = _mm_loadu_ps(src_ptr + C4NUM);
        v1_ma = _mm_loadu_ps(src_ptr + channel + C4NUM);
        v2_ma = _mm_loadu_ps(src_ptr + 2 * channel + C4NUM);
        v3_ma = _mm_loadu_ps(src_ptr + 3 * channel + C4NUM);

        v4_ma = _mm_unpacklo_ps(v0_ma, v1_ma);
        v5_ma = _mm_unpackhi_ps(v0_ma, v1_ma);
        v6_ma = _mm_unpacklo_ps(v2_ma, v3_ma);
        v7_ma = _mm_unpackhi_ps(v2_ma, v3_ma);

        v8_ma = _mm_movelh_ps(v4_ma, v6_ma);
        v9_ma = _mm_movehl_ps(v6_ma, v4_ma);
        v10_ma = _mm_movelh_ps(v5_ma, v7_ma);
        v11_ma = _mm_movehl_ps(v7_ma, v5_ma);

        _mm_storeu_ps(dst_ptr + C4NUM * plane, v8_ma);
        _mm_storeu_ps(dst_ptr + (C4NUM + 1) * plane, v9_ma);
        _mm_storeu_ps(dst_ptr + (C4NUM + 2) * plane, v10_ma);
        _mm_storeu_ps(dst_ptr + (C4NUM + 3) * plane, v11_ma);

        // 21-24
        v0_ma = _mm_loadu_ps(src_ptr + C4NUM * channel);
        v1_ma = _mm_loadu_ps(src_ptr + (C4NUM + 1) * channel);
        v2_ma = _mm_loadu_ps(src_ptr + (C4NUM + 2) * channel);
        v3_ma = _mm_loadu_ps(src_ptr + (C4NUM + 3) * channel);

        v4_ma = _mm_unpacklo_ps(v0_ma, v1_ma);
        v5_ma = _mm_unpackhi_ps(v0_ma, v1_ma);
        v6_ma = _mm_unpacklo_ps(v2_ma, v3_ma);
        v7_ma = _mm_unpackhi_ps(v2_ma, v3_ma);

        v8_ma = _mm_movelh_ps(v4_ma, v6_ma);
        v9_ma = _mm_movehl_ps(v6_ma, v4_ma);
        v10_ma = _mm_movelh_ps(v5_ma, v7_ma);
        v11_ma = _mm_movehl_ps(v7_ma, v5_ma);

        _mm_storeu_ps(dst_ptr + C4NUM, v8_ma);
        _mm_storeu_ps(dst_ptr + plane + C4NUM, v9_ma);
        _mm_storeu_ps(dst_ptr + 2 * plane + C4NUM, v10_ma);
        _mm_storeu_ps(dst_ptr + 3 * plane + C4NUM, v11_ma);

        // 25-28
        v0_ma = _mm_loadu_ps(src_ptr + C4NUM * channel + C4NUM);
        v1_ma = _mm_loadu_ps(src_ptr + (C4NUM + 1) * channel + C4NUM);
        v2_ma = _mm_loadu_ps(src_ptr + (C4NUM + 2) * channel + C4NUM);
        v3_ma = _mm_loadu_ps(src_ptr + (C4NUM + 3) * channel + C4NUM);

        v4_ma = _mm_unpacklo_ps(v0_ma, v1_ma);
        v5_ma = _mm_unpackhi_ps(v0_ma, v1_ma);
        v6_ma = _mm_unpacklo_ps(v2_ma, v3_ma);
        v7_ma = _mm_unpackhi_ps(v2_ma, v3_ma);

        v8_ma = _mm_movelh_ps(v4_ma, v6_ma);
        v9_ma = _mm_movehl_ps(v6_ma, v4_ma);
        v10_ma = _mm_movelh_ps(v5_ma, v7_ma);
        v11_ma = _mm_movehl_ps(v7_ma, v5_ma);

        _mm_storeu_ps(dst_ptr + C4NUM * plane + C4NUM, v8_ma);
        _mm_storeu_ps(dst_ptr + (C4NUM + 1) * plane + C4NUM, v9_ma);
        _mm_storeu_ps(dst_ptr + (C4NUM + 2) * plane + C4NUM, v10_ma);
        _mm_storeu_ps(dst_ptr + (C4NUM + 3) * plane + C4NUM, v11_ma);
      }

      for (; c < channel; c++) {
        const float *src_ptr = src_batch + hw * channel + c;
        float *dst_ptr = dst_batch + c * plane + hw;
        for (size_t i = 0; i < C8NUM; i++) {
          dst_ptr[i] = src_ptr[i * channel];
        }
      }
    }
    for (; hw < plane; hw++) {
      const float *src_ptr = src_batch + hw * channel;
      float *dst_ptr = dst_batch + hw;
      for (size_t i = 0; i < channel; i++) {
        dst_ptr[i * plane] = src_ptr[i];
      }
    }
  }
  return;
}

#endif
