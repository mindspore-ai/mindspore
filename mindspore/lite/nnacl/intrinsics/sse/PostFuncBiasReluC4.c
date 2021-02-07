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
#include "nnacl/fp32/common_func_fp32.h"
#include "nnacl/intrinsics/sse/sse_common.h"

void PostFuncBiasReluC4(float *dst, const float *src, const float *bias, size_t oc4div, size_t oc4mod,
                        size_t plane_size, size_t plane_stride, size_t relu_type) {
  size_t stride = oc4div + oc4mod;
  plane_stride /= sizeof(float);
  for (size_t loop_c4 = 0; loop_c4 < oc4div; loop_c4 += C4NUM) {
    size_t plane_size_tmp = plane_size;
    float *dst_c4 = dst + loop_c4;
    __m128 bias1 = _mm_setzero_ps();
    if (bias != NULL) {
      bias1 = _mm_loadu_ps(bias);
      bias += 4;
    }
    for (; plane_size_tmp >= C4NUM; plane_size_tmp -= C4NUM) {
      __m128 src1 = _mm_loadu_ps(src);
      __m128 src2 = _mm_loadu_ps(src + 4);
      __m128 src3 = _mm_loadu_ps(src + 8);
      __m128 src4 = _mm_loadu_ps(src + 12);
      src += 16;
      src1 = _mm_add_ps(src1, bias1);
      src2 = _mm_add_ps(src2, bias1);
      src3 = _mm_add_ps(src3, bias1);
      src4 = _mm_add_ps(src4, bias1);

      ActBlock4(&src1, &src2, &src3, &src4, relu_type == 1, relu_type == 3);

      _mm_storeu_ps(dst_c4, src1);
      dst_c4 += stride;
      _mm_storeu_ps(dst_c4, src2);
      dst_c4 += stride;
      _mm_storeu_ps(dst_c4, src3);
      dst_c4 += stride;
      _mm_storeu_ps(dst_c4, src4);
      dst_c4 += stride;
    }
    for (; plane_size_tmp > 0; plane_size_tmp -= 1) {
      __m128 src1 = _mm_loadu_ps(src);
      src1 = _mm_add_ps(src1, bias1);

      ActBlock1(&src1, relu_type == 1, relu_type == 3);

      _mm_storeu_ps(dst_c4, src1);
      dst_c4 += stride;
      src += 4;
    }
    src += plane_stride;
  }
  if (oc4mod == 0) {
    return;
  }
  __m128 bias1 = _mm_setzero_ps();
  if (bias != NULL) {
    bias1 = _mm_loadu_ps(bias);
    bias += 4;
  }
  float *dst_c1 = dst + oc4div;
  for (size_t plane_size_tmp = plane_size; plane_size_tmp > 0; plane_size_tmp -= 1) {
    __m128 src1 = _mm_loadu_ps(src);
    src += 4;
    src1 = _mm_add_ps(src1, bias1);

    ActBlock1(&src1, relu_type == 1, relu_type == 3);

    switch (oc4mod) {
      case 1:
        _mm_store_ss(dst_c1, src1);
        dst_c1 += stride;
        break;
      case 2:
        _mm_storel_pi((__m64 *)(dst_c1), src1);
        dst_c1 += stride;
        break;
      case 3:
        _mm_storel_pi((__m64 *)(dst_c1), src1);
        src1 = _mm_unpackhi_ps(src1, src1);
        _mm_store_ss(dst_c1 + 2, src1);
        dst_c1 += stride;
        break;
      case 4:
        _mm_storeu_ps(dst_c1, src1);
        dst_c1 += stride;
        break;
    }
  }
}
#endif
