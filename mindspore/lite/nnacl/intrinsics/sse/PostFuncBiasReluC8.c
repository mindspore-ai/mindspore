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

void PostFuncBiasReluC8(float *dst, const float *src, const float *bias, size_t oc8div, size_t oc8mod,
                        size_t plane_size, size_t stride, size_t relu_type) {
  stride /= sizeof(float);
  for (int loop_c8 = 0; !(loop_c8 == oc8div); loop_c8 += C8NUM) {
    size_t plane_size_tmp = plane_size;
    float *dst_c8 = dst + loop_c8;
    __m128 bias1 = _mm_setzero_ps(), bias2 = _mm_setzero_ps();
    if (bias != NULL) {
      bias1 = _mm_loadu_ps(bias);
      bias2 = _mm_loadu_ps(bias + 4);
      bias += 8;
    }
    for (; plane_size_tmp >= C4NUM; plane_size_tmp -= C4NUM, src += 32) {
      __m128 src1 = _mm_loadu_ps(src);
      __m128 src2 = _mm_loadu_ps(src + 4);
      __m128 src3 = _mm_loadu_ps(src + 8);
      __m128 src4 = _mm_loadu_ps(src + 12);
      __m128 src5 = _mm_loadu_ps(src + 16);
      __m128 src6 = _mm_loadu_ps(src + 20);
      __m128 src7 = _mm_loadu_ps(src + 24);
      __m128 src8 = _mm_loadu_ps(src + 28);
      src1 = _mm_add_ps(src1, bias1);
      src2 = _mm_add_ps(src2, bias2);
      src3 = _mm_add_ps(src3, bias1);
      src4 = _mm_add_ps(src4, bias2);
      src5 = _mm_add_ps(src5, bias1);
      src6 = _mm_add_ps(src6, bias2);
      src7 = _mm_add_ps(src7, bias1);
      src8 = _mm_add_ps(src8, bias2);

      ActBlock8(&src1, &src2, &src3, &src4, &src5, &src6, &src7, &src8, relu_type);

      _mm_storeu_ps(dst_c8, src1);
      _mm_storeu_ps(dst_c8 + 4, src2);
      dst_c8 += stride;
      _mm_storeu_ps(dst_c8, src3);
      _mm_storeu_ps(dst_c8 + 4, src4);
      dst_c8 += stride;
      _mm_storeu_ps(dst_c8, src5);
      _mm_storeu_ps(dst_c8 + 4, src6);
      dst_c8 += stride;
      _mm_storeu_ps(dst_c8, src7);
      _mm_storeu_ps(dst_c8 + 4, src8);
      dst_c8 += stride;
    }
    for (; plane_size_tmp > 0; plane_size_tmp -= 1, src += 8, dst_c8 += stride) {
      __m128 src1 = _mm_loadu_ps(src);
      __m128 src2 = _mm_loadu_ps(src + 4);
      src1 = _mm_add_ps(src1, bias1);
      src2 = _mm_add_ps(src2, bias2);

      ActBlock2(&src1, &src2, relu_type == 1, relu_type == 3);

      _mm_storeu_ps(dst_c8, src1);
      _mm_storeu_ps(dst_c8 + 4, src2);
    }
  }

  if ((oc8mod == 0)) return;

  __m128 bias1 = _mm_setzero_ps();
  __m128 bias2 = _mm_setzero_ps();
  if (bias != NULL) {
    bias1 = _mm_loadu_ps(bias);
    bias2 = _mm_loadu_ps(bias + 4);
    bias += 8;
  }
  float *dst_c1 = dst + oc8div;
  for (size_t plane_size_tmp = plane_size; plane_size_tmp > 0; plane_size_tmp -= 1, src += 8, dst_c1 += stride) {
    __m128 src1 = _mm_loadu_ps(src);
    __m128 src2 = _mm_loadu_ps(src + 4);
    src1 = _mm_add_ps(src1, bias1);
    src2 = _mm_add_ps(src2, bias2);

    ActBlock2(&src1, &src2, relu_type == 1, relu_type == 3);

    switch (oc8mod) {
      case 1:
        _mm_store_ss(dst_c1, src1);
        break;
      case 2:
        _mm_storel_pi((__m64 *)(dst_c1), src1);
        break;
      case 3:
        _mm_storel_pi((__m64 *)(dst_c1), src1);
        src1 = _mm_unpackhi_ps(src1, src1);
        _mm_store_ss(dst_c1 + 2, src1);
        break;
      case 4:
        _mm_storeu_ps(dst_c1, src1);
        break;
      case 5:
        _mm_storeu_ps(dst_c1, src1);
        _mm_store_ss(dst_c1 + 4, src2);
        break;
      case 6:
        _mm_storeu_ps(dst_c1, src1);
        _mm_storel_pi((__m64 *)(dst_c1 + 4), src2);
        break;
      case 7:
        _mm_storeu_ps(dst_c1, src1);
        _mm_storel_pi((__m64 *)(dst_c1 + 4), src2);
        src2 = _mm_unpackhi_ps(src2, src2);
        _mm_store_ss(dst_c1 + 6, src2);
        break;
    }
  }
}
#endif
