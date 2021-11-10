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

#if defined(ENABLE_SSE) && !defined(ENABLE_AVX)
#include "nnacl/intrinsics/ms_simd_instructions.h"
#include "nnacl/fp32/common_func_fp32.h"
#include "nnacl/intrinsics/sse/sse_common.h"

void WinogradPostFuncBiasReluC4(float *dst, const float *src, const float *bias, size_t oc4div, size_t oc4mod,
                                size_t plane_size, size_t plane_stride, size_t relu_type) {
  size_t stride = oc4div + oc4mod;
  plane_stride /= sizeof(float);
  int loop_c4 = 0;
  size_t src_stride = plane_size * C4NUM + plane_stride;
  for (; loop_c4 <= (int)(oc4div)-C16NUM; loop_c4 += C16NUM) {
    size_t plane_size_tmp = plane_size;
    float *dst_c4 = dst + loop_c4;
    __m128 bias1 = _mm_setzero_ps();
    __m128 bias2 = _mm_setzero_ps();
    __m128 bias3 = _mm_setzero_ps();
    __m128 bias4 = _mm_setzero_ps();
    if (bias != NULL) {
      bias1 = _mm_loadu_ps(bias);
      bias2 = _mm_loadu_ps(bias + C4NUM);
      bias3 = _mm_loadu_ps(bias + C8NUM);
      bias4 = _mm_loadu_ps(bias + C12NUM);
      bias += C16NUM;
    }
    for (; plane_size_tmp >= C4NUM; plane_size_tmp -= C4NUM) {
      __m128 src1 = _mm_loadu_ps(src);
      __m128 src2 = _mm_loadu_ps(src + C4NUM);
      __m128 src5 = _mm_loadu_ps(src + src_stride);
      __m128 src6 = _mm_loadu_ps(src + src_stride + C4NUM);
      __m128 src9 = _mm_loadu_ps(src + src_stride * C2NUM);
      __m128 src10 = _mm_loadu_ps(src + src_stride * C2NUM + C4NUM);
      __m128 src13 = _mm_loadu_ps(src + src_stride * C3NUM);
      __m128 src14 = _mm_loadu_ps(src + src_stride * C3NUM + C4NUM);

      src1 = _mm_add_ps(src1, bias1);
      src2 = _mm_add_ps(src2, bias1);
      src5 = _mm_add_ps(src5, bias2);
      src6 = _mm_add_ps(src6, bias2);
      src9 = _mm_add_ps(src9, bias3);
      src10 = _mm_add_ps(src10, bias3);
      src13 = _mm_add_ps(src13, bias4);
      src14 = _mm_add_ps(src14, bias4);

      ActBlock8(&src1, &src2, &src5, &src6, &src9, &src10, &src13, &src14, relu_type);

      _mm_storeu_ps(dst_c4, src1);
      _mm_storeu_ps(dst_c4 + C4NUM, src5);
      _mm_storeu_ps(dst_c4 + C8NUM, src9);
      _mm_storeu_ps(dst_c4 + C12NUM, src13);
      dst_c4 += stride;
      _mm_storeu_ps(dst_c4, src2);
      _mm_storeu_ps(dst_c4 + C4NUM, src6);
      _mm_storeu_ps(dst_c4 + C8NUM, src10);
      _mm_storeu_ps(dst_c4 + C12NUM, src14);
      dst_c4 += stride;

      __m128 src3 = _mm_loadu_ps(src + C8NUM);
      __m128 src4 = _mm_loadu_ps(src + C12NUM);
      __m128 src7 = _mm_loadu_ps(src + src_stride + C8NUM);
      __m128 src8 = _mm_loadu_ps(src + src_stride + C12NUM);
      __m128 src11 = _mm_loadu_ps(src + src_stride * C2NUM + C8NUM);
      __m128 src12 = _mm_loadu_ps(src + src_stride * C2NUM + C12NUM);
      __m128 src15 = _mm_loadu_ps(src + src_stride * C3NUM + C8NUM);
      __m128 src16 = _mm_loadu_ps(src + src_stride * C3NUM + C12NUM);
      src3 = _mm_add_ps(src3, bias1);
      src4 = _mm_add_ps(src4, bias1);
      src7 = _mm_add_ps(src7, bias2);
      src8 = _mm_add_ps(src8, bias2);
      src11 = _mm_add_ps(src11, bias3);
      src12 = _mm_add_ps(src12, bias3);
      src15 = _mm_add_ps(src15, bias4);
      src16 = _mm_add_ps(src16, bias4);

      ActBlock8(&src3, &src4, &src7, &src8, &src11, &src12, &src15, &src16, relu_type);

      _mm_storeu_ps(dst_c4, src3);
      _mm_storeu_ps(dst_c4 + C4NUM, src7);
      _mm_storeu_ps(dst_c4 + C8NUM, src11);
      _mm_storeu_ps(dst_c4 + C12NUM, src15);
      dst_c4 += stride;
      _mm_storeu_ps(dst_c4, src4);
      _mm_storeu_ps(dst_c4 + C4NUM, src8);
      _mm_storeu_ps(dst_c4 + C8NUM, src12);
      _mm_storeu_ps(dst_c4 + C12NUM, src16);
      dst_c4 += stride;
      src += C16NUM;
    }
    for (; plane_size_tmp > 0; plane_size_tmp -= 1) {
      __m128 src1 = _mm_loadu_ps(src);
      __m128 src2 = _mm_loadu_ps(src + src_stride);
      __m128 src3 = _mm_loadu_ps(src + src_stride * C2NUM);
      __m128 src4 = _mm_loadu_ps(src + src_stride * C3NUM);
      src1 = _mm_add_ps(src1, bias1);
      src2 = _mm_add_ps(src2, bias2);
      src3 = _mm_add_ps(src3, bias3);
      src4 = _mm_add_ps(src4, bias4);

      ActBlock4(&src1, &src2, &src3, &src4, relu_type == 1, relu_type == C3NUM);

      _mm_storeu_ps(dst_c4, src1);
      _mm_storeu_ps(dst_c4 + C4NUM, src2);
      _mm_storeu_ps(dst_c4 + C8NUM, src3);
      _mm_storeu_ps(dst_c4 + C12NUM, src4);
      dst_c4 += stride;
      src += C4NUM;
    }
    src += plane_stride;
    src += C3NUM * src_stride;
  }
  for (; loop_c4 <= (int)(oc4div)-C12NUM; loop_c4 += C12NUM) {
    size_t plane_size_tmp = plane_size;
    float *dst_c4 = dst + loop_c4;
    __m128 bias1 = _mm_setzero_ps();
    __m128 bias2 = _mm_setzero_ps();
    __m128 bias3 = _mm_setzero_ps();
    if (bias != NULL) {
      bias1 = _mm_loadu_ps(bias);
      bias2 = _mm_loadu_ps(bias + C4NUM);
      bias3 = _mm_loadu_ps(bias + C8NUM);
      bias += C12NUM;
    }
    for (; plane_size_tmp >= C4NUM; plane_size_tmp -= C4NUM) {
      __m128 src1 = _mm_loadu_ps(src);
      __m128 src2 = _mm_loadu_ps(src + C4NUM);
      __m128 src3 = _mm_loadu_ps(src + C8NUM);
      __m128 src4 = _mm_loadu_ps(src + C12NUM);
      __m128 src5 = _mm_loadu_ps(src + src_stride);
      __m128 src6 = _mm_loadu_ps(src + src_stride + C4NUM);
      __m128 src7 = _mm_loadu_ps(src + src_stride + C8NUM);
      __m128 src8 = _mm_loadu_ps(src + src_stride + C12NUM);
      __m128 src9 = _mm_loadu_ps(src + src_stride * C2NUM);
      __m128 src10 = _mm_loadu_ps(src + src_stride * C2NUM + C4NUM);
      __m128 src11 = _mm_loadu_ps(src + src_stride * C2NUM + C8NUM);
      __m128 src12 = _mm_loadu_ps(src + src_stride * C2NUM + C12NUM);
      src += C16NUM;
      src1 = _mm_add_ps(src1, bias1);
      src2 = _mm_add_ps(src2, bias1);
      src3 = _mm_add_ps(src3, bias1);
      src4 = _mm_add_ps(src4, bias1);
      src5 = _mm_add_ps(src5, bias2);
      src6 = _mm_add_ps(src6, bias2);
      src7 = _mm_add_ps(src7, bias2);
      src8 = _mm_add_ps(src8, bias2);
      src9 = _mm_add_ps(src9, bias3);
      src10 = _mm_add_ps(src10, bias3);
      src11 = _mm_add_ps(src11, bias3);
      src12 = _mm_add_ps(src12, bias3);

      ActBlock12(&src1, &src2, &src3, &src4, &src5, &src6, &src7, &src8, &src9, &src10, &src11, &src12, relu_type == 1,
                 relu_type == C3NUM);

      _mm_storeu_ps(dst_c4, src1);
      _mm_storeu_ps(dst_c4 + C4NUM, src5);
      _mm_storeu_ps(dst_c4 + C8NUM, src9);
      dst_c4 += stride;
      _mm_storeu_ps(dst_c4, src2);
      _mm_storeu_ps(dst_c4 + C4NUM, src6);
      _mm_storeu_ps(dst_c4 + C8NUM, src10);
      dst_c4 += stride;
      _mm_storeu_ps(dst_c4, src3);
      _mm_storeu_ps(dst_c4 + C4NUM, src7);
      _mm_storeu_ps(dst_c4 + C8NUM, src11);
      dst_c4 += stride;
      _mm_storeu_ps(dst_c4, src4);
      _mm_storeu_ps(dst_c4 + C4NUM, src8);
      _mm_storeu_ps(dst_c4 + C8NUM, src12);
      dst_c4 += stride;
    }
    for (; plane_size_tmp > 0; plane_size_tmp -= 1) {
      __m128 src1 = _mm_loadu_ps(src);
      __m128 src2 = _mm_loadu_ps(src + src_stride);
      __m128 src3 = _mm_loadu_ps(src + src_stride * C2NUM);
      src1 = _mm_add_ps(src1, bias1);
      src2 = _mm_add_ps(src2, bias2);
      src3 = _mm_add_ps(src3, bias3);

      ActBlock1(&src1, relu_type == 1, relu_type == C3NUM);
      ActBlock1(&src2, relu_type == 1, relu_type == C3NUM);
      ActBlock1(&src3, relu_type == 1, relu_type == C3NUM);

      _mm_storeu_ps(dst_c4, src1);
      _mm_storeu_ps(dst_c4 + C4NUM, src2);
      _mm_storeu_ps(dst_c4 + C8NUM, src3);
      dst_c4 += stride;
      src += C4NUM;
    }
    src += plane_stride;
    src += C2NUM * src_stride;
  }

  for (; loop_c4 <= (int)(oc4div)-C8NUM; loop_c4 += C8NUM) {
    size_t plane_size_tmp = plane_size;
    float *dst_c4 = dst + loop_c4;
    __m128 bias1 = _mm_setzero_ps();
    __m128 bias2 = _mm_setzero_ps();
    if (bias != NULL) {
      bias1 = _mm_loadu_ps(bias);
      bias2 = _mm_loadu_ps(bias + C4NUM);
      bias += C8NUM;
    }
    for (; plane_size_tmp >= C4NUM; plane_size_tmp -= C4NUM) {
      __m128 src1 = _mm_loadu_ps(src);
      __m128 src2 = _mm_loadu_ps(src + C4NUM);
      __m128 src3 = _mm_loadu_ps(src + C8NUM);
      __m128 src4 = _mm_loadu_ps(src + C12NUM);
      __m128 src5 = _mm_loadu_ps(src + src_stride);
      __m128 src6 = _mm_loadu_ps(src + src_stride + C4NUM);
      __m128 src7 = _mm_loadu_ps(src + src_stride + C8NUM);
      __m128 src8 = _mm_loadu_ps(src + src_stride + C12NUM);
      src += C16NUM;
      src1 = _mm_add_ps(src1, bias1);
      src2 = _mm_add_ps(src2, bias1);
      src3 = _mm_add_ps(src3, bias1);
      src4 = _mm_add_ps(src4, bias1);
      src5 = _mm_add_ps(src5, bias2);
      src6 = _mm_add_ps(src6, bias2);
      src7 = _mm_add_ps(src7, bias2);
      src8 = _mm_add_ps(src8, bias2);

      ActBlock8(&src1, &src2, &src3, &src4, &src5, &src6, &src7, &src8, relu_type);

      _mm_storeu_ps(dst_c4, src1);
      _mm_storeu_ps(dst_c4 + C4NUM, src5);
      dst_c4 += stride;
      _mm_storeu_ps(dst_c4, src2);
      _mm_storeu_ps(dst_c4 + C4NUM, src6);
      dst_c4 += stride;
      _mm_storeu_ps(dst_c4, src3);
      _mm_storeu_ps(dst_c4 + C4NUM, src7);
      dst_c4 += stride;
      _mm_storeu_ps(dst_c4, src4);
      _mm_storeu_ps(dst_c4 + C4NUM, src8);
      dst_c4 += stride;
    }
    for (; plane_size_tmp > 0; plane_size_tmp -= 1) {
      __m128 src1 = _mm_loadu_ps(src);
      __m128 src2 = _mm_loadu_ps(src + src_stride);
      src1 = _mm_add_ps(src1, bias1);
      src2 = _mm_add_ps(src2, bias2);

      ActBlock1(&src1, relu_type == 1, relu_type == C3NUM);
      ActBlock1(&src2, relu_type == 1, relu_type == C3NUM);

      _mm_storeu_ps(dst_c4, src1);
      _mm_storeu_ps(dst_c4 + C4NUM, src2);
      dst_c4 += stride;
      src += C4NUM;
    }
    src += plane_stride;
    src += src_stride;
  }
  for (; loop_c4 < (int)(oc4div); loop_c4 += C4NUM) {
    size_t plane_size_tmp = plane_size;
    float *dst_c4 = dst + loop_c4;
    __m128 bias1 = _mm_setzero_ps();
    if (bias != NULL) {
      bias1 = _mm_loadu_ps(bias);
      bias += C4NUM;
    }
    for (; plane_size_tmp >= C4NUM; plane_size_tmp -= C4NUM) {
      __m128 src1 = _mm_loadu_ps(src);
      __m128 src2 = _mm_loadu_ps(src + C4NUM);
      __m128 src3 = _mm_loadu_ps(src + 8);
      __m128 src4 = _mm_loadu_ps(src + 12);
      src += C16NUM;
      src1 = _mm_add_ps(src1, bias1);
      src2 = _mm_add_ps(src2, bias1);
      src3 = _mm_add_ps(src3, bias1);
      src4 = _mm_add_ps(src4, bias1);

      ActBlock4(&src1, &src2, &src3, &src4, relu_type == 1, relu_type == C3NUM);

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

      ActBlock1(&src1, relu_type == 1, relu_type == C3NUM);

      _mm_storeu_ps(dst_c4, src1);
      dst_c4 += stride;
      src += C4NUM;
    }
    src += plane_stride;
  }
  if (oc4mod == 0) {
    return;
  }
  __m128 bias1 = _mm_setzero_ps();
  if (bias != NULL) {
    bias1 = _mm_loadu_ps(bias);
    bias += C4NUM;
  }
  float *dst_c1 = dst + oc4div;
  for (size_t plane_size_tmp = plane_size; plane_size_tmp > 0; plane_size_tmp -= 1) {
    __m128 src1 = _mm_loadu_ps(src);
    src += C4NUM;
    src1 = _mm_add_ps(src1, bias1);

    ActBlock1(&src1, relu_type == 1, relu_type == C3NUM);

    switch (oc4mod) {
      case 1:
        _mm_store_ss(dst_c1, src1);
        dst_c1 += stride;
        break;
      case C2NUM:
        _mm_storel_pi((__m64 *)(dst_c1), src1);
        dst_c1 += stride;
        break;
      case C3NUM:
        _mm_storel_pi((__m64 *)(dst_c1), src1);
        src1 = _mm_unpackhi_ps(src1, src1);
        _mm_store_ss(dst_c1 + C2NUM, src1);
        dst_c1 += stride;
        break;
      case C4NUM:
        _mm_storeu_ps(dst_c1, src1);
        dst_c1 += stride;
        break;
    }
  }
}
#endif
