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

#ifdef ENABLE_AVX
#include "nnacl/intrinsics/ms_simd_instructions.h"
#include "nnacl/fp32/common_func_fp32.h"
#include "nnacl/intrinsics/avx/common_utils.h"

void WinogradPostFuncBiasReluC8(float *dst, const float *src, const float *bias, size_t oc8div, size_t oc8mod,
                                size_t plane_size, size_t plane_stride, size_t relu_type) {
  size_t stride = oc8div + oc8mod;
  plane_stride /= sizeof(float);
  int loop_c8 = 0;
  size_t src_stride = plane_size * C8NUM + plane_stride;
  for (; loop_c8 <= (int)(oc8div)-C32NUM; loop_c8 += C32NUM) {
    size_t plane_size_tmp = plane_size;
    float *dst_c8 = dst + loop_c8;
    __m256 bias1 = _mm256_setzero_ps();
    __m256 bias2 = _mm256_setzero_ps();
    __m256 bias3 = _mm256_setzero_ps();
    __m256 bias4 = _mm256_setzero_ps();
    if (bias != NULL) {
      bias1 = _mm256_loadu_ps(bias);
      bias2 = _mm256_loadu_ps(bias + C8NUM);
      bias3 = _mm256_loadu_ps(bias + C16NUM);
      bias4 = _mm256_loadu_ps(bias + C24NUM);
      bias += C32NUM;
    }
    for (; plane_size_tmp >= C4NUM; plane_size_tmp -= C4NUM) {
      __m256 src1 = _mm256_loadu_ps(src);
      __m256 src2 = _mm256_loadu_ps(src + C8NUM);
      __m256 src5 = _mm256_loadu_ps(src + src_stride);
      __m256 src6 = _mm256_loadu_ps(src + src_stride + C8NUM);
      __m256 src9 = _mm256_loadu_ps(src + src_stride * C2NUM);
      __m256 src10 = _mm256_loadu_ps(src + src_stride * C2NUM + C8NUM);
      __m256 src13 = _mm256_loadu_ps(src + src_stride * C3NUM);
      __m256 src14 = _mm256_loadu_ps(src + src_stride * C3NUM + C8NUM);

      src1 = _mm256_add_ps(src1, bias1);
      src2 = _mm256_add_ps(src2, bias1);
      src5 = _mm256_add_ps(src5, bias2);
      src6 = _mm256_add_ps(src6, bias2);
      src9 = _mm256_add_ps(src9, bias3);
      src10 = _mm256_add_ps(src10, bias3);
      src13 = _mm256_add_ps(src13, bias4);
      src14 = _mm256_add_ps(src14, bias4);

      ActBlock8Avx(&src1, &src2, &src5, &src6, &src9, &src10, &src13, &src14, relu_type);

      _mm256_stream_ps(dst_c8, src1);
      _mm256_stream_ps(dst_c8 + C8NUM, src5);
      _mm256_stream_ps(dst_c8 + C16NUM, src9);
      _mm256_stream_ps(dst_c8 + C24NUM, src13);
      dst_c8 += stride;
      _mm256_stream_ps(dst_c8, src2);
      _mm256_stream_ps(dst_c8 + C8NUM, src6);
      _mm256_stream_ps(dst_c8 + C16NUM, src10);
      _mm256_stream_ps(dst_c8 + C24NUM, src14);
      dst_c8 += stride;

      __m256 src3 = _mm256_loadu_ps(src + C16NUM);
      __m256 src4 = _mm256_loadu_ps(src + C24NUM);
      __m256 src7 = _mm256_loadu_ps(src + src_stride + C16NUM);
      __m256 src8 = _mm256_loadu_ps(src + src_stride + C24NUM);
      __m256 src11 = _mm256_loadu_ps(src + src_stride * C2NUM + C16NUM);
      __m256 src12 = _mm256_loadu_ps(src + src_stride * C2NUM + C24NUM);
      __m256 src15 = _mm256_loadu_ps(src + src_stride * C3NUM + C16NUM);
      __m256 src16 = _mm256_loadu_ps(src + src_stride * C3NUM + C24NUM);
      src3 = _mm256_add_ps(src3, bias1);
      src4 = _mm256_add_ps(src4, bias1);
      src7 = _mm256_add_ps(src7, bias2);
      src8 = _mm256_add_ps(src8, bias2);
      src11 = _mm256_add_ps(src11, bias3);
      src12 = _mm256_add_ps(src12, bias3);
      src15 = _mm256_add_ps(src15, bias4);
      src16 = _mm256_add_ps(src16, bias4);

      ActBlock8Avx(&src3, &src4, &src7, &src8, &src11, &src12, &src15, &src16, relu_type);

      _mm256_stream_ps(dst_c8, src3);
      _mm256_stream_ps(dst_c8 + C8NUM, src7);
      _mm256_stream_ps(dst_c8 + C16NUM, src11);
      _mm256_stream_ps(dst_c8 + C24NUM, src15);
      dst_c8 += stride;
      _mm256_stream_ps(dst_c8, src4);
      _mm256_stream_ps(dst_c8 + C8NUM, src8);
      _mm256_stream_ps(dst_c8 + C16NUM, src12);
      _mm256_stream_ps(dst_c8 + C24NUM, src16);
      dst_c8 += stride;
      src += C32NUM;
    }
    for (; plane_size_tmp > 0; plane_size_tmp -= 1) {
      __m256 src1 = _mm256_loadu_ps(src);
      __m256 src2 = _mm256_loadu_ps(src + src_stride);
      __m256 src3 = _mm256_loadu_ps(src + src_stride * C2NUM);
      __m256 src4 = _mm256_loadu_ps(src + src_stride * C3NUM);
      src1 = _mm256_add_ps(src1, bias1);
      src2 = _mm256_add_ps(src2, bias2);
      src3 = _mm256_add_ps(src3, bias3);
      src4 = _mm256_add_ps(src4, bias4);

      ActBlock4Avx(&src1, &src2, &src3, &src4, relu_type == 1, relu_type == C3NUM);

      _mm256_stream_ps(dst_c8, src1);
      _mm256_stream_ps(dst_c8 + C8NUM, src2);
      _mm256_stream_ps(dst_c8 + C16NUM, src3);
      _mm256_stream_ps(dst_c8 + C24NUM, src4);
      dst_c8 += stride;
      src += C8NUM;
    }
    src += plane_stride;
    src += C3NUM * src_stride;
  }
  for (; loop_c8 <= (int)(oc8div)-C24NUM; loop_c8 += C24NUM) {
    size_t plane_size_tmp = plane_size;
    float *dst_c8 = dst + loop_c8;
    __m256 bias1 = _mm256_setzero_ps();
    __m256 bias2 = _mm256_setzero_ps();
    __m256 bias3 = _mm256_setzero_ps();
    if (bias != NULL) {
      bias1 = _mm256_loadu_ps(bias);
      bias2 = _mm256_loadu_ps(bias + C8NUM);
      bias3 = _mm256_loadu_ps(bias + C16NUM);
      bias += C24NUM;
    }
    for (; plane_size_tmp >= C4NUM; plane_size_tmp -= C4NUM) {
      __m256 src1 = _mm256_loadu_ps(src);
      __m256 src2 = _mm256_loadu_ps(src + C8NUM);
      __m256 src3 = _mm256_loadu_ps(src + C16NUM);
      __m256 src4 = _mm256_loadu_ps(src + C24NUM);
      __m256 src5 = _mm256_loadu_ps(src + src_stride);
      __m256 src6 = _mm256_loadu_ps(src + src_stride + C8NUM);
      __m256 src7 = _mm256_loadu_ps(src + src_stride + C16NUM);
      __m256 src8 = _mm256_loadu_ps(src + src_stride + C24NUM);
      __m256 src9 = _mm256_loadu_ps(src + src_stride * C2NUM);
      __m256 src10 = _mm256_loadu_ps(src + src_stride * C2NUM + C8NUM);
      __m256 src11 = _mm256_loadu_ps(src + src_stride * C2NUM + C16NUM);
      __m256 src12 = _mm256_loadu_ps(src + src_stride * C2NUM + C24NUM);
      src += C32NUM;
      src1 = _mm256_add_ps(src1, bias1);
      src2 = _mm256_add_ps(src2, bias1);
      src3 = _mm256_add_ps(src3, bias1);
      src4 = _mm256_add_ps(src4, bias1);
      src5 = _mm256_add_ps(src5, bias2);
      src6 = _mm256_add_ps(src6, bias2);
      src7 = _mm256_add_ps(src7, bias2);
      src8 = _mm256_add_ps(src8, bias2);
      src9 = _mm256_add_ps(src9, bias3);
      src10 = _mm256_add_ps(src10, bias3);
      src11 = _mm256_add_ps(src11, bias3);
      src12 = _mm256_add_ps(src12, bias3);

      ActBlock12Avx(&src1, &src2, &src3, &src4, &src5, &src6, &src7, &src8, &src9, &src10, &src11, &src12,
                    relu_type == 1, relu_type == C3NUM);

      _mm256_stream_ps(dst_c8, src1);
      _mm256_stream_ps(dst_c8 + C8NUM, src5);
      _mm256_stream_ps(dst_c8 + C16NUM, src9);
      dst_c8 += stride;
      _mm256_stream_ps(dst_c8, src2);
      _mm256_stream_ps(dst_c8 + C8NUM, src6);
      _mm256_stream_ps(dst_c8 + C16NUM, src10);
      dst_c8 += stride;
      _mm256_stream_ps(dst_c8, src3);
      _mm256_stream_ps(dst_c8 + C8NUM, src7);
      _mm256_stream_ps(dst_c8 + C16NUM, src11);
      dst_c8 += stride;
      _mm256_stream_ps(dst_c8, src4);
      _mm256_stream_ps(dst_c8 + C8NUM, src8);
      _mm256_stream_ps(dst_c8 + C16NUM, src12);
      dst_c8 += stride;
    }
    for (; plane_size_tmp > 0; plane_size_tmp -= 1) {
      __m256 src1 = _mm256_loadu_ps(src);
      __m256 src2 = _mm256_loadu_ps(src + src_stride);
      __m256 src3 = _mm256_loadu_ps(src + src_stride * C2NUM);
      src1 = _mm256_add_ps(src1, bias1);
      src2 = _mm256_add_ps(src2, bias2);
      src3 = _mm256_add_ps(src3, bias3);

      ActBlock1Avx(&src1, relu_type == 1, relu_type == C3NUM);
      ActBlock1Avx(&src2, relu_type == 1, relu_type == C3NUM);
      ActBlock1Avx(&src3, relu_type == 1, relu_type == C3NUM);

      _mm256_stream_ps(dst_c8, src1);
      _mm256_stream_ps(dst_c8 + C8NUM, src2);
      _mm256_stream_ps(dst_c8 + C16NUM, src3);
      dst_c8 += stride;
      src += C8NUM;
    }
    src += plane_stride;
    src += C2NUM * src_stride;
  }
  for (; loop_c8 <= (int)(oc8div)-C16NUM; loop_c8 += C16NUM) {
    size_t plane_size_tmp = plane_size;
    float *dst_c8 = dst + loop_c8;
    __m256 bias1 = _mm256_setzero_ps();
    __m256 bias2 = _mm256_setzero_ps();
    if (bias != NULL) {
      bias1 = _mm256_loadu_ps(bias);
      bias2 = _mm256_loadu_ps(bias + C8NUM);
      bias += C16NUM;
    }
    for (; plane_size_tmp >= C4NUM; plane_size_tmp -= C4NUM) {
      __m256 src1 = _mm256_loadu_ps(src);
      __m256 src2 = _mm256_loadu_ps(src + C8NUM);
      __m256 src3 = _mm256_loadu_ps(src + C16NUM);
      __m256 src4 = _mm256_loadu_ps(src + C24NUM);
      __m256 src5 = _mm256_loadu_ps(src + src_stride);
      __m256 src6 = _mm256_loadu_ps(src + src_stride + C8NUM);
      __m256 src7 = _mm256_loadu_ps(src + src_stride + C16NUM);
      __m256 src8 = _mm256_loadu_ps(src + src_stride + C24NUM);
      src += C32NUM;
      src1 = _mm256_add_ps(src1, bias1);
      src2 = _mm256_add_ps(src2, bias1);
      src3 = _mm256_add_ps(src3, bias1);
      src4 = _mm256_add_ps(src4, bias1);
      src5 = _mm256_add_ps(src5, bias2);
      src6 = _mm256_add_ps(src6, bias2);
      src7 = _mm256_add_ps(src7, bias2);
      src8 = _mm256_add_ps(src8, bias2);

      ActBlock8Avx(&src1, &src2, &src3, &src4, &src5, &src6, &src7, &src8, relu_type);

      _mm256_stream_ps(dst_c8, src1);
      _mm256_stream_ps(dst_c8 + C8NUM, src5);
      dst_c8 += stride;
      _mm256_stream_ps(dst_c8, src2);
      _mm256_stream_ps(dst_c8 + C8NUM, src6);
      dst_c8 += stride;
      _mm256_stream_ps(dst_c8, src3);
      _mm256_stream_ps(dst_c8 + C8NUM, src7);
      dst_c8 += stride;
      _mm256_stream_ps(dst_c8, src4);
      _mm256_stream_ps(dst_c8 + C8NUM, src8);
      dst_c8 += stride;
    }
    for (; plane_size_tmp > 0; plane_size_tmp -= 1) {
      __m256 src1 = _mm256_loadu_ps(src);
      __m256 src2 = _mm256_loadu_ps(src + src_stride);
      src1 = _mm256_add_ps(src1, bias1);
      src2 = _mm256_add_ps(src2, bias2);

      ActBlock2Avx(&src1, &src2, relu_type == 1, relu_type == C3NUM);

      _mm256_stream_ps(dst_c8, src1);
      _mm256_stream_ps(dst_c8 + C8NUM, src2);
      dst_c8 += stride;
      src += C8NUM;
    }
    src += plane_stride;
    src += src_stride;
  }
  for (; loop_c8 < (int)(oc8div); loop_c8 += C8NUM) {
    size_t plane_size_tmp = plane_size;
    float *dst_c8 = dst + loop_c8;
    __m256 bias1 = _mm256_setzero_ps();
    if (bias != NULL) {
      bias1 = _mm256_loadu_ps(bias);
      bias += C8NUM;
    }
    for (; plane_size_tmp >= C4NUM; plane_size_tmp -= C4NUM) {
      __m256 src1 = _mm256_loadu_ps(src);
      __m256 src2 = _mm256_loadu_ps(src + C8NUM);
      __m256 src3 = _mm256_loadu_ps(src + C16NUM);
      __m256 src4 = _mm256_loadu_ps(src + C24NUM);
      src += C32NUM;
      src1 = _mm256_add_ps(src1, bias1);
      src2 = _mm256_add_ps(src2, bias1);
      src3 = _mm256_add_ps(src3, bias1);
      src4 = _mm256_add_ps(src4, bias1);

      ActBlock4Avx(&src1, &src2, &src3, &src4, relu_type == 1, relu_type == C3NUM);

      _mm256_stream_ps(dst_c8, src1);
      dst_c8 += stride;
      _mm256_stream_ps(dst_c8, src2);
      dst_c8 += stride;
      _mm256_stream_ps(dst_c8, src3);
      dst_c8 += stride;
      _mm256_stream_ps(dst_c8, src4);
      dst_c8 += stride;
    }
    for (; plane_size_tmp > 0; plane_size_tmp -= 1) {
      __m256 src1 = _mm256_loadu_ps(src);
      src1 = _mm256_add_ps(src1, bias1);

      ActBlock1Avx(&src1, relu_type == 1, relu_type == C3NUM);

      _mm256_stream_ps(dst_c8, src1);
      dst_c8 += stride;
      src += C8NUM;
    }
    src += plane_stride;
  }
  if (oc8mod == 0) {
    return;
  }
  __m256 bias1 = _mm256_setzero_ps();
  if (bias != NULL) {
    bias1 = _mm256_loadu_ps(bias);
    bias += C8NUM;
  }
  float *dst_c1 = dst + oc8div;
  for (size_t plane_size_tmp = plane_size; plane_size_tmp > 0; plane_size_tmp -= 1, src += C8NUM, dst_c1 += stride) {
    __m256 src1 = _mm256_loadu_ps(src);
    src1 = _mm256_add_ps(src1, bias1);

    ActBlock1Avx(&src1, relu_type == 1, relu_type == C3NUM);
    __m128 src_high = _mm256_extractf128_ps(src1, 1);

    switch (oc8mod) {
      case 1:
        dst_c1[0] = _mm256_cvtss_f32(src1);
        break;
      case C2NUM:
        _mm_storel_pi((__m64 *)(dst_c1), _mm256_castps256_ps128(src1));
        break;
      case C3NUM:
        _mm_storel_pi((__m64 *)(dst_c1), _mm256_castps256_ps128(src1));
        dst_c1[C2NUM] = MS_F32X8_GETI(src1, C2NUM);
        break;
      case C4NUM:
        _mm_storeu_ps(dst_c1, _mm256_castps256_ps128(src1));
        break;
      case C5NUM:
        _mm_storeu_ps(dst_c1, _mm256_castps256_ps128(src1));
        _mm_store_ss(dst_c1 + C4NUM, src_high);
        break;
      case C6NUM:
        _mm_storeu_ps(dst_c1, _mm256_castps256_ps128(src1));
        _mm_storel_pi((__m64 *)(dst_c1 + C4NUM), src_high);
        break;
      case C7NUM:
        _mm_storeu_ps(dst_c1, _mm256_castps256_ps128(src1));
        _mm_storel_pi((__m64 *)(dst_c1 + C4NUM), src_high);
        dst_c1[C6NUM] = MS_F32X4_GETI(src_high, C2NUM);
        break;
      default:
        break;
    }
  }
}
#endif
