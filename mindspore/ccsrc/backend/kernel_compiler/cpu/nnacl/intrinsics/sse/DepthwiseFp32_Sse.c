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
#include "nnacl/fp32/conv_depthwise_fp32.h"
#include "nnacl/intrinsics/sse/sse_common.h"

#ifndef ENABLE_AVX
void ConvDwFp32Border(float *dst, const float *src, const float *weight, const float *bias, size_t height, size_t width,
                      size_t in_kh_step, size_t in_kw_step, size_t kernel_w_step, size_t relu, size_t relu6) {
  in_kh_step /= sizeof(float);
  in_kw_step /= sizeof(float);
  kernel_w_step /= sizeof(float);

  const float *src_kh = src;
  const float *weight_kh = weight;
  __m128 dst_ma = _mm_setzero_ps();

  for (int kh = 0; kh < height; kh++) {
    const float *src_kw = src_kh;
    const float *weight_kw = weight_kh;

    int c1 = 0;
    int c4 = DOWN_DIV(width, C4NUM) * C4NUM;
    int c2 = DOWN_DIV(width, C2NUM) * C2NUM;
    // c4 loop
    for (; c1 < c4; c1 += C4NUM) {
      __m128 src_ma1 = _mm_loadu_ps(src_kw);
      __m128 src_ma2 = _mm_loadu_ps(src_kw + in_kw_step);
      __m128 src_ma3 = _mm_loadu_ps(src_kw + 2 * in_kw_step);
      __m128 src_ma4 = _mm_loadu_ps(src_kw + 3 * in_kw_step);

      __m128 weight_ma1 = _mm_loadu_ps(weight_kw);
      __m128 weight_ma2 = _mm_loadu_ps(weight_kw + C4NUM);
      __m128 weight_ma3 = _mm_loadu_ps(weight_kw + 2 * C4NUM);
      __m128 weight_ma4 = _mm_loadu_ps(weight_kw + 3 * C4NUM);

      __m128 mul_ma1 = _mm_mul_ps(src_ma1, weight_ma1);
      __m128 mul_ma2 = _mm_mul_ps(src_ma2, weight_ma2);
      __m128 mul_ma3 = _mm_mul_ps(src_ma3, weight_ma3);
      __m128 mul_ma4 = _mm_mul_ps(src_ma4, weight_ma4);

      dst_ma = _mm_add_ps(dst_ma, mul_ma1);
      dst_ma = _mm_add_ps(dst_ma, mul_ma2);
      dst_ma = _mm_add_ps(dst_ma, mul_ma3);
      dst_ma = _mm_add_ps(dst_ma, mul_ma4);

      src_kw += in_kw_step * 4;
      weight_kw += C4NUM * 4;
    }

    // c2 loop
    for (; c1 < c2; c1 += C2NUM) {
      __m128 src_ma1 = _mm_loadu_ps(src_kw);
      __m128 src_ma2 = _mm_loadu_ps(src_kw + in_kw_step);
      __m128 weight_ma1 = _mm_loadu_ps(weight_kw);
      __m128 weight_ma2 = _mm_loadu_ps(weight_kw + C4NUM);
      __m128 mul_ma1 = _mm_mul_ps(src_ma1, weight_ma1);
      __m128 mul_ma2 = _mm_mul_ps(src_ma2, weight_ma2);
      dst_ma = _mm_add_ps(dst_ma, mul_ma1);
      dst_ma = _mm_add_ps(dst_ma, mul_ma2);

      src_kw += in_kw_step * 2;
      weight_kw += C4NUM * 2;
    }

    // remaining
    for (; c1 < width; ++c1) {
      __m128 src_ma1 = _mm_loadu_ps(src_kw);
      __m128 weight_ma1 = _mm_loadu_ps(weight_kw);
      __m128 mul_ma1 = _mm_mul_ps(src_ma1, weight_ma1);
      dst_ma = _mm_add_ps(dst_ma, mul_ma1);

      src_kw += in_kw_step;
      weight_kw += C4NUM;
    }

    src_kh += in_kh_step;
    weight_kh += kernel_w_step;
  }

  __m128 bias_ma = _mm_loadu_ps(bias);
  dst_ma = _mm_add_ps(dst_ma, bias_ma);
  __m128 zero_ma = _mm_setzero_ps();
  if (relu || relu6) {
    dst_ma = _mm_max_ps(zero_ma, dst_ma);
    if (relu6) {
      __m128 const_ma = _mm_set_ps(6.0f, 6.0f, 6.0f, 6.0f);
      dst_ma = _mm_min_ps(const_ma, dst_ma);
    }
  }
  _mm_storeu_ps(dst, dst_ma);
}
#endif

void ConvDwFp32Center(float *dst, const float *src, const float *weight, const float *bias, size_t height, size_t width,
                      size_t kernel_h, size_t kernel_w, size_t out_h_step, size_t block_channel, size_t in_sh_step,
                      size_t in_sw_step, size_t in_kh_step, size_t in_kw_step, size_t relu, size_t relu6) {
  out_h_step /= sizeof(float);
  block_channel /= sizeof(float);
  in_sh_step /= sizeof(float);
  in_sw_step /= sizeof(float);
  in_kh_step /= sizeof(float);
  in_kw_step /= sizeof(float);

  float *dst_h = dst;
  const float *src_h = src;
  for (int oh = 0; oh < height; oh++) {
    float *dst_w = dst_h;
    const float *src_w = src_h;
    int c4 = DOWN_DIV(width, C4NUM) * C4NUM;
    int c2 = DOWN_DIV(width, C2NUM) * C2NUM;
    int c1 = 0;
    // c4 loop
    for (; c1 < c4; c1 += C4NUM, dst_w += C4NUM * block_channel, src_w += C4NUM * in_sw_step) {
      const float *src_kh = src_w, *weight_kh = weight;
      __m128 dst_w_ma1 = _mm_setzero_ps();
      __m128 dst_w_ma2 = _mm_setzero_ps();
      __m128 dst_w_ma3 = _mm_setzero_ps();
      __m128 dst_w_ma4 = _mm_setzero_ps();

      for (int kh = 0; kh < kernel_h; kh++, src_kh += in_kh_step, weight_kh += kernel_w * C4NUM) {
        const float *src_kw = src_kh, *weight_kw = weight_kh;
        for (int kw = 0; kw < kernel_w; kw++, src_kw += in_kw_step, weight_kw += C4NUM) {
          __m128 src_kw_ma1 = _mm_loadu_ps(src_kw);
          __m128 weight_kw_ma1 = _mm_loadu_ps(weight_kw);
          __m128 tmp_ma1 = _mm_mul_ps(src_kw_ma1, weight_kw_ma1);
          dst_w_ma1 = _mm_add_ps(dst_w_ma1, tmp_ma1);

          __m128 src_kw_ma2 = _mm_loadu_ps(src_kw + in_sw_step);
          __m128 weight_kw_ma2 = _mm_loadu_ps(weight_kw);
          __m128 tmp_ma2 = _mm_mul_ps(src_kw_ma2, weight_kw_ma2);
          dst_w_ma2 = _mm_add_ps(dst_w_ma2, tmp_ma2);

          __m128 src_kw_ma3 = _mm_loadu_ps(src_kw + 2 * in_sw_step);
          __m128 weight_kw_ma3 = _mm_loadu_ps(weight_kw);
          __m128 tmp_ma3 = _mm_mul_ps(src_kw_ma3, weight_kw_ma3);
          dst_w_ma3 = _mm_add_ps(dst_w_ma3, tmp_ma3);

          __m128 src_kw_ma4 = _mm_loadu_ps(src_kw + 3 * in_sw_step);
          __m128 weight_kw_ma4 = _mm_loadu_ps(weight_kw);
          __m128 tmp_ma4 = _mm_mul_ps(src_kw_ma4, weight_kw_ma4);
          dst_w_ma4 = _mm_add_ps(dst_w_ma4, tmp_ma4);
        }  // kernel_w loop
      }    // kernel_h loop

      // add bias relu
      __m128 bias_ma = _mm_loadu_ps(bias);
      dst_w_ma1 = _mm_add_ps(dst_w_ma1, bias_ma);
      dst_w_ma2 = _mm_add_ps(dst_w_ma2, bias_ma);
      dst_w_ma3 = _mm_add_ps(dst_w_ma3, bias_ma);
      dst_w_ma4 = _mm_add_ps(dst_w_ma4, bias_ma);

      ActBlock4(&dst_w_ma1, &dst_w_ma2, &dst_w_ma3, &dst_w_ma4, relu, relu6);

      _mm_storeu_ps(dst_w, dst_w_ma1);
      _mm_storeu_ps(dst_w + block_channel, dst_w_ma2);
      _mm_storeu_ps(dst_w + 2 * block_channel, dst_w_ma3);
      _mm_storeu_ps(dst_w + 3 * block_channel, dst_w_ma4);
    }  // dst_width loop

    // c2 loop
    for (; c1 < c2; c1 += C2NUM, dst_w += C2NUM * block_channel, src_w += C2NUM * in_sw_step) {
      const float *src_kh = src_w, *weight_kh = weight;
      __m128 dst_w_ma1 = _mm_setzero_ps();
      __m128 dst_w_ma2 = _mm_setzero_ps();

      for (int kh = 0; kh < kernel_h; kh++, src_kh += in_kh_step, weight_kh += kernel_w * C4NUM) {
        const float *src_kw = src_kh, *weight_kw = weight_kh;
        for (int kw = 0; kw < kernel_w; kw++, src_kw += in_kw_step, weight_kw += C4NUM) {
          __m128 src_kw_ma1 = _mm_loadu_ps(src_kw);
          __m128 weight_kw_ma1 = _mm_loadu_ps(weight_kw);
          __m128 tmp_ma1 = _mm_mul_ps(src_kw_ma1, weight_kw_ma1);
          dst_w_ma1 = _mm_add_ps(dst_w_ma1, tmp_ma1);

          __m128 src_kw_ma2 = _mm_loadu_ps(src_kw + in_sw_step);
          __m128 weight_kw_ma2 = _mm_loadu_ps(weight_kw);
          __m128 tmp_ma2 = _mm_mul_ps(src_kw_ma2, weight_kw_ma2);
          dst_w_ma2 = _mm_add_ps(dst_w_ma2, tmp_ma2);
        }  // kernel_w loop
      }    // kernel_h loop
      // add bias relu
      __m128 bias_ma = _mm_loadu_ps(bias);
      dst_w_ma1 = _mm_add_ps(dst_w_ma1, bias_ma);
      dst_w_ma2 = _mm_add_ps(dst_w_ma2, bias_ma);

      ActBlock2(&dst_w_ma1, &dst_w_ma2, relu, relu6);

      _mm_storeu_ps(dst_w, dst_w_ma1);
      _mm_storeu_ps(dst_w + block_channel, dst_w_ma2);
    }

    // remaining
    for (; c1 < width; c1++, dst_w += block_channel, src_w += in_sw_step) {
      const float *src_kh = src_w, *weight_kh = weight;
      __m128 dst_w_ma1 = _mm_setzero_ps();
      for (int kh = 0; kh < kernel_h; kh++, src_kh += in_kh_step, weight_kh += kernel_w * C4NUM) {
        const float *src_kw = src_kh, *weight_kw = weight_kh;
        for (int kw = 0; kw < kernel_w; kw++, src_kw += in_kw_step, weight_kw += C4NUM) {
          __m128 src_kw_ma1 = _mm_loadu_ps(src_kw);
          __m128 weight_kw_ma1 = _mm_loadu_ps(weight_kw);
          __m128 tmp_ma1 = _mm_mul_ps(src_kw_ma1, weight_kw_ma1);
          dst_w_ma1 = _mm_add_ps(dst_w_ma1, tmp_ma1);
        }  // kernel_w loop
      }    // kernel_h loop

      // add bias relu
      __m128 bias_ma = _mm_loadu_ps(bias);
      dst_w_ma1 = _mm_add_ps(dst_w_ma1, bias_ma);
      ActBlock1(&dst_w_ma1, relu, relu6);
      _mm_storeu_ps(dst_w, dst_w_ma1);
    }
    dst_h += out_h_step;
    src_h += in_sh_step;
  }  // dst_height loop
}

void DeconvDwFp32Center(float *dst, const float *src, const float *weight, size_t height, size_t width, size_t kernel_h,
                        size_t kernel_w, size_t out_h_step, size_t block_channel, size_t in_sh_step, size_t in_sw_step,
                        size_t in_kh_step, size_t in_kw_step) {
  out_h_step /= sizeof(float);
  block_channel /= sizeof(float);
  in_sh_step /= sizeof(float);
  in_sw_step /= sizeof(float);
  in_kh_step /= sizeof(float);
  in_kw_step /= sizeof(float);

  float *dst_h = dst;
  const float *src_h = src;
  for (int oh = 0; oh < height; oh++) {
    float *dst_w = dst_h;
    const float *src_w = src_h;
    for (int ow = 0; ow < width; ow++) {
      float *dst_kh = dst_w;
      const float *weight_kh = weight;
      __m128 src_w_ma = _mm_loadu_ps(src_w);
      for (int kh = 0; kh < kernel_h; kh++) {
        float *dst_kw = dst_kh;
        const float *weight_kw = weight_kh;

        int c4 = DOWN_DIV(kernel_w, C4NUM) * C4NUM;
        int c2 = DOWN_DIV(kernel_w, C2NUM) * C2NUM;
        int c1 = 0;
        // c4 loop
        for (; c1 < c4; c1 += C4NUM) {
          __m128 dst_w_ma1 = _mm_loadu_ps(dst_kw);
          __m128 weight_kw_ma1 = _mm_loadu_ps(weight_kw);
          __m128 tmp_ma1 = _mm_mul_ps(src_w_ma, weight_kw_ma1);
          dst_w_ma1 = _mm_add_ps(dst_w_ma1, tmp_ma1);
          _mm_storeu_ps(dst_kw, dst_w_ma1);

          __m128 dst_w_ma2 = _mm_loadu_ps(dst_kw + in_kw_step);
          __m128 weight_kw_ma2 = _mm_loadu_ps(weight_kw + C4NUM);
          __m128 tmp_ma2 = _mm_mul_ps(src_w_ma, weight_kw_ma2);
          dst_w_ma2 = _mm_add_ps(dst_w_ma2, tmp_ma2);
          _mm_storeu_ps(dst_kw + in_kw_step, dst_w_ma2);

          __m128 dst_w_ma3 = _mm_loadu_ps(dst_kw + 2 * in_kw_step);
          __m128 weight_kw_ma3 = _mm_loadu_ps(weight_kw + 2 * C4NUM);
          __m128 tmp_ma3 = _mm_mul_ps(src_w_ma, weight_kw_ma3);
          dst_w_ma3 = _mm_add_ps(dst_w_ma3, tmp_ma3);
          _mm_storeu_ps(dst_kw + 2 * in_kw_step, dst_w_ma3);

          __m128 dst_w_ma4 = _mm_loadu_ps(dst_kw + 3 * in_kw_step);
          __m128 weight_kw_ma4 = _mm_loadu_ps(weight_kw + 3 * C4NUM);
          __m128 tmp_ma4 = _mm_mul_ps(src_w_ma, weight_kw_ma4);
          dst_w_ma4 = _mm_add_ps(dst_w_ma4, tmp_ma4);
          _mm_storeu_ps(dst_kw + 3 * in_kw_step, dst_w_ma4);

          dst_kw += 4 * in_kw_step;
          weight_kw += 4 * C4NUM;
        }
        // c2 loop
        for (; c1 < c2; c1 += C2NUM) {
          __m128 dst_w_ma1 = _mm_loadu_ps(dst_kw);
          __m128 weight_kw_ma1 = _mm_loadu_ps(weight_kw);
          __m128 tmp_ma1 = _mm_mul_ps(src_w_ma, weight_kw_ma1);
          dst_w_ma1 = _mm_add_ps(dst_w_ma1, tmp_ma1);
          _mm_storeu_ps(dst_kw, dst_w_ma1);

          __m128 dst_w_ma2 = _mm_loadu_ps(dst_kw + in_kw_step);
          __m128 weight_kw_ma2 = _mm_loadu_ps(weight_kw + C4NUM);
          __m128 tmp_ma2 = _mm_mul_ps(src_w_ma, weight_kw_ma2);
          dst_w_ma2 = _mm_add_ps(dst_w_ma2, tmp_ma2);
          _mm_storeu_ps(dst_kw + in_kw_step, dst_w_ma2);

          dst_kw += 2 * in_kw_step;
          weight_kw += 2 * C4NUM;
        }
        // remaining
        for (; c1 < kernel_w; ++c1) {
          __m128 dst_w_ma1 = _mm_loadu_ps(dst_kw);
          __m128 weight_kw_ma1 = _mm_loadu_ps(weight_kw);
          __m128 tmp_ma1 = _mm_mul_ps(src_w_ma, weight_kw_ma1);
          dst_w_ma1 = _mm_add_ps(dst_w_ma1, tmp_ma1);
          _mm_storeu_ps(dst_kw, dst_w_ma1);

          dst_kw += in_kw_step;
          weight_kw += C4NUM;
        }  // kernel_w loop

        dst_kh += in_kh_step;
        weight_kh += kernel_w * C4NUM;
      }  // kernel_h loop
      dst_w += in_sw_step;
      src_w += block_channel;
    }  // dst_width loop
    dst_h += in_sh_step;
    src_h += out_h_step;
  }  // dst_height loop
}

#endif
