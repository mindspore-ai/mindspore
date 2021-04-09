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

#include <x86intrin.h>
#include "nnacl/fp32/conv_depthwise_fp32.h"

void ConvDwFp32Avx5x5(float *output, float **input, const float *weights, const float *bias, size_t channels,
                      size_t output_width, size_t input_stride, size_t relu, size_t relu6) {
  input_stride /= sizeof(float *);
  size_t c8 = UP_DIV(channels, C8NUM) * C8NUM;
  size_t c8_mod = channels % C8NUM;
  const int kernel = 25;
  for (int i = 0; i < output_width; ++i) {
    float *in[kernel];
    for (int k = 0; k < kernel; k++) {
      in[k] = input[k];
    }
    input += input_stride;
    size_t c = c8;
    const float *w = weights;
    const float *bias1 = bias;
    for (; c >= C8NUM; c -= C8NUM) {
      __m256 out1 = _mm256_loadu_ps(bias1);
      bias1 += 8;
      for (int k = 0; k < kernel; k += 5) {
        __m256 in1 = _mm256_loadu_ps(in[k]);
        __m256 w1 = _mm256_loadu_ps(w);
        __m256 in2 = _mm256_loadu_ps(in[k + 1]);
        __m256 w2 = _mm256_loadu_ps(w + 8);
        out1 = _mm256_fmadd_ps(in1, w1, out1);
        __m256 in3 = _mm256_loadu_ps(in[k + 2]);
        __m256 w3 = _mm256_loadu_ps(w + 16);
        out1 = _mm256_fmadd_ps(in2, w2, out1);
        __m256 in4 = _mm256_loadu_ps(in[k + 3]);
        __m256 w4 = _mm256_loadu_ps(w + 24);
        out1 = _mm256_fmadd_ps(in3, w3, out1);
        __m256 in5 = _mm256_loadu_ps(in[k + 4]);
        __m256 w5 = _mm256_loadu_ps(w + 32);
        out1 = _mm256_fmadd_ps(in4, w4, out1);
        w += 40;
        in[k] += C8NUM;
        in[k + 1] += C8NUM;
        in[k + 2] += C8NUM;
        in[k + 3] += C8NUM;
        in[k + 4] += C8NUM;
        out1 = _mm256_fmadd_ps(in5, w5, out1);
      }
      if (relu6 != 0) {
        __m256 relu6_data = _mm256_set1_ps(6.0);
        out1 = _mm256_min_ps(out1, relu6_data);
      }
      if (relu != 0 || relu6 != 0) {
        __m256 zero = _mm256_setzero_ps();
        out1 = _mm256_max_ps(out1, zero);
      }
      if (c > C8NUM || c8_mod == 0) {
        _mm256_storeu_ps(output, out1);
        output += C8NUM;
      } else {
        __m128 tmp;
        switch (c8_mod) {
          case 1:
            _mm_store_ss(output, _mm256_castps256_ps128(out1));
            break;
          case 2:
            _mm_storel_pi((__m64 *)output, _mm256_castps256_ps128(out1));
            break;
          case 3:
            tmp = _mm256_castps256_ps128(out1);
            _mm_storel_pi((__m64 *)output, tmp);
            tmp = _mm_unpackhi_ps(tmp, tmp);
            _mm_store_ss(output + 2, tmp);
            break;
          case 4:
            _mm_storeu_ps(output, _mm256_castps256_ps128(out1));
            break;
          case 5:
            _mm_storeu_ps(output, _mm256_castps256_ps128(out1));
            _mm_store_ss(output + 4, _mm256_extractf128_ps(out1, 1));
            break;
          case 6:
            _mm_storeu_ps(output, _mm256_castps256_ps128(out1));
            _mm_storel_pi((__m64 *)(output + 4), _mm256_extractf128_ps(out1, 1));
            break;
          case 7:
            _mm_storeu_ps(output, _mm256_castps256_ps128(out1));
            tmp = _mm256_extractf128_ps(out1, 1);
            _mm_storel_pi((__m64 *)(output + 4), tmp);
            tmp = _mm_unpackhi_ps(tmp, tmp);
            _mm_store_ss(output + 6, tmp);
            break;
          default:
            _mm256_storeu_ps(output, out1);
            break;
        }
        output += c8_mod;
      }
    }
  }
}
#endif
