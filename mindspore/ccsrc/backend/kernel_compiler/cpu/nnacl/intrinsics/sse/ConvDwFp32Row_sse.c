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

#if defined(ENABLE_SSE) && !defined(ENABLE_AVX)
#include <x86intrin.h>
#include "nnacl/fp32/common_func_fp32.h"

void ConvDwFp32Row(float *output_ptr, const float *input_ptr, const float *weight_ptr, size_t num_pixels,
                   size_t output_channel, size_t input_step) {
  size_t out_c16 = DOWN_DIV(output_channel, C16NUM) * C16NUM;
  size_t out_c8 = DOWN_DIV(output_channel, C8NUM) * C8NUM;
  size_t out_c4 = DOWN_DIV(output_channel, C4NUM) * C4NUM;
  for (int i = 0; i < num_pixels; i++) {
    const float *weight_tmp = weight_ptr;
    const float *input_tmp = input_ptr;
    size_t out_c = 0;
    for (; out_c < out_c16; out_c += C16NUM) {
      __m128 dst1 = _mm_loadu_ps(output_ptr);
      __m128 dst2 = _mm_loadu_ps(output_ptr + 4);
      __m128 dst3 = _mm_loadu_ps(output_ptr + 8);
      __m128 dst4 = _mm_loadu_ps(output_ptr + 12);
      __m128 w1 = _mm_loadu_ps(weight_tmp);
      __m128 w2 = _mm_loadu_ps(weight_tmp + 4);
      __m128 w3 = _mm_loadu_ps(weight_tmp + 8);
      __m128 w4 = _mm_loadu_ps(weight_tmp + 12);
      __m128 in1 = _mm_loadu_ps(input_tmp);
      __m128 in2 = _mm_loadu_ps(input_tmp + 4);
      __m128 in3 = _mm_loadu_ps(input_tmp + 8);
      __m128 in4 = _mm_loadu_ps(input_tmp + 12);
      dst1 = MS_MLAQ_F32(dst1, w1, in1);
      dst2 = MS_MLAQ_F32(dst2, w2, in2);
      dst3 = MS_MLAQ_F32(dst3, w3, in3);
      dst4 = MS_MLAQ_F32(dst4, w4, in4);
      _mm_storeu_ps(output_ptr, dst1);
      _mm_storeu_ps(output_ptr + 4, dst2);
      _mm_storeu_ps(output_ptr + 8, dst3);
      _mm_storeu_ps(output_ptr + 12, dst4);
      output_ptr += 16;
      input_tmp += 16;
      weight_tmp += 16;
    }
    for (; out_c < out_c8; out_c += C8NUM) {
      __m128 dst1 = _mm_loadu_ps(output_ptr);
      __m128 dst2 = _mm_loadu_ps(output_ptr + 4);
      __m128 w1 = _mm_loadu_ps(weight_tmp);
      __m128 w2 = _mm_loadu_ps(weight_tmp + 4);
      __m128 in1 = _mm_loadu_ps(input_tmp);
      __m128 in2 = _mm_loadu_ps(input_tmp + 4);
      dst1 = MS_MLAQ_F32(dst1, w1, in1);
      dst2 = MS_MLAQ_F32(dst2, w2, in2);
      _mm_storeu_ps(output_ptr, dst1);
      _mm_storeu_ps(output_ptr + 4, dst2);
      output_ptr += 8;
      input_tmp += 8;
      weight_tmp += 8;
    }
    for (; out_c < out_c4; out_c += C4NUM) {
      __m128 dst1 = _mm_loadu_ps(output_ptr);
      __m128 w1 = _mm_loadu_ps(weight_tmp);
      __m128 in1 = _mm_loadu_ps(input_tmp);
      dst1 = MS_MLAQ_F32(dst1, w1, in1);
      _mm_storeu_ps(output_ptr, dst1);
      output_ptr += 4;
      input_tmp += 4;
      weight_tmp += 4;
    }
    for (; out_c < output_channel; out_c++) {
      *output_ptr++ += weight_ptr[out_c] * input_ptr[out_c];
    }
    input_ptr += input_step;
  }
}
#endif
