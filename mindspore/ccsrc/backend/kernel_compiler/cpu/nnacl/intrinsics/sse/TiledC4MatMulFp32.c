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

void TiledC4MatmulFp32_Transfer(__m128 *dst1, __m128 *dst2, __m128 *dst3, __m128 *dst4, const __m128 weight,
                                const float v1, const float v2, const float v3, const float v4) {
  *dst1 = _mm_add_ps(*dst1, _mm_mul_ps(weight, _mm_set_ps1(v1)));
  *dst2 = _mm_add_ps(*dst2, _mm_mul_ps(weight, _mm_set_ps1(v2)));
  *dst3 = _mm_add_ps(*dst3, _mm_mul_ps(weight, _mm_set_ps1(v3)));
  *dst4 = _mm_add_ps(*dst4, _mm_mul_ps(weight, _mm_set_ps1(v4)));
}

void TiledC4MatmulFp32_LoadData(__m128 *src1, __m128 *src2, __m128 *src3, __m128 *src4, const float *src) {
  *src1 = _mm_loadu_ps(src);
  *src2 = _mm_loadu_ps(src + 4);
  *src3 = _mm_loadu_ps(src + 8);
  *src4 = _mm_loadu_ps(src + 12);
}
void TiledC4MatmulFp32(float *dst, const float *src, const float *weight, size_t cal_num, size_t ic4, size_t oc4) {
  const float *src_tmp = src;
  for (int i = 0; i < oc4; ++i) {
    float *dst_tmp = dst;
    src = src_tmp;
    size_t ic4_tmp = ic4 - 1;
    __m128 src1 = _mm_loadu_ps(src);
    __m128 src2 = _mm_loadu_ps(src + 4);
    __m128 src3 = _mm_loadu_ps(src + 8);
    __m128 src4 = _mm_loadu_ps(src + 12);
    src += 16;
    __m128 weight_data[4];
    weight_data[0] = _mm_loadu_ps(weight);
    weight_data[1] = _mm_loadu_ps(weight + 4);
    weight_data[2] = _mm_loadu_ps(weight + 8);
    weight_data[3] = _mm_loadu_ps(weight + 12);
    weight += 16;
    __m128 dst1 = _mm_mul_ps(weight_data[0], _mm_set_ps1(src1[0]));
    __m128 dst2 = _mm_mul_ps(weight_data[0], _mm_set_ps1(src2[0]));
    __m128 dst3 = _mm_mul_ps(weight_data[0], _mm_set_ps1(src3[0]));
    __m128 dst4 = _mm_mul_ps(weight_data[0], _mm_set_ps1(src4[0]));
    for (int j = 1; j < 4; ++j) {
      TiledC4MatmulFp32_Transfer(&dst1, &dst2, &dst3, &dst4, weight_data[j], src1[j], src2[j], src3[j], src4[j]);
    }
    TiledC4MatmulFp32_LoadData(&src1, &src2, &src3, &src4, src);
    src += 16;
    __m128 dst5 = _mm_mul_ps(weight_data[0], _mm_set_ps1(src1[0]));
    __m128 dst6 = _mm_mul_ps(weight_data[0], _mm_set_ps1(src2[0]));
    __m128 dst7 = _mm_mul_ps(weight_data[0], _mm_set_ps1(src3[0]));
    __m128 dst8 = _mm_mul_ps(weight_data[0], _mm_set_ps1(src4[0]));
    for (int j = 1; j < 4; ++j) {
      TiledC4MatmulFp32_Transfer(&dst5, &dst6, &dst7, &dst8, weight_data[j], src1[j], src2[j], src3[j], src4[j]);
    }
    if (ic4_tmp != 0) {
      ic4_tmp -= 1;
      TiledC4MatmulFp32_LoadData(&src1, &src2, &src3, &src4, src);
      src += 16;
      weight_data[0] = _mm_loadu_ps(weight);
      weight_data[1] = _mm_loadu_ps(weight + 4);
      weight += 8;

      dst1 = _mm_add_ps(dst1, _mm_mul_ps(weight_data[0], _mm_set_ps1(src1[0])));
      dst2 = _mm_add_ps(dst2, _mm_mul_ps(weight_data[0], _mm_set_ps1(src2[0])));
      for (; ic4_tmp != 0; ic4_tmp -= 1) {
        dst3 = _mm_add_ps(dst3, _mm_mul_ps(weight_data[0], _mm_set_ps1(src3[0])));
        dst4 = _mm_add_ps(dst4, _mm_mul_ps(weight_data[0], _mm_set_ps1(src4[0])));

        TiledC4MatmulFp32_Transfer(&dst1, &dst2, &dst3, &dst4, weight_data[1], src1[1], src2[1], src3[1], src4[1]);

        weight_data[2] = _mm_loadu_ps(weight);
        weight_data[3] = _mm_loadu_ps(weight + 4);
        weight += 8;

        TiledC4MatmulFp32_Transfer(&dst1, &dst2, &dst3, &dst4, weight_data[2], src1[2], src2[2], src3[2], src4[2]);

        dst1 = _mm_add_ps(dst1, _mm_mul_ps(weight_data[3], _mm_set_ps1(src1[3])));
        dst2 = _mm_add_ps(dst2, _mm_mul_ps(weight_data[3], _mm_set_ps1(src2[3])));
        src1 = _mm_loadu_ps(src);
        src2 = _mm_loadu_ps(src + 4);
        dst3 = _mm_add_ps(dst3, _mm_mul_ps(weight_data[3], _mm_set_ps1(src3[3])));
        dst4 = _mm_add_ps(dst4, _mm_mul_ps(weight_data[3], _mm_set_ps1(src4[3])));
        src3 = _mm_loadu_ps(src + 8);
        src4 = _mm_loadu_ps(src + 12);
        src += 16;

        TiledC4MatmulFp32_Transfer(&dst5, &dst6, &dst7, &dst8, weight_data[0], src1[0], src2[0], src3[0], src4[0]);

        TiledC4MatmulFp32_Transfer(&dst5, &dst6, &dst7, &dst8, weight_data[1], src1[1], src2[1], src3[1], src4[1]);

        TiledC4MatmulFp32_Transfer(&dst5, &dst6, &dst7, &dst8, weight_data[2], src1[2], src2[2], src3[2], src4[2]);

        weight_data[0] = _mm_loadu_ps(weight);
        weight_data[1] = _mm_loadu_ps(weight + 4);
        weight += 8;

        TiledC4MatmulFp32_Transfer(&dst5, &dst6, &dst7, &dst8, weight_data[3], src1[3], src2[3], src3[3], src4[3]);
        TiledC4MatmulFp32_LoadData(&src1, &src2, &src3, &src4, src);
        src += 16;

        dst1 = _mm_add_ps(dst1, _mm_mul_ps(weight_data[0], _mm_set_ps1(src1[0])));
        dst2 = _mm_add_ps(dst2, _mm_mul_ps(weight_data[0], _mm_set_ps1(src2[0])));
      }
      dst3 = _mm_add_ps(dst3, _mm_mul_ps(weight_data[0], _mm_set_ps1(src3[0])));
      dst4 = _mm_add_ps(dst4, _mm_mul_ps(weight_data[0], _mm_set_ps1(src4[0])));

      TiledC4MatmulFp32_Transfer(&dst1, &dst2, &dst3, &dst4, weight_data[1], src1[1], src2[1], src3[1], src4[1]);

      weight_data[2] = _mm_loadu_ps(weight);
      weight_data[3] = _mm_loadu_ps(weight + 4);
      weight += 8;

      TiledC4MatmulFp32_Transfer(&dst1, &dst2, &dst3, &dst4, weight_data[2], src1[2], src2[2], src3[2], src4[2]);

      TiledC4MatmulFp32_Transfer(&dst1, &dst2, &dst3, &dst4, weight_data[3], src1[3], src2[3], src3[3], src4[3]);

      TiledC4MatmulFp32_LoadData(&src1, &src2, &src3, &src4, src);
      src += 16;
      for (int j = 0; j < 4; ++j) {
        TiledC4MatmulFp32_Transfer(&dst5, &dst6, &dst7, &dst8, weight_data[j], src1[j], src2[j], src3[j], src4[j]);
      }
    }
    _mm_storeu_ps(dst, dst1);
    _mm_storeu_ps(dst + 4, dst2);
    _mm_storeu_ps(dst + 8, dst3);
    _mm_storeu_ps(dst + 12, dst4);
    _mm_storeu_ps(dst + 16, dst5);
    _mm_storeu_ps(dst + 20, dst6);
    _mm_storeu_ps(dst + 24, dst7);
    _mm_storeu_ps(dst + 28, dst8);
    dst = dst_tmp + cal_num;
  }
}
#endif
