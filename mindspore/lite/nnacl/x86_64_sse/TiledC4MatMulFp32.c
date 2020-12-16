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
      dst1 = _mm_add_ps(dst1, _mm_mul_ps(weight_data[j], _mm_set_ps1(src1[j])));
      dst2 = _mm_add_ps(dst2, _mm_mul_ps(weight_data[j], _mm_set_ps1(src2[j])));
      dst3 = _mm_add_ps(dst3, _mm_mul_ps(weight_data[j], _mm_set_ps1(src3[j])));
      dst4 = _mm_add_ps(dst4, _mm_mul_ps(weight_data[j], _mm_set_ps1(src4[j])));
    }
    src1 = _mm_loadu_ps(src);
    src2 = _mm_loadu_ps(src + 4);
    src3 = _mm_loadu_ps(src + 8);
    src4 = _mm_loadu_ps(src + 12);
    src += 16;
    __m128 dst5 = _mm_mul_ps(weight_data[0], _mm_set_ps1(src1[0]));
    __m128 dst6 = _mm_mul_ps(weight_data[0], _mm_set_ps1(src2[0]));
    __m128 dst7 = _mm_mul_ps(weight_data[0], _mm_set_ps1(src3[0]));
    __m128 dst8 = _mm_mul_ps(weight_data[0], _mm_set_ps1(src4[0]));
    for (int j = 1; j < 4; ++j) {
      dst5 = _mm_add_ps(dst5, _mm_mul_ps(weight_data[j], _mm_set_ps1(src1[j])));
      dst6 = _mm_add_ps(dst6, _mm_mul_ps(weight_data[j], _mm_set_ps1(src2[j])));
      dst7 = _mm_add_ps(dst7, _mm_mul_ps(weight_data[j], _mm_set_ps1(src3[j])));
      dst8 = _mm_add_ps(dst8, _mm_mul_ps(weight_data[j], _mm_set_ps1(src4[j])));
    }
    if (ic4_tmp != 0) {
      ic4_tmp -= 1;
      src1 = _mm_loadu_ps(src);
      src2 = _mm_loadu_ps(src + 4);
      src3 = _mm_loadu_ps(src + 8);
      src4 = _mm_loadu_ps(src + 12);
      src += 16;
      weight_data[0] = _mm_loadu_ps(weight);
      weight_data[1] = _mm_loadu_ps(weight + 4);
      weight += 8;

      dst1 = _mm_add_ps(dst1, _mm_mul_ps(weight_data[0], _mm_set_ps1(src1[0])));
      dst2 = _mm_add_ps(dst2, _mm_mul_ps(weight_data[0], _mm_set_ps1(src2[0])));
      for (; ic4_tmp != 0; ic4_tmp -= 1) {
        dst3 = _mm_add_ps(dst3, _mm_mul_ps(weight_data[0], _mm_set_ps1(src3[0])));
        dst4 = _mm_add_ps(dst4, _mm_mul_ps(weight_data[0], _mm_set_ps1(src4[0])));

        dst1 = _mm_add_ps(dst1, _mm_mul_ps(weight_data[1], _mm_set_ps1(src1[1])));
        dst2 = _mm_add_ps(dst2, _mm_mul_ps(weight_data[1], _mm_set_ps1(src2[1])));
        weight_data[2] = _mm_loadu_ps(weight);
        weight_data[3] = _mm_loadu_ps(weight + 4);
        weight += 8;
        dst3 = _mm_add_ps(dst3, _mm_mul_ps(weight_data[1], _mm_set_ps1(src3[1])));
        dst4 = _mm_add_ps(dst4, _mm_mul_ps(weight_data[1], _mm_set_ps1(src4[1])));

        dst1 = _mm_add_ps(dst1, _mm_mul_ps(weight_data[2], _mm_set_ps1(src1[2])));
        dst2 = _mm_add_ps(dst2, _mm_mul_ps(weight_data[2], _mm_set_ps1(src2[2])));
        dst3 = _mm_add_ps(dst3, _mm_mul_ps(weight_data[2], _mm_set_ps1(src3[2])));
        dst4 = _mm_add_ps(dst4, _mm_mul_ps(weight_data[2], _mm_set_ps1(src4[2])));

        dst1 = _mm_add_ps(dst1, _mm_mul_ps(weight_data[3], _mm_set_ps1(src1[3])));
        dst2 = _mm_add_ps(dst2, _mm_mul_ps(weight_data[3], _mm_set_ps1(src2[3])));
        src1 = _mm_loadu_ps(src);
        src2 = _mm_loadu_ps(src + 4);
        dst3 = _mm_add_ps(dst3, _mm_mul_ps(weight_data[3], _mm_set_ps1(src3[3])));
        dst4 = _mm_add_ps(dst4, _mm_mul_ps(weight_data[3], _mm_set_ps1(src4[3])));
        src3 = _mm_loadu_ps(src + 8);
        src4 = _mm_loadu_ps(src + 12);
        src += 16;

        dst5 = _mm_add_ps(dst5, _mm_mul_ps(weight_data[0], _mm_set_ps1(src1[0])));
        dst6 = _mm_add_ps(dst6, _mm_mul_ps(weight_data[0], _mm_set_ps1(src2[0])));
        dst7 = _mm_add_ps(dst7, _mm_mul_ps(weight_data[0], _mm_set_ps1(src3[0])));
        dst8 = _mm_add_ps(dst8, _mm_mul_ps(weight_data[0], _mm_set_ps1(src4[0])));

        dst5 = _mm_add_ps(dst5, _mm_mul_ps(weight_data[1], _mm_set_ps1(src1[1])));
        dst6 = _mm_add_ps(dst6, _mm_mul_ps(weight_data[1], _mm_set_ps1(src2[1])));
        dst7 = _mm_add_ps(dst7, _mm_mul_ps(weight_data[1], _mm_set_ps1(src3[1])));
        dst8 = _mm_add_ps(dst8, _mm_mul_ps(weight_data[1], _mm_set_ps1(src4[1])));

        dst5 = _mm_add_ps(dst5, _mm_mul_ps(weight_data[2], _mm_set_ps1(src1[2])));
        dst6 = _mm_add_ps(dst6, _mm_mul_ps(weight_data[2], _mm_set_ps1(src2[2])));
        dst7 = _mm_add_ps(dst7, _mm_mul_ps(weight_data[2], _mm_set_ps1(src3[2])));
        weight_data[0] = _mm_loadu_ps(weight);
        weight_data[1] = _mm_loadu_ps(weight + 4);
        weight += 8;
        dst8 = _mm_add_ps(dst8, _mm_mul_ps(weight_data[2], _mm_set_ps1(src4[2])));

        dst5 = _mm_add_ps(dst5, _mm_mul_ps(weight_data[3], _mm_set_ps1(src1[3])));
        dst6 = _mm_add_ps(dst6, _mm_mul_ps(weight_data[3], _mm_set_ps1(src2[3])));
        dst7 = _mm_add_ps(dst7, _mm_mul_ps(weight_data[3], _mm_set_ps1(src3[3])));
        src1 = _mm_loadu_ps(src);
        src2 = _mm_loadu_ps(src + 4);
        dst8 = _mm_add_ps(dst8, _mm_mul_ps(weight_data[3], _mm_set_ps1(src4[3])));
        src3 = _mm_loadu_ps(src + 8);
        src4 = _mm_loadu_ps(src + 12);
        src += 16;

        dst1 = _mm_add_ps(dst1, _mm_mul_ps(weight_data[0], _mm_set_ps1(src1[0])));
        dst2 = _mm_add_ps(dst2, _mm_mul_ps(weight_data[0], _mm_set_ps1(src2[0])));
      }
      dst3 = _mm_add_ps(dst3, _mm_mul_ps(weight_data[0], _mm_set_ps1(src3[0])));
      dst4 = _mm_add_ps(dst4, _mm_mul_ps(weight_data[0], _mm_set_ps1(src4[0])));

      dst1 = _mm_add_ps(dst1, _mm_mul_ps(weight_data[1], _mm_set_ps1(src1[1])));
      dst2 = _mm_add_ps(dst2, _mm_mul_ps(weight_data[1], _mm_set_ps1(src2[1])));
      weight_data[2] = _mm_loadu_ps(weight);
      weight_data[3] = _mm_loadu_ps(weight + 4);
      weight += 8;
      dst3 = _mm_add_ps(dst3, _mm_mul_ps(weight_data[1], _mm_set_ps1(src3[1])));
      dst4 = _mm_add_ps(dst4, _mm_mul_ps(weight_data[1], _mm_set_ps1(src4[1])));

      dst1 = _mm_add_ps(dst1, _mm_mul_ps(weight_data[2], _mm_set_ps1(src1[2])));
      dst2 = _mm_add_ps(dst2, _mm_mul_ps(weight_data[2], _mm_set_ps1(src2[2])));
      dst3 = _mm_add_ps(dst3, _mm_mul_ps(weight_data[2], _mm_set_ps1(src3[2])));
      dst4 = _mm_add_ps(dst4, _mm_mul_ps(weight_data[2], _mm_set_ps1(src4[2])));

      dst1 = _mm_add_ps(dst1, _mm_mul_ps(weight_data[3], _mm_set_ps1(src1[3])));
      dst2 = _mm_add_ps(dst2, _mm_mul_ps(weight_data[3], _mm_set_ps1(src2[3])));
      dst3 = _mm_add_ps(dst3, _mm_mul_ps(weight_data[3], _mm_set_ps1(src3[3])));
      src1 = _mm_loadu_ps(src);
      src2 = _mm_loadu_ps(src + 4);
      dst4 = _mm_add_ps(dst4, _mm_mul_ps(weight_data[3], _mm_set_ps1(src4[3])));
      src3 = _mm_loadu_ps(src + 8);
      src4 = _mm_loadu_ps(src + 12);
      src += 16;
      for (int j = 0; j < 4; ++j) {
        dst5 = _mm_add_ps(dst5, _mm_mul_ps(weight_data[j], _mm_set_ps1(src1[j])));
        dst6 = _mm_add_ps(dst6, _mm_mul_ps(weight_data[j], _mm_set_ps1(src2[j])));
        dst7 = _mm_add_ps(dst7, _mm_mul_ps(weight_data[j], _mm_set_ps1(src3[j])));
        dst8 = _mm_add_ps(dst8, _mm_mul_ps(weight_data[j], _mm_set_ps1(src4[j])));
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
