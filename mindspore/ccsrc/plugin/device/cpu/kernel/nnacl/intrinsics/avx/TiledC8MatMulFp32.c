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
#ifdef _MSC_VER
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif
#include "nnacl/fp32/common_func_fp32.h"

void TiledC8MatmulFp32(float *dst, const float *src, const float *weight, size_t cal_num, size_t ic8, size_t oc8) {
  const float *src_tmp = src;
  for (int i = 0; i < oc8; ++i) {
    src = src_tmp;
    register __m256 dst1 asm("ymm0") = _mm256_setzero_ps();
    register __m256 dst2 asm("ymm1") = _mm256_setzero_ps();
    register __m256 dst3 asm("ymm2") = _mm256_setzero_ps();
    register __m256 dst4 asm("ymm3") = _mm256_setzero_ps();
    register __m256 dst5 asm("ymm4") = _mm256_setzero_ps();
    register __m256 dst6 asm("ymm5") = _mm256_setzero_ps();
    register __m256 dst7 asm("ymm6") = _mm256_setzero_ps();
    register __m256 dst8 asm("ymm7") = _mm256_setzero_ps();
    for (size_t ic8_tmp = 0; ic8_tmp < ic8; ++ic8_tmp) {
#ifndef ENABLE_DEBUG
      asm volatile(
        // 1
        "vmovups (%1), %%ymm8\n"

        "vbroadcastss (%0), %%ymm9\n"
        "vbroadcastss 32(%0), %%ymm10\n"
        "vbroadcastss 64(%0), %%ymm11\n"
        "vbroadcastss 96(%0), %%ymm12\n"
        "vbroadcastss 128(%0), %%ymm13\n"
        "vbroadcastss 160(%0), %%ymm14\n"

        "vfmadd231ps %%ymm9, %%ymm8, %%ymm0\n"
        "vfmadd231ps %%ymm10, %%ymm8, %%ymm1\n"
        "vfmadd231ps %%ymm11, %%ymm8, %%ymm2\n"
        "vfmadd231ps %%ymm12, %%ymm8, %%ymm3\n"
        "vfmadd231ps %%ymm13, %%ymm8, %%ymm4\n"
        "vfmadd231ps %%ymm14, %%ymm8, %%ymm5\n"

        "vbroadcastss 192(%0), %%ymm9\n"
        "vbroadcastss 224(%0), %%ymm10\n"
        "vfmadd231ps %%ymm9, %%ymm8, %%ymm6\n"
        "vfmadd231ps %%ymm10, %%ymm8, %%ymm7\n"

        // 2
        "vmovups 32(%1), %%ymm15\n"

        "vbroadcastss 4(%0), %%ymm11\n"
        "vbroadcastss 36(%0), %%ymm12\n"
        "vbroadcastss 68(%0), %%ymm13\n"
        "vbroadcastss 100(%0), %%ymm14\n"
        "vbroadcastss 132(%0), %%ymm9\n"
        "vbroadcastss 164(%0), %%ymm10\n"

        "vfmadd231ps %%ymm11, %%ymm15, %%ymm0\n"
        "vfmadd231ps %%ymm12, %%ymm15, %%ymm1\n"
        "vfmadd231ps %%ymm13, %%ymm15, %%ymm2\n"
        "vfmadd231ps %%ymm14, %%ymm15, %%ymm3\n"
        "vfmadd231ps %%ymm9, %%ymm15, %%ymm4\n"
        "vfmadd231ps %%ymm10, %%ymm15, %%ymm5\n"

        "vbroadcastss 196(%0), %%ymm11\n"
        "vbroadcastss 228(%0), %%ymm12\n"
        "vfmadd231ps %%ymm11, %%ymm15, %%ymm6\n"
        "vfmadd231ps %%ymm12, %%ymm15, %%ymm7\n"

        // 3
        "vmovups 64(%1), %%ymm8\n"

        "vbroadcastss 8(%0), %%ymm13\n"
        "vbroadcastss 40(%0), %%ymm14\n"
        "vbroadcastss 72(%0), %%ymm9\n"
        "vbroadcastss 104(%0), %%ymm10\n"
        "vbroadcastss 136(%0), %%ymm11\n"
        "vbroadcastss 168(%0), %%ymm12\n"

        "vfmadd231ps %%ymm13, %%ymm8, %%ymm0\n"
        "vfmadd231ps %%ymm14, %%ymm8, %%ymm1\n"
        "vfmadd231ps %%ymm9, %%ymm8, %%ymm2\n"
        "vfmadd231ps %%ymm10, %%ymm8, %%ymm3\n"
        "vfmadd231ps %%ymm11, %%ymm8, %%ymm4\n"
        "vfmadd231ps %%ymm12, %%ymm8, %%ymm5\n"

        "vbroadcastss 200(%0), %%ymm13\n"
        "vbroadcastss 232(%0), %%ymm14\n"
        "vfmadd231ps %%ymm13, %%ymm8, %%ymm6\n"
        "vfmadd231ps %%ymm14, %%ymm8, %%ymm7\n"

        // 4
        "vmovups 96(%1), %%ymm15\n"

        "vbroadcastss 12(%0), %%ymm9\n"
        "vbroadcastss 44(%0), %%ymm10\n"
        "vbroadcastss 76(%0), %%ymm11\n"
        "vbroadcastss 108(%0), %%ymm12\n"
        "vbroadcastss 140(%0), %%ymm13\n"
        "vbroadcastss 172(%0), %%ymm14\n"

        "vfmadd231ps %%ymm9, %%ymm15, %%ymm0\n"
        "vfmadd231ps %%ymm10, %%ymm15, %%ymm1\n"
        "vfmadd231ps %%ymm11, %%ymm15, %%ymm2\n"
        "vfmadd231ps %%ymm12, %%ymm15, %%ymm3\n"
        "vfmadd231ps %%ymm13, %%ymm15, %%ymm4\n"
        "vfmadd231ps %%ymm14, %%ymm15, %%ymm5\n"

        "vbroadcastss 204(%0), %%ymm9\n"
        "vbroadcastss 236(%0), %%ymm10\n"
        "vfmadd231ps %%ymm9, %%ymm15, %%ymm6\n"
        "vfmadd231ps %%ymm10, %%ymm15, %%ymm7\n"

        // 5
        "vmovups 128(%1), %%ymm8\n"

        "vbroadcastss 16(%0), %%ymm11\n"
        "vbroadcastss 48(%0), %%ymm12\n"
        "vbroadcastss 80(%0), %%ymm13\n"
        "vbroadcastss 112(%0), %%ymm14\n"
        "vbroadcastss 144(%0), %%ymm9\n"
        "vbroadcastss 176(%0), %%ymm10\n"

        "vfmadd231ps %%ymm11, %%ymm8, %%ymm0\n"
        "vfmadd231ps %%ymm12, %%ymm8, %%ymm1\n"
        "vfmadd231ps %%ymm13, %%ymm8, %%ymm2\n"
        "vfmadd231ps %%ymm14, %%ymm8, %%ymm3\n"
        "vfmadd231ps %%ymm9, %%ymm8, %%ymm4\n"
        "vfmadd231ps %%ymm10, %%ymm8, %%ymm5\n"

        "vbroadcastss 208(%0), %%ymm11\n"
        "vbroadcastss 240(%0), %%ymm12\n"
        "vfmadd231ps %%ymm11, %%ymm8, %%ymm6\n"
        "vfmadd231ps %%ymm12, %%ymm8, %%ymm7\n"

        // 6
        "vmovups 160(%1), %%ymm15\n"

        "vbroadcastss 20(%0), %%ymm13\n"
        "vbroadcastss 52(%0), %%ymm14\n"
        "vbroadcastss 84(%0), %%ymm9\n"
        "vbroadcastss 116(%0), %%ymm10\n"
        "vbroadcastss 148(%0), %%ymm11\n"
        "vbroadcastss 180(%0), %%ymm12\n"

        "vfmadd231ps %%ymm13, %%ymm15, %%ymm0\n"
        "vfmadd231ps %%ymm14, %%ymm15, %%ymm1\n"
        "vfmadd231ps %%ymm9, %%ymm15, %%ymm2\n"
        "vfmadd231ps %%ymm10, %%ymm15, %%ymm3\n"
        "vfmadd231ps %%ymm11, %%ymm15, %%ymm4\n"
        "vfmadd231ps %%ymm12, %%ymm15, %%ymm5\n"

        "vbroadcastss 212(%0), %%ymm13\n"
        "vbroadcastss 244(%0), %%ymm14\n"
        "vfmadd231ps %%ymm13, %%ymm15, %%ymm6\n"
        "vfmadd231ps %%ymm14, %%ymm15, %%ymm7\n"

        // 7
        "vmovups 192(%1), %%ymm8\n"

        "vbroadcastss 24(%0), %%ymm9\n"
        "vbroadcastss 56(%0), %%ymm10\n"
        "vbroadcastss 88(%0), %%ymm11\n"
        "vbroadcastss 120(%0), %%ymm12\n"
        "vbroadcastss 152(%0), %%ymm13\n"
        "vbroadcastss 184(%0), %%ymm14\n"

        "vfmadd231ps %%ymm9, %%ymm8, %%ymm0\n"
        "vfmadd231ps %%ymm10, %%ymm8, %%ymm1\n"
        "vfmadd231ps %%ymm11, %%ymm8, %%ymm2\n"
        "vfmadd231ps %%ymm12, %%ymm8, %%ymm3\n"
        "vfmadd231ps %%ymm13, %%ymm8, %%ymm4\n"
        "vfmadd231ps %%ymm14, %%ymm8, %%ymm5\n"

        "vbroadcastss 216(%0), %%ymm9\n"
        "vbroadcastss 248(%0), %%ymm10\n"
        "vfmadd231ps %%ymm9, %%ymm8, %%ymm6\n"
        "vfmadd231ps %%ymm10, %%ymm8, %%ymm7\n"

        // 8
        "vmovups 224(%1), %%ymm15\n"

        "vbroadcastss 28(%0), %%ymm11\n"
        "vbroadcastss 60(%0), %%ymm12\n"
        "vbroadcastss 92(%0), %%ymm13\n"
        "vbroadcastss 124(%0), %%ymm14\n"
        "vbroadcastss 156(%0), %%ymm9\n"
        "vbroadcastss 188(%0), %%ymm10\n"

        "vfmadd231ps %%ymm11, %%ymm15, %%ymm0\n"
        "vfmadd231ps %%ymm12, %%ymm15, %%ymm1\n"
        "vfmadd231ps %%ymm13, %%ymm15, %%ymm2\n"
        "vfmadd231ps %%ymm14, %%ymm15, %%ymm3\n"
        "vfmadd231ps %%ymm9, %%ymm15, %%ymm4\n"
        "vfmadd231ps %%ymm10, %%ymm15, %%ymm5\n"

        "vbroadcastss 220(%0), %%ymm11\n"
        "vbroadcastss 252(%0), %%ymm12\n"
        "vfmadd231ps %%ymm11, %%ymm15, %%ymm6\n"
        "vfmadd231ps %%ymm12, %%ymm15, %%ymm7\n"
        :
        : "r"(src), "r"(weight)
        : "memory");
#else
      for (int j = 0; j < C8NUM; ++j) {
        __m256 weight_data = _mm256_loadu_ps(weight + j * C8NUM);
        dst1 = _mm256_fmadd_ps(weight_data, _mm256_set1_ps(*(src + j)), dst1);
        dst2 = _mm256_fmadd_ps(weight_data, _mm256_set1_ps(*(src + j + C8NUM)), dst2);
        dst3 = _mm256_fmadd_ps(weight_data, _mm256_set1_ps(*(src + j + C16NUM)), dst3);
        dst4 = _mm256_fmadd_ps(weight_data, _mm256_set1_ps(*(src + j + C24NUM)), dst4);
        dst5 = _mm256_fmadd_ps(weight_data, _mm256_set1_ps(*(src + j + C32NUM)), dst5);
        dst6 = _mm256_fmadd_ps(weight_data, _mm256_set1_ps(*(src + j + C40NUM)), dst6);
        dst7 = _mm256_fmadd_ps(weight_data, _mm256_set1_ps(*(src + j + C48NUM)), dst7);
        dst8 = _mm256_fmadd_ps(weight_data, _mm256_set1_ps(*(src + j + C56NUM)), dst8);
      }
#endif
      src += C64NUM;
      weight += C64NUM;
    }
    _mm256_storeu_ps(dst, dst1);
    _mm256_storeu_ps(dst + C8NUM, dst2);
    _mm256_storeu_ps(dst + C16NUM, dst3);
    _mm256_storeu_ps(dst + C24NUM, dst4);
    _mm256_storeu_ps(dst + C32NUM, dst5);
    _mm256_storeu_ps(dst + C40NUM, dst6);
    _mm256_storeu_ps(dst + C48NUM, dst7);
    _mm256_storeu_ps(dst + C56NUM, dst8);
    dst += cal_num;
  }
}
#endif
