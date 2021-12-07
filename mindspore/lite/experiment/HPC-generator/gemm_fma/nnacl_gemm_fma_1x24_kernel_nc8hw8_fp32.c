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
#include <x86intrin.h>

// nnacl gemm in x86 fma intrinsic code
void nnacl_gemm_fma_1x24_kernel_nc8hw8_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                            const size_t act_flag, const size_t row_block, const size_t col_block,
                                            const size_t deep, const size_t src_stride, const size_t dst_stride,
                                            const size_t inc_flag) {
  __m256 dst0;
  __m256 dst1;
  __m256 dst2;
  if (inc_flag) {
    dst0 = _mm256_load_ps(dst + 0 * dst_stride + 0);
    dst1 = _mm256_load_ps(dst + 1 * dst_stride + 0);
    dst2 = _mm256_load_ps(dst + 2 * dst_stride + 0);
  } else if (bias == NULL) {
    dst0 = _mm256_setzero_ps();
    dst1 = _mm256_setzero_ps();
    dst2 = _mm256_setzero_ps();
  } else {
    dst0 = _mm256_load_ps(bias + 0);
    dst1 = _mm256_load_ps(bias + 8);
    dst2 = _mm256_load_ps(bias + 16);
  }
  for (int i = 0; i < (deep >> 3); ++i) {
    // bock0
    __m256 weight00 = _mm256_load_ps(weight + 0);
    __m256 weight10 = _mm256_load_ps(weight + 8);
    __m256 weight20 = _mm256_load_ps(weight + 16);
    __m256 src00 = _mm256_set1_ps(*(src + 0));
    dst0 = _mm256_fmadd_ps(dst0, src00, weight00);
    dst1 = _mm256_fmadd_ps(dst1, src00, weight10);
    dst2 = _mm256_fmadd_ps(dst2, src00, weight20);
    // bock1
    __m256 weight01 = _mm256_load_ps(weight + 24);
    __m256 weight11 = _mm256_load_ps(weight + 32);
    __m256 weight21 = _mm256_load_ps(weight + 40);
    __m256 src01 = _mm256_set1_ps(*(src + 1));
    dst0 = _mm256_fmadd_ps(dst0, src01, weight01);
    dst1 = _mm256_fmadd_ps(dst1, src01, weight11);
    dst2 = _mm256_fmadd_ps(dst2, src01, weight21);
    // bock2
    __m256 weight02 = _mm256_load_ps(weight + 48);
    __m256 weight12 = _mm256_load_ps(weight + 56);
    __m256 weight22 = _mm256_load_ps(weight + 64);
    __m256 src02 = _mm256_set1_ps(*(src + 2));
    dst0 = _mm256_fmadd_ps(dst0, src02, weight02);
    dst1 = _mm256_fmadd_ps(dst1, src02, weight12);
    dst2 = _mm256_fmadd_ps(dst2, src02, weight22);
    // bock3
    __m256 weight03 = _mm256_load_ps(weight + 72);
    __m256 weight13 = _mm256_load_ps(weight + 80);
    __m256 weight23 = _mm256_load_ps(weight + 88);
    __m256 src03 = _mm256_set1_ps(*(src + 3));
    dst0 = _mm256_fmadd_ps(dst0, src03, weight03);
    dst1 = _mm256_fmadd_ps(dst1, src03, weight13);
    dst2 = _mm256_fmadd_ps(dst2, src03, weight23);
    // bock4
    __m256 weight04 = _mm256_load_ps(weight + 96);
    __m256 weight14 = _mm256_load_ps(weight + 104);
    __m256 weight24 = _mm256_load_ps(weight + 112);
    __m256 src04 = _mm256_set1_ps(*(src + 4));
    dst0 = _mm256_fmadd_ps(dst0, src04, weight04);
    dst1 = _mm256_fmadd_ps(dst1, src04, weight14);
    dst2 = _mm256_fmadd_ps(dst2, src04, weight24);
    // bock5
    __m256 weight05 = _mm256_load_ps(weight + 120);
    __m256 weight15 = _mm256_load_ps(weight + 128);
    __m256 weight25 = _mm256_load_ps(weight + 136);
    __m256 src05 = _mm256_set1_ps(*(src + 5));
    dst0 = _mm256_fmadd_ps(dst0, src05, weight05);
    dst1 = _mm256_fmadd_ps(dst1, src05, weight15);
    dst2 = _mm256_fmadd_ps(dst2, src05, weight25);
    // bock6
    __m256 weight06 = _mm256_load_ps(weight + 144);
    __m256 weight16 = _mm256_load_ps(weight + 152);
    __m256 weight26 = _mm256_load_ps(weight + 160);
    __m256 src06 = _mm256_set1_ps(*(src + 6));
    dst0 = _mm256_fmadd_ps(dst0, src06, weight06);
    dst1 = _mm256_fmadd_ps(dst1, src06, weight16);
    dst2 = _mm256_fmadd_ps(dst2, src06, weight26);
    // bock7
    __m256 weight07 = _mm256_load_ps(weight + 168);
    __m256 weight17 = _mm256_load_ps(weight + 176);
    __m256 weight27 = _mm256_load_ps(weight + 184);
    __m256 src07 = _mm256_set1_ps(*(src + 7));
    dst0 = _mm256_fmadd_ps(dst0, src07, weight07);
    dst1 = _mm256_fmadd_ps(dst1, src07, weight17);
    dst2 = _mm256_fmadd_ps(dst2, src07, weight27);
    src = src + src_stride;
    weight += 768;
  }
  if (act_flag & 0x02) {
    // relu6
    __m256 relu6 = _mm256_set1_ps(6.0f);
    __m256 relu = _mm256_setzero_ps();
    dst0 = _mm256_min_ps(dst0, relu6);
    dst1 = _mm256_min_ps(dst1, relu6);
    dst2 = _mm256_min_ps(dst2, relu6);
    // relu
    dst0 = _mm256_max_ps(dst0, relu);
    dst1 = _mm256_max_ps(dst1, relu);
    dst2 = _mm256_max_ps(dst2, relu);
  }
  if (act_flag & 0x01) {
    // relu
    __m256 relu = _mm256_setzero_ps();
    dst0 = _mm256_max_ps(dst0, relu);
    dst1 = _mm256_max_ps(dst1, relu);
    dst2 = _mm256_max_ps(dst2, relu);
  }
  _mm256_store_ps(dst + 0 * src_stride + 0, dst0);
  _mm256_store_ps(dst + 1 * src_stride + 0, dst1);
  _mm256_store_ps(dst + 2 * src_stride + 0, dst2);
}
