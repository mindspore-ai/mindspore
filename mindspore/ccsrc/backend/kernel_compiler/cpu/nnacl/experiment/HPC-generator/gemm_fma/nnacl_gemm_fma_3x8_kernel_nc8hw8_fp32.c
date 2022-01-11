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
void nnacl_gemm_fma_3x8_kernel_nc8hw8_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                           const size_t act_flag, const size_t row_block, const size_t col_block,
                                           const size_t deep, const size_t src_stride, const size_t dst_stride,
                                           const size_t inc_flag) {
  __m256 dst0;
  __m256 dst1;
  __m256 dst2;
  if (inc_flag) {
    dst0 = _mm256_load_ps(dst + 0 * dst_stride + 0);
    dst1 = _mm256_load_ps(dst + 0 * dst_stride + 8);
    dst2 = _mm256_load_ps(dst + 0 * dst_stride + 16);
  } else if (bias == NULL) {
    dst0 = _mm256_setzero_ps();
    dst1 = _mm256_setzero_ps();
    dst2 = _mm256_setzero_ps();
  } else {
    dst0 = _mm256_load_ps(bias + 0);
    dst1 = _mm256_load_ps(bias + 0);
    dst2 = _mm256_load_ps(bias + 0);
  }
  for (int i = 0; i < (deep >> 3); ++i) {
    // bock0
    __m256 weight00 = _mm256_load_ps(weight + 0);
    __m256 src00 = _mm256_set1_ps(*(src + 0));
    dst0 = _mm256_fmadd_ps(dst0, src00, weight00);
    __m256 src10 = _mm256_set1_ps(*(src + 8));
    dst1 = _mm256_fmadd_ps(dst1, src10, weight00);
    __m256 src20 = _mm256_set1_ps(*(src + 16));
    dst2 = _mm256_fmadd_ps(dst2, src20, weight00);
    // bock1
    __m256 weight01 = _mm256_load_ps(weight + 8);
    __m256 src01 = _mm256_set1_ps(*(src + 1));
    dst0 = _mm256_fmadd_ps(dst0, src01, weight01);
    __m256 src11 = _mm256_set1_ps(*(src + 9));
    dst1 = _mm256_fmadd_ps(dst1, src11, weight01);
    __m256 src21 = _mm256_set1_ps(*(src + 17));
    dst2 = _mm256_fmadd_ps(dst2, src21, weight01);
    // bock2
    __m256 weight02 = _mm256_load_ps(weight + 16);
    __m256 src02 = _mm256_set1_ps(*(src + 2));
    dst0 = _mm256_fmadd_ps(dst0, src02, weight02);
    __m256 src12 = _mm256_set1_ps(*(src + 10));
    dst1 = _mm256_fmadd_ps(dst1, src12, weight02);
    __m256 src22 = _mm256_set1_ps(*(src + 18));
    dst2 = _mm256_fmadd_ps(dst2, src22, weight02);
    // bock3
    __m256 weight03 = _mm256_load_ps(weight + 24);
    __m256 src03 = _mm256_set1_ps(*(src + 3));
    dst0 = _mm256_fmadd_ps(dst0, src03, weight03);
    __m256 src13 = _mm256_set1_ps(*(src + 11));
    dst1 = _mm256_fmadd_ps(dst1, src13, weight03);
    __m256 src23 = _mm256_set1_ps(*(src + 19));
    dst2 = _mm256_fmadd_ps(dst2, src23, weight03);
    // bock4
    __m256 weight04 = _mm256_load_ps(weight + 32);
    __m256 src04 = _mm256_set1_ps(*(src + 4));
    dst0 = _mm256_fmadd_ps(dst0, src04, weight04);
    __m256 src14 = _mm256_set1_ps(*(src + 12));
    dst1 = _mm256_fmadd_ps(dst1, src14, weight04);
    __m256 src24 = _mm256_set1_ps(*(src + 20));
    dst2 = _mm256_fmadd_ps(dst2, src24, weight04);
    // bock5
    __m256 weight05 = _mm256_load_ps(weight + 40);
    __m256 src05 = _mm256_set1_ps(*(src + 5));
    dst0 = _mm256_fmadd_ps(dst0, src05, weight05);
    __m256 src15 = _mm256_set1_ps(*(src + 13));
    dst1 = _mm256_fmadd_ps(dst1, src15, weight05);
    __m256 src25 = _mm256_set1_ps(*(src + 21));
    dst2 = _mm256_fmadd_ps(dst2, src25, weight05);
    // bock6
    __m256 weight06 = _mm256_load_ps(weight + 48);
    __m256 src06 = _mm256_set1_ps(*(src + 6));
    dst0 = _mm256_fmadd_ps(dst0, src06, weight06);
    __m256 src16 = _mm256_set1_ps(*(src + 14));
    dst1 = _mm256_fmadd_ps(dst1, src16, weight06);
    __m256 src26 = _mm256_set1_ps(*(src + 22));
    dst2 = _mm256_fmadd_ps(dst2, src26, weight06);
    // bock7
    __m256 weight07 = _mm256_load_ps(weight + 56);
    __m256 src07 = _mm256_set1_ps(*(src + 7));
    dst0 = _mm256_fmadd_ps(dst0, src07, weight07);
    __m256 src17 = _mm256_set1_ps(*(src + 15));
    dst1 = _mm256_fmadd_ps(dst1, src17, weight07);
    __m256 src27 = _mm256_set1_ps(*(src + 23));
    dst2 = _mm256_fmadd_ps(dst2, src27, weight07);
    src = src + src_stride;
    weight += 256;
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
  _mm256_store_ps(dst + 0 * src_stride + 8, dst1);
  _mm256_store_ps(dst + 0 * src_stride + 16, dst2);
}
