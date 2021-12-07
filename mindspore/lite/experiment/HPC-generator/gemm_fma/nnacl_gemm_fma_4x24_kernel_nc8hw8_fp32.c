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
void nnacl_gemm_fma_4x24_kernel_nc8hw8_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                            const size_t act_flag, const size_t row_block, const size_t col_block,
                                            const size_t deep, const size_t src_stride, const size_t dst_stride,
                                            const size_t inc_flag) {
  __m256 dst0;
  __m256 dst4;
  __m256 dst8;
  __m256 dst1;
  __m256 dst5;
  __m256 dst9;
  __m256 dst2;
  __m256 dst6;
  __m256 dst10;
  __m256 dst3;
  __m256 dst7;
  __m256 dst11;
  if (inc_flag) {
    dst0 = _mm256_load_ps(dst + 0 * dst_stride + 0);
    dst4 = _mm256_load_ps(dst + 1 * dst_stride + 0);
    dst8 = _mm256_load_ps(dst + 2 * dst_stride + 0);
    dst1 = _mm256_load_ps(dst + 0 * dst_stride + 8);
    dst5 = _mm256_load_ps(dst + 1 * dst_stride + 8);
    dst9 = _mm256_load_ps(dst + 2 * dst_stride + 8);
    dst2 = _mm256_load_ps(dst + 0 * dst_stride + 16);
    dst6 = _mm256_load_ps(dst + 1 * dst_stride + 16);
    dst10 = _mm256_load_ps(dst + 2 * dst_stride + 16);
    dst3 = _mm256_load_ps(dst + 0 * dst_stride + 24);
    dst7 = _mm256_load_ps(dst + 1 * dst_stride + 24);
    dst11 = _mm256_load_ps(dst + 2 * dst_stride + 24);
  } else if (bias == NULL) {
    dst0 = _mm256_setzero_ps();
    dst1 = _mm256_setzero_ps();
    dst2 = _mm256_setzero_ps();
    dst3 = _mm256_setzero_ps();
    dst4 = _mm256_setzero_ps();
    dst5 = _mm256_setzero_ps();
    dst6 = _mm256_setzero_ps();
    dst7 = _mm256_setzero_ps();
    dst8 = _mm256_setzero_ps();
    dst9 = _mm256_setzero_ps();
    dst10 = _mm256_setzero_ps();
    dst11 = _mm256_setzero_ps();
  } else {
    dst0 = _mm256_load_ps(bias + 0);
    dst4 = _mm256_load_ps(bias + 8);
    dst8 = _mm256_load_ps(bias + 16);
    dst1 = _mm256_load_ps(bias + 0);
    dst5 = _mm256_load_ps(bias + 8);
    dst9 = _mm256_load_ps(bias + 16);
    dst2 = _mm256_load_ps(bias + 0);
    dst6 = _mm256_load_ps(bias + 8);
    dst10 = _mm256_load_ps(bias + 16);
    dst3 = _mm256_load_ps(bias + 0);
    dst7 = _mm256_load_ps(bias + 8);
    dst11 = _mm256_load_ps(bias + 16);
  }
  for (int i = 0; i < (deep >> 3); ++i) {
    // bock0
    __m256 weight00 = _mm256_load_ps(weight + 0);
    __m256 weight10 = _mm256_load_ps(weight + 8);
    __m256 weight20 = _mm256_load_ps(weight + 16);
    __m256 src00 = _mm256_set1_ps(*(src + 0));
    dst0 = _mm256_fmadd_ps(dst0, src00, weight00);
    dst4 = _mm256_fmadd_ps(dst4, src00, weight10);
    dst8 = _mm256_fmadd_ps(dst8, src00, weight20);
    __m256 src10 = _mm256_set1_ps(*(src + 8));
    dst1 = _mm256_fmadd_ps(dst1, src10, weight00);
    dst5 = _mm256_fmadd_ps(dst5, src10, weight10);
    dst9 = _mm256_fmadd_ps(dst9, src10, weight20);
    __m256 src20 = _mm256_set1_ps(*(src + 16));
    dst2 = _mm256_fmadd_ps(dst2, src20, weight00);
    dst6 = _mm256_fmadd_ps(dst6, src20, weight10);
    dst10 = _mm256_fmadd_ps(dst10, src20, weight20);
    __m256 src30 = _mm256_set1_ps(*(src + 24));
    dst3 = _mm256_fmadd_ps(dst3, src30, weight00);
    dst7 = _mm256_fmadd_ps(dst7, src30, weight10);
    dst11 = _mm256_fmadd_ps(dst11, src30, weight20);
    // bock1
    __m256 weight01 = _mm256_load_ps(weight + 24);
    __m256 weight11 = _mm256_load_ps(weight + 32);
    __m256 weight21 = _mm256_load_ps(weight + 40);
    __m256 src01 = _mm256_set1_ps(*(src + 1));
    dst0 = _mm256_fmadd_ps(dst0, src01, weight01);
    dst4 = _mm256_fmadd_ps(dst4, src01, weight11);
    dst8 = _mm256_fmadd_ps(dst8, src01, weight21);
    __m256 src11 = _mm256_set1_ps(*(src + 9));
    dst1 = _mm256_fmadd_ps(dst1, src11, weight01);
    dst5 = _mm256_fmadd_ps(dst5, src11, weight11);
    dst9 = _mm256_fmadd_ps(dst9, src11, weight21);
    __m256 src21 = _mm256_set1_ps(*(src + 17));
    dst2 = _mm256_fmadd_ps(dst2, src21, weight01);
    dst6 = _mm256_fmadd_ps(dst6, src21, weight11);
    dst10 = _mm256_fmadd_ps(dst10, src21, weight21);
    __m256 src31 = _mm256_set1_ps(*(src + 25));
    dst3 = _mm256_fmadd_ps(dst3, src31, weight01);
    dst7 = _mm256_fmadd_ps(dst7, src31, weight11);
    dst11 = _mm256_fmadd_ps(dst11, src31, weight21);
    // bock2
    __m256 weight02 = _mm256_load_ps(weight + 48);
    __m256 weight12 = _mm256_load_ps(weight + 56);
    __m256 weight22 = _mm256_load_ps(weight + 64);
    __m256 src02 = _mm256_set1_ps(*(src + 2));
    dst0 = _mm256_fmadd_ps(dst0, src02, weight02);
    dst4 = _mm256_fmadd_ps(dst4, src02, weight12);
    dst8 = _mm256_fmadd_ps(dst8, src02, weight22);
    __m256 src12 = _mm256_set1_ps(*(src + 10));
    dst1 = _mm256_fmadd_ps(dst1, src12, weight02);
    dst5 = _mm256_fmadd_ps(dst5, src12, weight12);
    dst9 = _mm256_fmadd_ps(dst9, src12, weight22);
    __m256 src22 = _mm256_set1_ps(*(src + 18));
    dst2 = _mm256_fmadd_ps(dst2, src22, weight02);
    dst6 = _mm256_fmadd_ps(dst6, src22, weight12);
    dst10 = _mm256_fmadd_ps(dst10, src22, weight22);
    __m256 src32 = _mm256_set1_ps(*(src + 26));
    dst3 = _mm256_fmadd_ps(dst3, src32, weight02);
    dst7 = _mm256_fmadd_ps(dst7, src32, weight12);
    dst11 = _mm256_fmadd_ps(dst11, src32, weight22);
    // bock3
    __m256 weight03 = _mm256_load_ps(weight + 72);
    __m256 weight13 = _mm256_load_ps(weight + 80);
    __m256 weight23 = _mm256_load_ps(weight + 88);
    __m256 src03 = _mm256_set1_ps(*(src + 3));
    dst0 = _mm256_fmadd_ps(dst0, src03, weight03);
    dst4 = _mm256_fmadd_ps(dst4, src03, weight13);
    dst8 = _mm256_fmadd_ps(dst8, src03, weight23);
    __m256 src13 = _mm256_set1_ps(*(src + 11));
    dst1 = _mm256_fmadd_ps(dst1, src13, weight03);
    dst5 = _mm256_fmadd_ps(dst5, src13, weight13);
    dst9 = _mm256_fmadd_ps(dst9, src13, weight23);
    __m256 src23 = _mm256_set1_ps(*(src + 19));
    dst2 = _mm256_fmadd_ps(dst2, src23, weight03);
    dst6 = _mm256_fmadd_ps(dst6, src23, weight13);
    dst10 = _mm256_fmadd_ps(dst10, src23, weight23);
    __m256 src33 = _mm256_set1_ps(*(src + 27));
    dst3 = _mm256_fmadd_ps(dst3, src33, weight03);
    dst7 = _mm256_fmadd_ps(dst7, src33, weight13);
    dst11 = _mm256_fmadd_ps(dst11, src33, weight23);
    // bock4
    __m256 weight04 = _mm256_load_ps(weight + 96);
    __m256 weight14 = _mm256_load_ps(weight + 104);
    __m256 weight24 = _mm256_load_ps(weight + 112);
    __m256 src04 = _mm256_set1_ps(*(src + 4));
    dst0 = _mm256_fmadd_ps(dst0, src04, weight04);
    dst4 = _mm256_fmadd_ps(dst4, src04, weight14);
    dst8 = _mm256_fmadd_ps(dst8, src04, weight24);
    __m256 src14 = _mm256_set1_ps(*(src + 12));
    dst1 = _mm256_fmadd_ps(dst1, src14, weight04);
    dst5 = _mm256_fmadd_ps(dst5, src14, weight14);
    dst9 = _mm256_fmadd_ps(dst9, src14, weight24);
    __m256 src24 = _mm256_set1_ps(*(src + 20));
    dst2 = _mm256_fmadd_ps(dst2, src24, weight04);
    dst6 = _mm256_fmadd_ps(dst6, src24, weight14);
    dst10 = _mm256_fmadd_ps(dst10, src24, weight24);
    __m256 src34 = _mm256_set1_ps(*(src + 28));
    dst3 = _mm256_fmadd_ps(dst3, src34, weight04);
    dst7 = _mm256_fmadd_ps(dst7, src34, weight14);
    dst11 = _mm256_fmadd_ps(dst11, src34, weight24);
    // bock5
    __m256 weight05 = _mm256_load_ps(weight + 120);
    __m256 weight15 = _mm256_load_ps(weight + 128);
    __m256 weight25 = _mm256_load_ps(weight + 136);
    __m256 src05 = _mm256_set1_ps(*(src + 5));
    dst0 = _mm256_fmadd_ps(dst0, src05, weight05);
    dst4 = _mm256_fmadd_ps(dst4, src05, weight15);
    dst8 = _mm256_fmadd_ps(dst8, src05, weight25);
    __m256 src15 = _mm256_set1_ps(*(src + 13));
    dst1 = _mm256_fmadd_ps(dst1, src15, weight05);
    dst5 = _mm256_fmadd_ps(dst5, src15, weight15);
    dst9 = _mm256_fmadd_ps(dst9, src15, weight25);
    __m256 src25 = _mm256_set1_ps(*(src + 21));
    dst2 = _mm256_fmadd_ps(dst2, src25, weight05);
    dst6 = _mm256_fmadd_ps(dst6, src25, weight15);
    dst10 = _mm256_fmadd_ps(dst10, src25, weight25);
    __m256 src35 = _mm256_set1_ps(*(src + 29));
    dst3 = _mm256_fmadd_ps(dst3, src35, weight05);
    dst7 = _mm256_fmadd_ps(dst7, src35, weight15);
    dst11 = _mm256_fmadd_ps(dst11, src35, weight25);
    // bock6
    __m256 weight06 = _mm256_load_ps(weight + 144);
    __m256 weight16 = _mm256_load_ps(weight + 152);
    __m256 weight26 = _mm256_load_ps(weight + 160);
    __m256 src06 = _mm256_set1_ps(*(src + 6));
    dst0 = _mm256_fmadd_ps(dst0, src06, weight06);
    dst4 = _mm256_fmadd_ps(dst4, src06, weight16);
    dst8 = _mm256_fmadd_ps(dst8, src06, weight26);
    __m256 src16 = _mm256_set1_ps(*(src + 14));
    dst1 = _mm256_fmadd_ps(dst1, src16, weight06);
    dst5 = _mm256_fmadd_ps(dst5, src16, weight16);
    dst9 = _mm256_fmadd_ps(dst9, src16, weight26);
    __m256 src26 = _mm256_set1_ps(*(src + 22));
    dst2 = _mm256_fmadd_ps(dst2, src26, weight06);
    dst6 = _mm256_fmadd_ps(dst6, src26, weight16);
    dst10 = _mm256_fmadd_ps(dst10, src26, weight26);
    __m256 src36 = _mm256_set1_ps(*(src + 30));
    dst3 = _mm256_fmadd_ps(dst3, src36, weight06);
    dst7 = _mm256_fmadd_ps(dst7, src36, weight16);
    dst11 = _mm256_fmadd_ps(dst11, src36, weight26);
    // bock7
    __m256 weight07 = _mm256_load_ps(weight + 168);
    __m256 weight17 = _mm256_load_ps(weight + 176);
    __m256 weight27 = _mm256_load_ps(weight + 184);
    __m256 src07 = _mm256_set1_ps(*(src + 7));
    dst0 = _mm256_fmadd_ps(dst0, src07, weight07);
    dst4 = _mm256_fmadd_ps(dst4, src07, weight17);
    dst8 = _mm256_fmadd_ps(dst8, src07, weight27);
    __m256 src17 = _mm256_set1_ps(*(src + 15));
    dst1 = _mm256_fmadd_ps(dst1, src17, weight07);
    dst5 = _mm256_fmadd_ps(dst5, src17, weight17);
    dst9 = _mm256_fmadd_ps(dst9, src17, weight27);
    __m256 src27 = _mm256_set1_ps(*(src + 23));
    dst2 = _mm256_fmadd_ps(dst2, src27, weight07);
    dst6 = _mm256_fmadd_ps(dst6, src27, weight17);
    dst10 = _mm256_fmadd_ps(dst10, src27, weight27);
    __m256 src37 = _mm256_set1_ps(*(src + 31));
    dst3 = _mm256_fmadd_ps(dst3, src37, weight07);
    dst7 = _mm256_fmadd_ps(dst7, src37, weight17);
    dst11 = _mm256_fmadd_ps(dst11, src37, weight27);
    src = src + src_stride;
    weight += 768;
  }
  if (act_flag & 0x02) {
    // relu6
    __m256 relu6 = _mm256_set1_ps(6.0f);
    __m256 relu = _mm256_setzero_ps();
    dst0 = _mm256_min_ps(dst0, relu6);
    dst4 = _mm256_min_ps(dst4, relu6);
    dst8 = _mm256_min_ps(dst8, relu6);
    dst1 = _mm256_min_ps(dst1, relu6);
    dst5 = _mm256_min_ps(dst5, relu6);
    dst9 = _mm256_min_ps(dst9, relu6);
    dst2 = _mm256_min_ps(dst2, relu6);
    dst6 = _mm256_min_ps(dst6, relu6);
    dst10 = _mm256_min_ps(dst10, relu6);
    dst3 = _mm256_min_ps(dst3, relu6);
    dst7 = _mm256_min_ps(dst7, relu6);
    dst11 = _mm256_min_ps(dst11, relu6);
    // relu
    dst0 = _mm256_max_ps(dst0, relu);
    dst4 = _mm256_max_ps(dst4, relu);
    dst8 = _mm256_max_ps(dst8, relu);
    dst1 = _mm256_max_ps(dst1, relu);
    dst5 = _mm256_max_ps(dst5, relu);
    dst9 = _mm256_max_ps(dst9, relu);
    dst2 = _mm256_max_ps(dst2, relu);
    dst6 = _mm256_max_ps(dst6, relu);
    dst10 = _mm256_max_ps(dst10, relu);
    dst3 = _mm256_max_ps(dst3, relu);
    dst7 = _mm256_max_ps(dst7, relu);
    dst11 = _mm256_max_ps(dst11, relu);
  }
  if (act_flag & 0x01) {
    // relu
    __m256 relu = _mm256_setzero_ps();
    dst0 = _mm256_max_ps(dst0, relu);
    dst4 = _mm256_max_ps(dst4, relu);
    dst8 = _mm256_max_ps(dst8, relu);
    dst1 = _mm256_max_ps(dst1, relu);
    dst5 = _mm256_max_ps(dst5, relu);
    dst9 = _mm256_max_ps(dst9, relu);
    dst2 = _mm256_max_ps(dst2, relu);
    dst6 = _mm256_max_ps(dst6, relu);
    dst10 = _mm256_max_ps(dst10, relu);
    dst3 = _mm256_max_ps(dst3, relu);
    dst7 = _mm256_max_ps(dst7, relu);
    dst11 = _mm256_max_ps(dst11, relu);
  }
  _mm256_store_ps(dst + 0 * src_stride + 0, dst0);
  _mm256_store_ps(dst + 0 * src_stride + 8, dst1);
  _mm256_store_ps(dst + 0 * src_stride + 16, dst2);
  _mm256_store_ps(dst + 0 * src_stride + 24, dst3);
  _mm256_store_ps(dst + 1 * src_stride + 0, dst4);
  _mm256_store_ps(dst + 1 * src_stride + 8, dst5);
  _mm256_store_ps(dst + 1 * src_stride + 16, dst6);
  _mm256_store_ps(dst + 1 * src_stride + 24, dst7);
  _mm256_store_ps(dst + 2 * src_stride + 0, dst8);
  _mm256_store_ps(dst + 2 * src_stride + 8, dst9);
  _mm256_store_ps(dst + 2 * src_stride + 16, dst10);
  _mm256_store_ps(dst + 2 * src_stride + 24, dst11);
}
