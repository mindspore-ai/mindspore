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
void nnacl_gemm_fma_3x32_kernel_nc8hw8_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                            const size_t act_flag, const size_t row_block, const size_t col_block,
                                            const size_t deep, const size_t src_stride, const size_t dst_stride,
                                            const size_t inc_flag) {
  __m256 dst0;
  __m256 dst3;
  __m256 dst6;
  __m256 dst9;
  __m256 dst1;
  __m256 dst4;
  __m256 dst7;
  __m256 dst10;
  __m256 dst2;
  __m256 dst5;
  __m256 dst8;
  __m256 dst11;
  if (inc_flag) {
    dst0 = _mm256_load_ps(dst + 0 * dst_stride + 0);
    dst3 = _mm256_load_ps(dst + 1 * dst_stride + 0);
    dst6 = _mm256_load_ps(dst + 2 * dst_stride + 0);
    dst9 = _mm256_load_ps(dst + 3 * dst_stride + 0);
    dst1 = _mm256_load_ps(dst + 0 * dst_stride + 8);
    dst4 = _mm256_load_ps(dst + 1 * dst_stride + 8);
    dst7 = _mm256_load_ps(dst + 2 * dst_stride + 8);
    dst10 = _mm256_load_ps(dst + 3 * dst_stride + 8);
    dst2 = _mm256_load_ps(dst + 0 * dst_stride + 16);
    dst5 = _mm256_load_ps(dst + 1 * dst_stride + 16);
    dst8 = _mm256_load_ps(dst + 2 * dst_stride + 16);
    dst11 = _mm256_load_ps(dst + 3 * dst_stride + 16);
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
    dst3 = _mm256_load_ps(bias + 8);
    dst6 = _mm256_load_ps(bias + 16);
    dst9 = _mm256_load_ps(bias + 24);
    dst1 = _mm256_load_ps(bias + 0);
    dst4 = _mm256_load_ps(bias + 8);
    dst7 = _mm256_load_ps(bias + 16);
    dst10 = _mm256_load_ps(bias + 24);
    dst2 = _mm256_load_ps(bias + 0);
    dst5 = _mm256_load_ps(bias + 8);
    dst8 = _mm256_load_ps(bias + 16);
    dst11 = _mm256_load_ps(bias + 24);
  }
  for (int i = 0; i < (deep >> 3); ++i) {
    // bock0
    __m256 src00 = _mm256_set1_ps(*(src + 0));
    __m256 src10 = _mm256_set1_ps(*(src + 8));
    __m256 src20 = _mm256_set1_ps(*(src + 16));
    __m256 weight00 = _mm256_load_ps(weight + 0);
    dst0 = _mm256_fmadd_ps(dst0, src00, weight00);
    dst1 = _mm256_fmadd_ps(dst1, src10, weight00);
    dst2 = _mm256_fmadd_ps(dst2, src20, weight00);
    __m256 weight10 = _mm256_load_ps(weight + 8);
    dst3 = _mm256_fmadd_ps(dst3, src00, weight10);
    dst4 = _mm256_fmadd_ps(dst4, src10, weight10);
    dst5 = _mm256_fmadd_ps(dst5, src20, weight10);
    __m256 weight20 = _mm256_load_ps(weight + 16);
    dst6 = _mm256_fmadd_ps(dst6, src00, weight20);
    dst7 = _mm256_fmadd_ps(dst7, src10, weight20);
    dst8 = _mm256_fmadd_ps(dst8, src20, weight20);
    __m256 weight30 = _mm256_load_ps(weight + 24);
    dst9 = _mm256_fmadd_ps(dst9, src00, weight30);
    dst10 = _mm256_fmadd_ps(dst10, src10, weight30);
    dst11 = _mm256_fmadd_ps(dst11, src20, weight30);
    // bock1
    __m256 src01 = _mm256_set1_ps(*(src + 1));
    __m256 src11 = _mm256_set1_ps(*(src + 9));
    __m256 src21 = _mm256_set1_ps(*(src + 17));
    __m256 weight01 = _mm256_load_ps(weight + 32);
    dst0 = _mm256_fmadd_ps(dst0, src01, weight01);
    dst1 = _mm256_fmadd_ps(dst1, src11, weight01);
    dst2 = _mm256_fmadd_ps(dst2, src21, weight01);
    __m256 weight11 = _mm256_load_ps(weight + 40);
    dst3 = _mm256_fmadd_ps(dst3, src01, weight11);
    dst4 = _mm256_fmadd_ps(dst4, src11, weight11);
    dst5 = _mm256_fmadd_ps(dst5, src21, weight11);
    __m256 weight21 = _mm256_load_ps(weight + 48);
    dst6 = _mm256_fmadd_ps(dst6, src01, weight21);
    dst7 = _mm256_fmadd_ps(dst7, src11, weight21);
    dst8 = _mm256_fmadd_ps(dst8, src21, weight21);
    __m256 weight31 = _mm256_load_ps(weight + 56);
    dst9 = _mm256_fmadd_ps(dst9, src01, weight31);
    dst10 = _mm256_fmadd_ps(dst10, src11, weight31);
    dst11 = _mm256_fmadd_ps(dst11, src21, weight31);
    // bock2
    __m256 src02 = _mm256_set1_ps(*(src + 2));
    __m256 src12 = _mm256_set1_ps(*(src + 10));
    __m256 src22 = _mm256_set1_ps(*(src + 18));
    __m256 weight02 = _mm256_load_ps(weight + 64);
    dst0 = _mm256_fmadd_ps(dst0, src02, weight02);
    dst1 = _mm256_fmadd_ps(dst1, src12, weight02);
    dst2 = _mm256_fmadd_ps(dst2, src22, weight02);
    __m256 weight12 = _mm256_load_ps(weight + 72);
    dst3 = _mm256_fmadd_ps(dst3, src02, weight12);
    dst4 = _mm256_fmadd_ps(dst4, src12, weight12);
    dst5 = _mm256_fmadd_ps(dst5, src22, weight12);
    __m256 weight22 = _mm256_load_ps(weight + 80);
    dst6 = _mm256_fmadd_ps(dst6, src02, weight22);
    dst7 = _mm256_fmadd_ps(dst7, src12, weight22);
    dst8 = _mm256_fmadd_ps(dst8, src22, weight22);
    __m256 weight32 = _mm256_load_ps(weight + 88);
    dst9 = _mm256_fmadd_ps(dst9, src02, weight32);
    dst10 = _mm256_fmadd_ps(dst10, src12, weight32);
    dst11 = _mm256_fmadd_ps(dst11, src22, weight32);
    // bock3
    __m256 src03 = _mm256_set1_ps(*(src + 3));
    __m256 src13 = _mm256_set1_ps(*(src + 11));
    __m256 src23 = _mm256_set1_ps(*(src + 19));
    __m256 weight03 = _mm256_load_ps(weight + 96);
    dst0 = _mm256_fmadd_ps(dst0, src03, weight03);
    dst1 = _mm256_fmadd_ps(dst1, src13, weight03);
    dst2 = _mm256_fmadd_ps(dst2, src23, weight03);
    __m256 weight13 = _mm256_load_ps(weight + 104);
    dst3 = _mm256_fmadd_ps(dst3, src03, weight13);
    dst4 = _mm256_fmadd_ps(dst4, src13, weight13);
    dst5 = _mm256_fmadd_ps(dst5, src23, weight13);
    __m256 weight23 = _mm256_load_ps(weight + 112);
    dst6 = _mm256_fmadd_ps(dst6, src03, weight23);
    dst7 = _mm256_fmadd_ps(dst7, src13, weight23);
    dst8 = _mm256_fmadd_ps(dst8, src23, weight23);
    __m256 weight33 = _mm256_load_ps(weight + 120);
    dst9 = _mm256_fmadd_ps(dst9, src03, weight33);
    dst10 = _mm256_fmadd_ps(dst10, src13, weight33);
    dst11 = _mm256_fmadd_ps(dst11, src23, weight33);
    // bock4
    __m256 src04 = _mm256_set1_ps(*(src + 4));
    __m256 src14 = _mm256_set1_ps(*(src + 12));
    __m256 src24 = _mm256_set1_ps(*(src + 20));
    __m256 weight04 = _mm256_load_ps(weight + 128);
    dst0 = _mm256_fmadd_ps(dst0, src04, weight04);
    dst1 = _mm256_fmadd_ps(dst1, src14, weight04);
    dst2 = _mm256_fmadd_ps(dst2, src24, weight04);
    __m256 weight14 = _mm256_load_ps(weight + 136);
    dst3 = _mm256_fmadd_ps(dst3, src04, weight14);
    dst4 = _mm256_fmadd_ps(dst4, src14, weight14);
    dst5 = _mm256_fmadd_ps(dst5, src24, weight14);
    __m256 weight24 = _mm256_load_ps(weight + 144);
    dst6 = _mm256_fmadd_ps(dst6, src04, weight24);
    dst7 = _mm256_fmadd_ps(dst7, src14, weight24);
    dst8 = _mm256_fmadd_ps(dst8, src24, weight24);
    __m256 weight34 = _mm256_load_ps(weight + 152);
    dst9 = _mm256_fmadd_ps(dst9, src04, weight34);
    dst10 = _mm256_fmadd_ps(dst10, src14, weight34);
    dst11 = _mm256_fmadd_ps(dst11, src24, weight34);
    // bock5
    __m256 src05 = _mm256_set1_ps(*(src + 5));
    __m256 src15 = _mm256_set1_ps(*(src + 13));
    __m256 src25 = _mm256_set1_ps(*(src + 21));
    __m256 weight05 = _mm256_load_ps(weight + 160);
    dst0 = _mm256_fmadd_ps(dst0, src05, weight05);
    dst1 = _mm256_fmadd_ps(dst1, src15, weight05);
    dst2 = _mm256_fmadd_ps(dst2, src25, weight05);
    __m256 weight15 = _mm256_load_ps(weight + 168);
    dst3 = _mm256_fmadd_ps(dst3, src05, weight15);
    dst4 = _mm256_fmadd_ps(dst4, src15, weight15);
    dst5 = _mm256_fmadd_ps(dst5, src25, weight15);
    __m256 weight25 = _mm256_load_ps(weight + 176);
    dst6 = _mm256_fmadd_ps(dst6, src05, weight25);
    dst7 = _mm256_fmadd_ps(dst7, src15, weight25);
    dst8 = _mm256_fmadd_ps(dst8, src25, weight25);
    __m256 weight35 = _mm256_load_ps(weight + 184);
    dst9 = _mm256_fmadd_ps(dst9, src05, weight35);
    dst10 = _mm256_fmadd_ps(dst10, src15, weight35);
    dst11 = _mm256_fmadd_ps(dst11, src25, weight35);
    // bock6
    __m256 src06 = _mm256_set1_ps(*(src + 6));
    __m256 src16 = _mm256_set1_ps(*(src + 14));
    __m256 src26 = _mm256_set1_ps(*(src + 22));
    __m256 weight06 = _mm256_load_ps(weight + 192);
    dst0 = _mm256_fmadd_ps(dst0, src06, weight06);
    dst1 = _mm256_fmadd_ps(dst1, src16, weight06);
    dst2 = _mm256_fmadd_ps(dst2, src26, weight06);
    __m256 weight16 = _mm256_load_ps(weight + 200);
    dst3 = _mm256_fmadd_ps(dst3, src06, weight16);
    dst4 = _mm256_fmadd_ps(dst4, src16, weight16);
    dst5 = _mm256_fmadd_ps(dst5, src26, weight16);
    __m256 weight26 = _mm256_load_ps(weight + 208);
    dst6 = _mm256_fmadd_ps(dst6, src06, weight26);
    dst7 = _mm256_fmadd_ps(dst7, src16, weight26);
    dst8 = _mm256_fmadd_ps(dst8, src26, weight26);
    __m256 weight36 = _mm256_load_ps(weight + 216);
    dst9 = _mm256_fmadd_ps(dst9, src06, weight36);
    dst10 = _mm256_fmadd_ps(dst10, src16, weight36);
    dst11 = _mm256_fmadd_ps(dst11, src26, weight36);
    // bock7
    __m256 src07 = _mm256_set1_ps(*(src + 7));
    __m256 src17 = _mm256_set1_ps(*(src + 15));
    __m256 src27 = _mm256_set1_ps(*(src + 23));
    __m256 weight07 = _mm256_load_ps(weight + 224);
    dst0 = _mm256_fmadd_ps(dst0, src07, weight07);
    dst1 = _mm256_fmadd_ps(dst1, src17, weight07);
    dst2 = _mm256_fmadd_ps(dst2, src27, weight07);
    __m256 weight17 = _mm256_load_ps(weight + 232);
    dst3 = _mm256_fmadd_ps(dst3, src07, weight17);
    dst4 = _mm256_fmadd_ps(dst4, src17, weight17);
    dst5 = _mm256_fmadd_ps(dst5, src27, weight17);
    __m256 weight27 = _mm256_load_ps(weight + 240);
    dst6 = _mm256_fmadd_ps(dst6, src07, weight27);
    dst7 = _mm256_fmadd_ps(dst7, src17, weight27);
    dst8 = _mm256_fmadd_ps(dst8, src27, weight27);
    __m256 weight37 = _mm256_load_ps(weight + 248);
    dst9 = _mm256_fmadd_ps(dst9, src07, weight37);
    dst10 = _mm256_fmadd_ps(dst10, src17, weight37);
    dst11 = _mm256_fmadd_ps(dst11, src27, weight37);
    src = src + src_stride;
    weight += 1024;
  }
  if (act_flag & 0x02) {
    // relu6
    __m256 relu6 = _mm256_set1_ps(6.0f);
    __m256 relu = _mm256_setzero_ps();
    dst0 = _mm256_min_ps(dst0, relu6);
    dst3 = _mm256_min_ps(dst3, relu6);
    dst6 = _mm256_min_ps(dst6, relu6);
    dst9 = _mm256_min_ps(dst9, relu6);
    dst1 = _mm256_min_ps(dst1, relu6);
    dst4 = _mm256_min_ps(dst4, relu6);
    dst7 = _mm256_min_ps(dst7, relu6);
    dst10 = _mm256_min_ps(dst10, relu6);
    dst2 = _mm256_min_ps(dst2, relu6);
    dst5 = _mm256_min_ps(dst5, relu6);
    dst8 = _mm256_min_ps(dst8, relu6);
    dst11 = _mm256_min_ps(dst11, relu6);
    // relu
    dst0 = _mm256_max_ps(dst0, relu);
    dst3 = _mm256_max_ps(dst3, relu);
    dst6 = _mm256_max_ps(dst6, relu);
    dst9 = _mm256_max_ps(dst9, relu);
    dst1 = _mm256_max_ps(dst1, relu);
    dst4 = _mm256_max_ps(dst4, relu);
    dst7 = _mm256_max_ps(dst7, relu);
    dst10 = _mm256_max_ps(dst10, relu);
    dst2 = _mm256_max_ps(dst2, relu);
    dst5 = _mm256_max_ps(dst5, relu);
    dst8 = _mm256_max_ps(dst8, relu);
    dst11 = _mm256_max_ps(dst11, relu);
  }
  if (act_flag & 0x01) {
    // relu
    __m256 relu = _mm256_setzero_ps();
    dst0 = _mm256_max_ps(dst0, relu);
    dst3 = _mm256_max_ps(dst3, relu);
    dst6 = _mm256_max_ps(dst6, relu);
    dst9 = _mm256_max_ps(dst9, relu);
    dst1 = _mm256_max_ps(dst1, relu);
    dst4 = _mm256_max_ps(dst4, relu);
    dst7 = _mm256_max_ps(dst7, relu);
    dst10 = _mm256_max_ps(dst10, relu);
    dst2 = _mm256_max_ps(dst2, relu);
    dst5 = _mm256_max_ps(dst5, relu);
    dst8 = _mm256_max_ps(dst8, relu);
    dst11 = _mm256_max_ps(dst11, relu);
  }
  _mm256_store_ps(dst + 0 * src_stride + 0, dst0);
  _mm256_store_ps(dst + 0 * src_stride + 8, dst1);
  _mm256_store_ps(dst + 0 * src_stride + 16, dst2);
  _mm256_store_ps(dst + 1 * src_stride + 0, dst3);
  _mm256_store_ps(dst + 1 * src_stride + 8, dst4);
  _mm256_store_ps(dst + 1 * src_stride + 16, dst5);
  _mm256_store_ps(dst + 2 * src_stride + 0, dst6);
  _mm256_store_ps(dst + 2 * src_stride + 8, dst7);
  _mm256_store_ps(dst + 2 * src_stride + 16, dst8);
  _mm256_store_ps(dst + 3 * src_stride + 0, dst9);
  _mm256_store_ps(dst + 3 * src_stride + 8, dst10);
  _mm256_store_ps(dst + 3 * src_stride + 16, dst11);
}
