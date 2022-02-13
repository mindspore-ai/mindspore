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
void nnacl_gemm_fma_12x8_kernel_nc8hw8_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                            const size_t act_flag, const size_t row_block, const size_t col_block,
                                            const size_t deep, const size_t src_stride, const size_t dst_stride,
                                            const size_t inc_flag) {
  __m256 dst0;
  __m256 dst1;
  __m256 dst2;
  __m256 dst3;
  __m256 dst4;
  __m256 dst5;
  __m256 dst6;
  __m256 dst7;
  __m256 dst8;
  __m256 dst9;
  __m256 dst10;
  __m256 dst11;
  if (inc_flag) {
    dst0 = _mm256_load_ps(dst + 0 * dst_stride + 0);
    dst1 = _mm256_load_ps(dst + 0 * dst_stride + 8);
    dst2 = _mm256_load_ps(dst + 0 * dst_stride + 16);
    dst3 = _mm256_load_ps(dst + 0 * dst_stride + 24);
    dst4 = _mm256_load_ps(dst + 0 * dst_stride + 32);
    dst5 = _mm256_load_ps(dst + 0 * dst_stride + 40);
    dst6 = _mm256_load_ps(dst + 0 * dst_stride + 48);
    dst7 = _mm256_load_ps(dst + 0 * dst_stride + 56);
    dst8 = _mm256_load_ps(dst + 0 * dst_stride + 64);
    dst9 = _mm256_load_ps(dst + 0 * dst_stride + 72);
    dst10 = _mm256_load_ps(dst + 0 * dst_stride + 80);
    dst11 = _mm256_load_ps(dst + 0 * dst_stride + 88);
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
    dst1 = _mm256_load_ps(bias + 0);
    dst2 = _mm256_load_ps(bias + 0);
    dst3 = _mm256_load_ps(bias + 0);
    dst4 = _mm256_load_ps(bias + 0);
    dst5 = _mm256_load_ps(bias + 0);
    dst6 = _mm256_load_ps(bias + 0);
    dst7 = _mm256_load_ps(bias + 0);
    dst8 = _mm256_load_ps(bias + 0);
    dst9 = _mm256_load_ps(bias + 0);
    dst10 = _mm256_load_ps(bias + 0);
    dst11 = _mm256_load_ps(bias + 0);
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
    __m256 src30 = _mm256_set1_ps(*(src + 24));
    dst3 = _mm256_fmadd_ps(dst3, src30, weight00);
    __m256 src40 = _mm256_set1_ps(*(src + 32));
    dst4 = _mm256_fmadd_ps(dst4, src40, weight00);
    __m256 src50 = _mm256_set1_ps(*(src + 40));
    dst5 = _mm256_fmadd_ps(dst5, src50, weight00);
    __m256 src60 = _mm256_set1_ps(*(src + 48));
    dst6 = _mm256_fmadd_ps(dst6, src60, weight00);
    __m256 src70 = _mm256_set1_ps(*(src + 56));
    dst7 = _mm256_fmadd_ps(dst7, src70, weight00);
    __m256 src80 = _mm256_set1_ps(*(src + 64));
    dst8 = _mm256_fmadd_ps(dst8, src80, weight00);
    __m256 src90 = _mm256_set1_ps(*(src + 72));
    dst9 = _mm256_fmadd_ps(dst9, src90, weight00);
    __m256 src100 = _mm256_set1_ps(*(src + 80));
    dst10 = _mm256_fmadd_ps(dst10, src100, weight00);
    __m256 src110 = _mm256_set1_ps(*(src + 88));
    dst11 = _mm256_fmadd_ps(dst11, src110, weight00);
    // bock1
    __m256 weight01 = _mm256_load_ps(weight + 8);
    __m256 src01 = _mm256_set1_ps(*(src + 1));
    dst0 = _mm256_fmadd_ps(dst0, src01, weight01);
    __m256 src11 = _mm256_set1_ps(*(src + 9));
    dst1 = _mm256_fmadd_ps(dst1, src11, weight01);
    __m256 src21 = _mm256_set1_ps(*(src + 17));
    dst2 = _mm256_fmadd_ps(dst2, src21, weight01);
    __m256 src31 = _mm256_set1_ps(*(src + 25));
    dst3 = _mm256_fmadd_ps(dst3, src31, weight01);
    __m256 src41 = _mm256_set1_ps(*(src + 33));
    dst4 = _mm256_fmadd_ps(dst4, src41, weight01);
    __m256 src51 = _mm256_set1_ps(*(src + 41));
    dst5 = _mm256_fmadd_ps(dst5, src51, weight01);
    __m256 src61 = _mm256_set1_ps(*(src + 49));
    dst6 = _mm256_fmadd_ps(dst6, src61, weight01);
    __m256 src71 = _mm256_set1_ps(*(src + 57));
    dst7 = _mm256_fmadd_ps(dst7, src71, weight01);
    __m256 src81 = _mm256_set1_ps(*(src + 65));
    dst8 = _mm256_fmadd_ps(dst8, src81, weight01);
    __m256 src91 = _mm256_set1_ps(*(src + 73));
    dst9 = _mm256_fmadd_ps(dst9, src91, weight01);
    __m256 src101 = _mm256_set1_ps(*(src + 81));
    dst10 = _mm256_fmadd_ps(dst10, src101, weight01);
    __m256 src111 = _mm256_set1_ps(*(src + 89));
    dst11 = _mm256_fmadd_ps(dst11, src111, weight01);
    // bock2
    __m256 weight02 = _mm256_load_ps(weight + 16);
    __m256 src02 = _mm256_set1_ps(*(src + 2));
    dst0 = _mm256_fmadd_ps(dst0, src02, weight02);
    __m256 src12 = _mm256_set1_ps(*(src + 10));
    dst1 = _mm256_fmadd_ps(dst1, src12, weight02);
    __m256 src22 = _mm256_set1_ps(*(src + 18));
    dst2 = _mm256_fmadd_ps(dst2, src22, weight02);
    __m256 src32 = _mm256_set1_ps(*(src + 26));
    dst3 = _mm256_fmadd_ps(dst3, src32, weight02);
    __m256 src42 = _mm256_set1_ps(*(src + 34));
    dst4 = _mm256_fmadd_ps(dst4, src42, weight02);
    __m256 src52 = _mm256_set1_ps(*(src + 42));
    dst5 = _mm256_fmadd_ps(dst5, src52, weight02);
    __m256 src62 = _mm256_set1_ps(*(src + 50));
    dst6 = _mm256_fmadd_ps(dst6, src62, weight02);
    __m256 src72 = _mm256_set1_ps(*(src + 58));
    dst7 = _mm256_fmadd_ps(dst7, src72, weight02);
    __m256 src82 = _mm256_set1_ps(*(src + 66));
    dst8 = _mm256_fmadd_ps(dst8, src82, weight02);
    __m256 src92 = _mm256_set1_ps(*(src + 74));
    dst9 = _mm256_fmadd_ps(dst9, src92, weight02);
    __m256 src102 = _mm256_set1_ps(*(src + 82));
    dst10 = _mm256_fmadd_ps(dst10, src102, weight02);
    __m256 src112 = _mm256_set1_ps(*(src + 90));
    dst11 = _mm256_fmadd_ps(dst11, src112, weight02);
    // bock3
    __m256 weight03 = _mm256_load_ps(weight + 24);
    __m256 src03 = _mm256_set1_ps(*(src + 3));
    dst0 = _mm256_fmadd_ps(dst0, src03, weight03);
    __m256 src13 = _mm256_set1_ps(*(src + 11));
    dst1 = _mm256_fmadd_ps(dst1, src13, weight03);
    __m256 src23 = _mm256_set1_ps(*(src + 19));
    dst2 = _mm256_fmadd_ps(dst2, src23, weight03);
    __m256 src33 = _mm256_set1_ps(*(src + 27));
    dst3 = _mm256_fmadd_ps(dst3, src33, weight03);
    __m256 src43 = _mm256_set1_ps(*(src + 35));
    dst4 = _mm256_fmadd_ps(dst4, src43, weight03);
    __m256 src53 = _mm256_set1_ps(*(src + 43));
    dst5 = _mm256_fmadd_ps(dst5, src53, weight03);
    __m256 src63 = _mm256_set1_ps(*(src + 51));
    dst6 = _mm256_fmadd_ps(dst6, src63, weight03);
    __m256 src73 = _mm256_set1_ps(*(src + 59));
    dst7 = _mm256_fmadd_ps(dst7, src73, weight03);
    __m256 src83 = _mm256_set1_ps(*(src + 67));
    dst8 = _mm256_fmadd_ps(dst8, src83, weight03);
    __m256 src93 = _mm256_set1_ps(*(src + 75));
    dst9 = _mm256_fmadd_ps(dst9, src93, weight03);
    __m256 src103 = _mm256_set1_ps(*(src + 83));
    dst10 = _mm256_fmadd_ps(dst10, src103, weight03);
    __m256 src113 = _mm256_set1_ps(*(src + 91));
    dst11 = _mm256_fmadd_ps(dst11, src113, weight03);
    // bock4
    __m256 weight04 = _mm256_load_ps(weight + 32);
    __m256 src04 = _mm256_set1_ps(*(src + 4));
    dst0 = _mm256_fmadd_ps(dst0, src04, weight04);
    __m256 src14 = _mm256_set1_ps(*(src + 12));
    dst1 = _mm256_fmadd_ps(dst1, src14, weight04);
    __m256 src24 = _mm256_set1_ps(*(src + 20));
    dst2 = _mm256_fmadd_ps(dst2, src24, weight04);
    __m256 src34 = _mm256_set1_ps(*(src + 28));
    dst3 = _mm256_fmadd_ps(dst3, src34, weight04);
    __m256 src44 = _mm256_set1_ps(*(src + 36));
    dst4 = _mm256_fmadd_ps(dst4, src44, weight04);
    __m256 src54 = _mm256_set1_ps(*(src + 44));
    dst5 = _mm256_fmadd_ps(dst5, src54, weight04);
    __m256 src64 = _mm256_set1_ps(*(src + 52));
    dst6 = _mm256_fmadd_ps(dst6, src64, weight04);
    __m256 src74 = _mm256_set1_ps(*(src + 60));
    dst7 = _mm256_fmadd_ps(dst7, src74, weight04);
    __m256 src84 = _mm256_set1_ps(*(src + 68));
    dst8 = _mm256_fmadd_ps(dst8, src84, weight04);
    __m256 src94 = _mm256_set1_ps(*(src + 76));
    dst9 = _mm256_fmadd_ps(dst9, src94, weight04);
    __m256 src104 = _mm256_set1_ps(*(src + 84));
    dst10 = _mm256_fmadd_ps(dst10, src104, weight04);
    __m256 src114 = _mm256_set1_ps(*(src + 92));
    dst11 = _mm256_fmadd_ps(dst11, src114, weight04);
    // bock5
    __m256 weight05 = _mm256_load_ps(weight + 40);
    __m256 src05 = _mm256_set1_ps(*(src + 5));
    dst0 = _mm256_fmadd_ps(dst0, src05, weight05);
    __m256 src15 = _mm256_set1_ps(*(src + 13));
    dst1 = _mm256_fmadd_ps(dst1, src15, weight05);
    __m256 src25 = _mm256_set1_ps(*(src + 21));
    dst2 = _mm256_fmadd_ps(dst2, src25, weight05);
    __m256 src35 = _mm256_set1_ps(*(src + 29));
    dst3 = _mm256_fmadd_ps(dst3, src35, weight05);
    __m256 src45 = _mm256_set1_ps(*(src + 37));
    dst4 = _mm256_fmadd_ps(dst4, src45, weight05);
    __m256 src55 = _mm256_set1_ps(*(src + 45));
    dst5 = _mm256_fmadd_ps(dst5, src55, weight05);
    __m256 src65 = _mm256_set1_ps(*(src + 53));
    dst6 = _mm256_fmadd_ps(dst6, src65, weight05);
    __m256 src75 = _mm256_set1_ps(*(src + 61));
    dst7 = _mm256_fmadd_ps(dst7, src75, weight05);
    __m256 src85 = _mm256_set1_ps(*(src + 69));
    dst8 = _mm256_fmadd_ps(dst8, src85, weight05);
    __m256 src95 = _mm256_set1_ps(*(src + 77));
    dst9 = _mm256_fmadd_ps(dst9, src95, weight05);
    __m256 src105 = _mm256_set1_ps(*(src + 85));
    dst10 = _mm256_fmadd_ps(dst10, src105, weight05);
    __m256 src115 = _mm256_set1_ps(*(src + 93));
    dst11 = _mm256_fmadd_ps(dst11, src115, weight05);
    // bock6
    __m256 weight06 = _mm256_load_ps(weight + 48);
    __m256 src06 = _mm256_set1_ps(*(src + 6));
    dst0 = _mm256_fmadd_ps(dst0, src06, weight06);
    __m256 src16 = _mm256_set1_ps(*(src + 14));
    dst1 = _mm256_fmadd_ps(dst1, src16, weight06);
    __m256 src26 = _mm256_set1_ps(*(src + 22));
    dst2 = _mm256_fmadd_ps(dst2, src26, weight06);
    __m256 src36 = _mm256_set1_ps(*(src + 30));
    dst3 = _mm256_fmadd_ps(dst3, src36, weight06);
    __m256 src46 = _mm256_set1_ps(*(src + 38));
    dst4 = _mm256_fmadd_ps(dst4, src46, weight06);
    __m256 src56 = _mm256_set1_ps(*(src + 46));
    dst5 = _mm256_fmadd_ps(dst5, src56, weight06);
    __m256 src66 = _mm256_set1_ps(*(src + 54));
    dst6 = _mm256_fmadd_ps(dst6, src66, weight06);
    __m256 src76 = _mm256_set1_ps(*(src + 62));
    dst7 = _mm256_fmadd_ps(dst7, src76, weight06);
    __m256 src86 = _mm256_set1_ps(*(src + 70));
    dst8 = _mm256_fmadd_ps(dst8, src86, weight06);
    __m256 src96 = _mm256_set1_ps(*(src + 78));
    dst9 = _mm256_fmadd_ps(dst9, src96, weight06);
    __m256 src106 = _mm256_set1_ps(*(src + 86));
    dst10 = _mm256_fmadd_ps(dst10, src106, weight06);
    __m256 src116 = _mm256_set1_ps(*(src + 94));
    dst11 = _mm256_fmadd_ps(dst11, src116, weight06);
    // bock7
    __m256 weight07 = _mm256_load_ps(weight + 56);
    __m256 src07 = _mm256_set1_ps(*(src + 7));
    dst0 = _mm256_fmadd_ps(dst0, src07, weight07);
    __m256 src17 = _mm256_set1_ps(*(src + 15));
    dst1 = _mm256_fmadd_ps(dst1, src17, weight07);
    __m256 src27 = _mm256_set1_ps(*(src + 23));
    dst2 = _mm256_fmadd_ps(dst2, src27, weight07);
    __m256 src37 = _mm256_set1_ps(*(src + 31));
    dst3 = _mm256_fmadd_ps(dst3, src37, weight07);
    __m256 src47 = _mm256_set1_ps(*(src + 39));
    dst4 = _mm256_fmadd_ps(dst4, src47, weight07);
    __m256 src57 = _mm256_set1_ps(*(src + 47));
    dst5 = _mm256_fmadd_ps(dst5, src57, weight07);
    __m256 src67 = _mm256_set1_ps(*(src + 55));
    dst6 = _mm256_fmadd_ps(dst6, src67, weight07);
    __m256 src77 = _mm256_set1_ps(*(src + 63));
    dst7 = _mm256_fmadd_ps(dst7, src77, weight07);
    __m256 src87 = _mm256_set1_ps(*(src + 71));
    dst8 = _mm256_fmadd_ps(dst8, src87, weight07);
    __m256 src97 = _mm256_set1_ps(*(src + 79));
    dst9 = _mm256_fmadd_ps(dst9, src97, weight07);
    __m256 src107 = _mm256_set1_ps(*(src + 87));
    dst10 = _mm256_fmadd_ps(dst10, src107, weight07);
    __m256 src117 = _mm256_set1_ps(*(src + 95));
    dst11 = _mm256_fmadd_ps(dst11, src117, weight07);
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
    dst3 = _mm256_min_ps(dst3, relu6);
    dst4 = _mm256_min_ps(dst4, relu6);
    dst5 = _mm256_min_ps(dst5, relu6);
    dst6 = _mm256_min_ps(dst6, relu6);
    dst7 = _mm256_min_ps(dst7, relu6);
    dst8 = _mm256_min_ps(dst8, relu6);
    dst9 = _mm256_min_ps(dst9, relu6);
    dst10 = _mm256_min_ps(dst10, relu6);
    dst11 = _mm256_min_ps(dst11, relu6);
    // relu
    dst0 = _mm256_max_ps(dst0, relu);
    dst1 = _mm256_max_ps(dst1, relu);
    dst2 = _mm256_max_ps(dst2, relu);
    dst3 = _mm256_max_ps(dst3, relu);
    dst4 = _mm256_max_ps(dst4, relu);
    dst5 = _mm256_max_ps(dst5, relu);
    dst6 = _mm256_max_ps(dst6, relu);
    dst7 = _mm256_max_ps(dst7, relu);
    dst8 = _mm256_max_ps(dst8, relu);
    dst9 = _mm256_max_ps(dst9, relu);
    dst10 = _mm256_max_ps(dst10, relu);
    dst11 = _mm256_max_ps(dst11, relu);
  }
  if (act_flag & 0x01) {
    // relu
    __m256 relu = _mm256_setzero_ps();
    dst0 = _mm256_max_ps(dst0, relu);
    dst1 = _mm256_max_ps(dst1, relu);
    dst2 = _mm256_max_ps(dst2, relu);
    dst3 = _mm256_max_ps(dst3, relu);
    dst4 = _mm256_max_ps(dst4, relu);
    dst5 = _mm256_max_ps(dst5, relu);
    dst6 = _mm256_max_ps(dst6, relu);
    dst7 = _mm256_max_ps(dst7, relu);
    dst8 = _mm256_max_ps(dst8, relu);
    dst9 = _mm256_max_ps(dst9, relu);
    dst10 = _mm256_max_ps(dst10, relu);
    dst11 = _mm256_max_ps(dst11, relu);
  }
  _mm256_store_ps(dst + 0 * src_stride + 0, dst0);
  _mm256_store_ps(dst + 0 * src_stride + 8, dst1);
  _mm256_store_ps(dst + 0 * src_stride + 16, dst2);
  _mm256_store_ps(dst + 0 * src_stride + 24, dst3);
  _mm256_store_ps(dst + 0 * src_stride + 32, dst4);
  _mm256_store_ps(dst + 0 * src_stride + 40, dst5);
  _mm256_store_ps(dst + 0 * src_stride + 48, dst6);
  _mm256_store_ps(dst + 0 * src_stride + 56, dst7);
  _mm256_store_ps(dst + 0 * src_stride + 64, dst8);
  _mm256_store_ps(dst + 0 * src_stride + 72, dst9);
  _mm256_store_ps(dst + 0 * src_stride + 80, dst10);
  _mm256_store_ps(dst + 0 * src_stride + 88, dst11);
}
