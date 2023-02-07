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

#ifndef MINDSPORE_LITE_NNACL_INTRINSICS_SSE_SSE_COMMON_H_
#define MINDSPORE_LITE_NNACL_INTRINSICS_SSE_SSE_COMMON_H_

#define SSE_ROW_NUM_1 1
#define SSE_ROW_NUM_2 2
#define SSE_ROW_NUM_3 3

#define SSE_INDEX_1 1
#define SSE_INDEX_2 2
#define SSE_INDEX_3 3
#define SSE_INDEX_4 4
#define SSE_INDEX_5 5
#define SSE_INDEX_6 6

#define SSE_SHUFFLE_0321 (_MM_SHUFFLE(0, 3, 2, 1))

#define SSE_ACT_RELU 1
#define SSE_ACT_RELU6 3

static inline void ActBlock1(__m128 *v1, size_t relu, size_t relu6) {
  __m128 zero_ma = _mm_setzero_ps();
  if (relu || relu6) {
    *v1 = _mm_max_ps(zero_ma, *v1);
  }
  if (relu6) {
    __m128 relu6_ma = _mm_set_ps(6.0f, 6.0f, 6.0f, 6.0f);
    *v1 = _mm_min_ps(relu6_ma, *v1);
  }
}

static inline void ActBlock2(__m128 *v1, __m128 *v2, size_t relu, size_t relu6) {
  __m128 zero_ma = _mm_setzero_ps();
  if (relu || relu6) {
    *v1 = _mm_max_ps(zero_ma, *v1);
    *v2 = _mm_max_ps(zero_ma, *v2);
  }
  if (relu6) {
    __m128 relu6_ma = _mm_set_ps(6.0f, 6.0f, 6.0f, 6.0f);
    *v1 = _mm_min_ps(relu6_ma, *v1);
    *v2 = _mm_min_ps(relu6_ma, *v2);
  }
}

static inline void ActBlock4(__m128 *v1, __m128 *v2, __m128 *v3, __m128 *v4, size_t relu, size_t relu6) {
  __m128 zero_ma = _mm_setzero_ps();
  if (relu || relu6) {
    *v1 = _mm_max_ps(zero_ma, *v1);
    *v2 = _mm_max_ps(zero_ma, *v2);
    *v3 = _mm_max_ps(zero_ma, *v3);
    *v4 = _mm_max_ps(zero_ma, *v4);
  }
  if (relu6) {
    __m128 relu6_ma = _mm_set_ps(6.0f, 6.0f, 6.0f, 6.0f);
    *v1 = _mm_min_ps(relu6_ma, *v1);
    *v2 = _mm_min_ps(relu6_ma, *v2);
    *v3 = _mm_min_ps(relu6_ma, *v3);
    *v4 = _mm_min_ps(relu6_ma, *v4);
  }
}

static inline void ActBlock12(__m128 *v1, __m128 *v2, __m128 *v3, __m128 *v4, __m128 *v5, __m128 *v6, __m128 *v7,
                              __m128 *v8, __m128 *v9, __m128 *v10, __m128 *v11, __m128 *v12, size_t relu,
                              size_t relu6) {
  if (relu || relu6) {
    __m128 zero_ma = _mm_setzero_ps();
    *v1 = _mm_max_ps(zero_ma, *v1);
    *v2 = _mm_max_ps(zero_ma, *v2);
    *v3 = _mm_max_ps(zero_ma, *v3);
    *v4 = _mm_max_ps(zero_ma, *v4);
    *v5 = _mm_max_ps(zero_ma, *v5);
    *v6 = _mm_max_ps(zero_ma, *v6);
    *v7 = _mm_max_ps(zero_ma, *v7);
    *v8 = _mm_max_ps(zero_ma, *v8);
    *v9 = _mm_max_ps(zero_ma, *v9);
    *v10 = _mm_max_ps(zero_ma, *v10);
    *v11 = _mm_max_ps(zero_ma, *v11);
    *v12 = _mm_max_ps(zero_ma, *v12);
  }
  if (relu6) {
    __m128 relu6_ma = _mm_set_ps(6.0f, 6.0f, 6.0f, 6.0f);
    *v1 = _mm_min_ps(relu6_ma, *v1);
    *v2 = _mm_min_ps(relu6_ma, *v2);
    *v3 = _mm_min_ps(relu6_ma, *v3);
    *v4 = _mm_min_ps(relu6_ma, *v4);
    *v5 = _mm_min_ps(relu6_ma, *v5);
    *v6 = _mm_min_ps(relu6_ma, *v6);
    *v7 = _mm_min_ps(relu6_ma, *v7);
    *v8 = _mm_min_ps(relu6_ma, *v8);
    *v9 = _mm_min_ps(relu6_ma, *v9);
    *v10 = _mm_min_ps(relu6_ma, *v10);
    *v11 = _mm_min_ps(relu6_ma, *v11);
    *v12 = _mm_min_ps(relu6_ma, *v12);
  }
}

static inline void ActBlock8(__m128 *v1, __m128 *v2, __m128 *v3, __m128 *v4, __m128 *v5, __m128 *v6, __m128 *v7,
                             __m128 *v8, size_t relu_type) {
  __m128 relu6 = _mm_set_ps1(6.0);
  __m128 zero = _mm_setzero_ps();
  switch (relu_type) {
    case SSE_ACT_RELU6:
      *v1 = _mm_min_ps(*v1, relu6);
      *v2 = _mm_min_ps(*v2, relu6);
      *v3 = _mm_min_ps(*v3, relu6);
      *v4 = _mm_min_ps(*v4, relu6);
      *v5 = _mm_min_ps(*v5, relu6);
      *v6 = _mm_min_ps(*v6, relu6);
      *v7 = _mm_min_ps(*v7, relu6);
      *v8 = _mm_min_ps(*v8, relu6);
    case SSE_ACT_RELU:
      *v1 = _mm_max_ps(*v1, zero);
      *v2 = _mm_max_ps(*v2, zero);
      *v3 = _mm_max_ps(*v3, zero);
      *v4 = _mm_max_ps(*v4, zero);
      *v5 = _mm_max_ps(*v5, zero);
      *v6 = _mm_max_ps(*v6, zero);
      *v7 = _mm_max_ps(*v7, zero);
      *v8 = _mm_max_ps(*v8, zero);
    default:
      break;
  }
}

static inline void WriteCol1(float **dst, __m128 *dst1, __m128 *dst2, __m128 *dst3, __m128 *dst4, __m128 *dst5,
                             __m128 *dst6, __m128 *dst7, __m128 *dst8, int stride, int extra_stride, int r) {
  _mm_store_ss(*dst, *dst1);
  if (r > SSE_ROW_NUM_1) {
    *dst += stride;
    _mm_store_ss(*dst, *dst3);
  }
  if (r > SSE_ROW_NUM_2) {
    *dst += stride;
    _mm_store_ss(*dst, *dst5);
  }
  if (r > SSE_ROW_NUM_3) {
    *dst += stride;
    _mm_store_ss(*dst, *dst7);
    *dst += stride;
    *dst += extra_stride;
  }
}

static inline void WriteCol2(float **dst, __m128 *dst1, __m128 *dst2, __m128 *dst3, __m128 *dst4, __m128 *dst5,
                             __m128 *dst6, __m128 *dst7, __m128 *dst8, int stride, int r) {
  _mm_store_ss(*dst, *dst1);
  *dst1 = _mm_shuffle_ps(*dst1, *dst1, SSE_SHUFFLE_0321);
  _mm_store_ss(*dst, *dst1);
  if (r > SSE_ROW_NUM_1) {
    *dst += stride;
    _mm_store_ss(*dst, *dst3);
    *dst3 = _mm_shuffle_ps(*dst3, *dst3, SSE_SHUFFLE_0321);
    _mm_store_ss(*dst, *dst3);
  }
  if (r > SSE_ROW_NUM_2) {
    *dst += stride;
    _mm_store_ss(*dst, *dst5);
    *dst5 = _mm_shuffle_ps(*dst5, *dst5, SSE_SHUFFLE_0321);
    _mm_store_ss(*dst, *dst5);
  }
  if (r > SSE_ROW_NUM_3) {
    *dst += stride;
    _mm_store_ss(*dst, *dst7);
    *dst7 = _mm_shuffle_ps(*dst7, *dst7, SSE_SHUFFLE_0321);
    _mm_store_ss(*dst, *dst7);
  }
}

static inline void WriteCol2Opt(float **dst, __m128 *dst1, __m128 *dst2, __m128 *dst3, __m128 *dst4, __m128 *dst5,
                                __m128 *dst6, __m128 *dst7, __m128 *dst8, int stride, int r) {
  _mm_store_ss(*dst, *dst1);
  *dst1 = _mm_shuffle_ps(*dst1, *dst1, SSE_SHUFFLE_0321);
  _mm_store_ss(*dst + SSE_INDEX_1, *dst1);
  if (r > SSE_ROW_NUM_1) {
    *dst += stride;
    _mm_store_ss(*dst, *dst3);
    *dst3 = _mm_shuffle_ps(*dst3, *dst3, SSE_SHUFFLE_0321);
    _mm_store_ss(*dst + SSE_INDEX_1, *dst3);
  }
  if (r > SSE_ROW_NUM_2) {
    *dst += stride;
    _mm_store_ss(*dst, *dst5);
    *dst5 = _mm_shuffle_ps(*dst5, *dst5, SSE_SHUFFLE_0321);
    _mm_store_ss(*dst + SSE_INDEX_1, *dst5);
  }
  if (r > SSE_ROW_NUM_3) {
    *dst += stride;
    _mm_store_ss(*dst, *dst7);
    *dst7 = _mm_shuffle_ps(*dst7, *dst7, SSE_SHUFFLE_0321);
    _mm_store_ss(*dst + SSE_INDEX_1, *dst7);
    *dst += stride;
    *dst += SSE_INDEX_2;
  }
}

static inline void WriteCol3(float **dst, __m128 *dst1, __m128 *dst2, __m128 *dst3, __m128 *dst4, __m128 *dst5,
                             __m128 *dst6, __m128 *dst7, __m128 *dst8, int stride, int extra_stride, int r) {
  if (r > SSE_ROW_NUM_1) {
    *dst += stride;
    _mm_store_ss(*dst, *dst3);
    *dst3 = _mm_shuffle_ps(*dst3, *dst3, SSE_SHUFFLE_0321);
    _mm_store_ss(*dst + SSE_INDEX_1, *dst3);
    *dst3 = _mm_shuffle_ps(*dst3, *dst3, SSE_SHUFFLE_0321);
    _mm_store_ss(*dst + SSE_INDEX_2, *dst3);
  }
  if (r > SSE_ROW_NUM_2) {
    *dst += stride;
    _mm_store_ss(*dst, *dst5);
    *dst5 = _mm_shuffle_ps(*dst5, *dst5, SSE_SHUFFLE_0321);
    _mm_store_ss(*dst + SSE_INDEX_1, *dst5);
    *dst5 = _mm_shuffle_ps(*dst5, *dst5, SSE_SHUFFLE_0321);
    _mm_store_ss(*dst + SSE_INDEX_2, *dst5);
  }
  if (r > SSE_ROW_NUM_3) {
    *dst += stride;
    _mm_store_ss(*dst, *dst7);
    *dst7 = _mm_shuffle_ps(*dst7, *dst7, SSE_SHUFFLE_0321);
    _mm_store_ss(*dst + SSE_INDEX_1, *dst7);
    *dst7 = _mm_shuffle_ps(*dst7, *dst7, SSE_SHUFFLE_0321);
    _mm_store_ss(*dst + SSE_INDEX_2, *dst7);
    *dst += stride;
    *dst += extra_stride;
  }
}

static inline void WriteCol4(float **dst, __m128 *dst1, __m128 *dst2, __m128 *dst3, __m128 *dst4, __m128 *dst5,
                             __m128 *dst6, __m128 *dst7, __m128 *dst8, int stride, int extra_stride, int r) {
  _mm_storeu_ps(*dst, *dst1);
  if (r > SSE_ROW_NUM_1) {
    *dst += stride;
    _mm_storeu_ps(*dst, *dst3);
  }
  if (r > SSE_ROW_NUM_2) {
    *dst += stride;
    _mm_storeu_ps(*dst, *dst5);
  }
  if (r > SSE_ROW_NUM_3) {
    *dst += stride;
    _mm_storeu_ps(*dst, *dst7);
    *dst += stride;
    *dst += extra_stride;
  }
}

static inline void WriteCol5(float **dst, __m128 *dst1, __m128 *dst2, __m128 *dst3, __m128 *dst4, __m128 *dst5,
                             __m128 *dst6, __m128 *dst7, __m128 *dst8, int stride, int extra_stride, int r) {
  _mm_storeu_ps(*dst, *dst1);
  _mm_store_ss(*dst + SSE_INDEX_4, *dst2);
  if (r > SSE_ROW_NUM_1) {
    *dst += stride;
    _mm_storeu_ps(*dst, *dst3);
    _mm_store_ss(*dst + SSE_INDEX_4, *dst4);
  }
  if (r > SSE_ROW_NUM_2) {
    *dst += stride;
    _mm_storeu_ps(*dst, *dst5);
    _mm_store_ss(*dst + SSE_INDEX_4, *dst6);
  }
  if (r > SSE_ROW_NUM_3) {
    *dst += stride;
    _mm_storeu_ps(*dst, *dst7);
    _mm_store_ss(*dst + SSE_INDEX_4, *dst8);
    *dst += stride;
    *dst += extra_stride;
  }
}

static inline void WriteCol6(float **dst, __m128 *dst1, __m128 *dst2, __m128 *dst3, __m128 *dst4, __m128 *dst5,
                             __m128 *dst6, __m128 *dst7, __m128 *dst8, int stride, int extra_stride, int r) {
  _mm_storeu_ps(*dst, *dst1);
  _mm_store_ss(*dst + SSE_INDEX_4, *dst2);
  *dst2 = _mm_shuffle_ps(*dst2, *dst2, SSE_SHUFFLE_0321);
  _mm_store_ss(*dst + SSE_INDEX_5, *dst2);
  if (r > SSE_ROW_NUM_1) {
    *dst += stride;
    _mm_storeu_ps(*dst, *dst3);
    _mm_store_ss(*dst + SSE_INDEX_4, *dst4);
    *dst4 = _mm_shuffle_ps(*dst4, *dst4, SSE_SHUFFLE_0321);
    _mm_store_ss(*dst + SSE_INDEX_5, *dst4);
  }
  if (r > SSE_ROW_NUM_2) {
    *dst += stride;
    _mm_storeu_ps(*dst, *dst5);
    _mm_store_ss(*dst + SSE_INDEX_4, *dst6);
    *dst6 = _mm_shuffle_ps(*dst6, *dst6, SSE_SHUFFLE_0321);
    _mm_store_ss(*dst + SSE_INDEX_5, *dst6);
  }
  if (r > SSE_ROW_NUM_3) {
    *dst += stride;
    _mm_storeu_ps(*dst, *dst7);
    _mm_store_ss(*dst + SSE_INDEX_4, *dst8);
    *dst8 = _mm_shuffle_ps(*dst8, *dst8, SSE_SHUFFLE_0321);
    _mm_store_ss(*dst + SSE_INDEX_5, *dst8);
    *dst += stride;
    *dst += extra_stride;
  }
}

static inline void WriteCol7(float **dst, __m128 *dst1, __m128 *dst2, __m128 *dst3, __m128 *dst4, __m128 *dst5,
                             __m128 *dst6, __m128 *dst7, __m128 *dst8, int stride, int extra_stride, int r) {
  _mm_storeu_ps(*dst, *dst1);
  _mm_store_ss(*dst + SSE_INDEX_4, *dst2);
  *dst2 = _mm_shuffle_ps(*dst2, *dst2, SSE_SHUFFLE_0321);
  _mm_store_ss(*dst + SSE_INDEX_5, *dst2);
  *dst2 = _mm_shuffle_ps(*dst2, *dst2, SSE_SHUFFLE_0321);
  _mm_store_ss(*dst + SSE_INDEX_6, *dst2);
  if (r > SSE_ROW_NUM_1) {
    *dst += stride;
    _mm_storeu_ps(*dst, *dst3);
    _mm_store_ss(*dst + SSE_INDEX_4, *dst4);
    *dst4 = _mm_shuffle_ps(*dst4, *dst4, SSE_SHUFFLE_0321);
    _mm_store_ss(*dst + SSE_INDEX_5, *dst4);
    *dst4 = _mm_shuffle_ps(*dst4, *dst4, SSE_SHUFFLE_0321);
    _mm_store_ss(*dst + SSE_INDEX_6, *dst4);
  }
  if (r > SSE_ROW_NUM_2) {
    *dst += stride;
    _mm_storeu_ps(*dst, *dst5);
    _mm_store_ss(*dst + SSE_INDEX_4, *dst6);
    *dst6 = _mm_shuffle_ps(*dst6, *dst6, SSE_SHUFFLE_0321);
    _mm_store_ss(*dst + SSE_INDEX_5, *dst6);
    *dst6 = _mm_shuffle_ps(*dst6, *dst6, SSE_SHUFFLE_0321);
    _mm_store_ss(*dst + SSE_INDEX_6, *dst6);
  }
  if (r > SSE_ROW_NUM_3) {
    *dst += stride;
    _mm_storeu_ps(*dst, *dst7);
    _mm_store_ss(*dst + SSE_INDEX_4, *dst8);
    *dst8 = _mm_shuffle_ps(*dst8, *dst8, SSE_SHUFFLE_0321);
    _mm_store_ss(*dst + SSE_INDEX_5, *dst8);
    *dst8 = _mm_shuffle_ps(*dst8, *dst8, SSE_SHUFFLE_0321);
    _mm_store_ss(*dst + SSE_INDEX_6, *dst8);
    *dst += stride;
    *dst += extra_stride;
  }
}

static inline void WriteCol8(float **dst, __m128 *dst1, __m128 *dst2, __m128 *dst3, __m128 *dst4, __m128 *dst5,
                             __m128 *dst6, __m128 *dst7, __m128 *dst8, int stride, int extra_stride, int r) {
  _mm_storeu_ps(*dst, *dst1);
  _mm_storeu_ps(*dst + SSE_INDEX_4, *dst2);
  if (r > SSE_ROW_NUM_1) {
    *dst += stride;
    _mm_storeu_ps(*dst, *dst3);
    _mm_storeu_ps(*dst + SSE_INDEX_4, *dst4);
  }
  if (r > SSE_ROW_NUM_2) {
    *dst += stride;
    _mm_storeu_ps(*dst, *dst5);
    _mm_storeu_ps(*dst + SSE_INDEX_4, *dst6);
  }
  if (r > SSE_ROW_NUM_3) {
    *dst += stride;
    _mm_storeu_ps(*dst, *dst7);
    _mm_storeu_ps(*dst + SSE_INDEX_4, *dst8);
    *dst += stride;
    *dst += extra_stride;
  }
}

static inline void DoBiasBlock8(const float *bias_ptr, __m128 *dst1, __m128 *dst2, __m128 *dst3, __m128 *dst4,
                                __m128 *dst5, __m128 *dst6, __m128 *dst7, __m128 *dst8) {
  __m128 bias1 = _mm_loadu_ps(bias_ptr);
  __m128 bias2 = _mm_loadu_ps(bias_ptr + C4NUM);
  *dst1 = _mm_add_ps(*dst1, bias1);
  *dst2 = _mm_add_ps(*dst2, bias2);
  *dst3 = _mm_add_ps(*dst3, bias1);
  *dst4 = _mm_add_ps(*dst4, bias2);
  *dst5 = _mm_add_ps(*dst5, bias1);
  *dst6 = _mm_add_ps(*dst6, bias2);
  *dst7 = _mm_add_ps(*dst7, bias1);
  *dst8 = _mm_add_ps(*dst8, bias2);
}

#endif  // MINDSPORE_LITE_NNACL_INTRINSICS_SSE_SSE_COMMON_H_
