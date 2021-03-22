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

#ifdef __cplusplus
extern "C" {
#endif

void ActBlock1(__m128 *v1, size_t relu, size_t relu6);
void ActBlock2(__m128 *v1, __m128 *v2, size_t relu, size_t relu6);
void ActBlock4(__m128 *v1, __m128 *v2, __m128 *v3, __m128 *v4, size_t relu, size_t relu6);
void ActBlock8(__m128 *v1, __m128 *v2, __m128 *v3, __m128 *v4, __m128 *v5, __m128 *v6, __m128 *v7, __m128 *v8,
               size_t relu_type);

void WriteCol1(float **dst, __m128 *dst1, __m128 *dst2, __m128 *dst3, __m128 *dst4, __m128 *dst5, __m128 *dst6,
               __m128 *dst7, __m128 *dst8, int stride, int extra_stride, int r);
void WriteCol2(float **dst, __m128 *dst1, __m128 *dst2, __m128 *dst3, __m128 *dst4, __m128 *dst5, __m128 *dst6,
               __m128 *dst7, __m128 *dst8, int stride, int r);
void WriteCol2Opt(float **dst, __m128 *dst1, __m128 *dst2, __m128 *dst3, __m128 *dst4, __m128 *dst5, __m128 *dst6,
                  __m128 *dst7, __m128 *dst8, int stride, int r);
void WriteCol3(float **dst, __m128 *dst1, __m128 *dst2, __m128 *dst3, __m128 *dst4, __m128 *dst5, __m128 *dst6,
               __m128 *dst7, __m128 *dst8, int stride, int extra_stride, int r);
void WriteCol4(float **dst, __m128 *dst1, __m128 *dst2, __m128 *dst3, __m128 *dst4, __m128 *dst5, __m128 *dst6,
               __m128 *dst7, __m128 *dst8, int stride, int extra_stride, int r);
void WriteCol5(float **dst, __m128 *dst1, __m128 *dst2, __m128 *dst3, __m128 *dst4, __m128 *dst5, __m128 *dst6,
               __m128 *dst7, __m128 *dst8, int stride, int extra_stride, int r);
void WriteCol6(float **dst, __m128 *dst1, __m128 *dst2, __m128 *dst3, __m128 *dst4, __m128 *dst5, __m128 *dst6,
               __m128 *dst7, __m128 *dst8, int stride, int extra_stride, int r);
void WriteCol7(float **dst, __m128 *dst1, __m128 *dst2, __m128 *dst3, __m128 *dst4, __m128 *dst5, __m128 *dst6,
               __m128 *dst7, __m128 *dst8, int stride, int extra_stride, int r);
void WriteCol8(float **dst, __m128 *dst1, __m128 *dst2, __m128 *dst3, __m128 *dst4, __m128 *dst5, __m128 *dst6,
               __m128 *dst7, __m128 *dst8, int stride, int extra_stride, int r);

void DoBiasBlock8(const float *bias_ptr, __m128 *dst1, __m128 *dst2, __m128 *dst3, __m128 *dst4, __m128 *dst5,
                  __m128 *dst6, __m128 *dst7, __m128 *dst8);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_INTRINSICS_SSE_SSE_COMMON_H_
