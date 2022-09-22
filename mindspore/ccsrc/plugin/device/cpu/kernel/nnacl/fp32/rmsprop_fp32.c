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
#include "nnacl/fp32/rmsprop_fp32.h"
#ifdef ENABLE_SSE
#ifdef _MSC_VER
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif
#endif

#ifdef ENABLE_AVX
#include <immintrin.h>
#endif

#include <math.h>
#include "nnacl/errorcode.h"

int RMSPropUnuseCenterFp32(float *variable, float *mean_square, float *moment, float *gradients, float momentum,
                           float learning_rate, float decay, float epsilon, size_t start, size_t end) {
  size_t c1 = start;
#ifdef ENABLE_AVX
  size_t c8 = ((end - start) / C8NUM) * C8NUM;
  float *variable_ptr = variable + start;
  float *mean_square_ptr = mean_square + start;
  float *gradients_ptr = gradients + start;
  float *moment_ptr = moment + start;

  __m256 decay_r = _mm256_set1_ps(1.0 - decay);
  __m256 momentum_r = _mm256_set1_ps(momentum);
  __m256 lr_r = _mm256_set1_ps(learning_rate);
  __m256 epsi_r = _mm256_set1_ps(epsilon);
  __m256 gradient_r, mean_square_r, moment_r, variable_r, avx_r1, avx_r2;
  for (; c1 < start + c8; c1 += C8NUM) {
    gradient_r = _mm256_loadu_ps(gradients_ptr);
    mean_square_r = _mm256_loadu_ps(mean_square_ptr);
    avx_r1 = _mm256_sub_ps(_mm256_mul_ps(gradient_r, gradient_r), mean_square_r);
    avx_r2 = _mm256_mul_ps(avx_r1, decay_r);
    mean_square_r = _mm256_add_ps(mean_square_r, avx_r2);
    _mm256_storeu_ps(mean_square_ptr, mean_square_r);

    avx_r1 = _mm256_add_ps(_mm256_sqrt_ps(mean_square_r), epsi_r);
    avx_r2 = _mm256_div_ps(_mm256_mul_ps(gradient_r, lr_r), avx_r1);

    moment_r = _mm256_loadu_ps(moment_ptr);
    avx_r1 = _mm256_add_ps(_mm256_mul_ps(moment_r, momentum_r), avx_r2);
    _mm256_storeu_ps(moment_ptr, avx_r1);

    variable_r = _mm256_loadu_ps(variable_ptr);
    variable_r = _mm256_sub_ps(variable_r, avx_r1);
    _mm256_storeu_ps(variable_ptr, variable_r);

    gradients_ptr += C8NUM;
    mean_square_ptr += C8NUM;
    moment_ptr += C8NUM;
    variable_ptr += C8NUM;
  }
#endif

  for (; c1 < end; c1++) {
    mean_square[c1] += (gradients[c1] * gradients[c1] - mean_square[c1]) * (1.0 - decay);
    moment[c1] = moment[c1] * momentum + (gradients[c1] * learning_rate) / sqrt(mean_square[c1] + epsilon);
    variable[c1] -= moment[c1];
  }
  return NNACL_OK;
}

int RMSPropUseCenterFp32(float *variable, float *mean_square, float *moment, float *gradients, float *mean_gradients,
                         float momentum, float learning_rate, float decay, float epsilon, size_t start, size_t end) {
  size_t c1 = start;
#ifdef ENABLE_AVX
  size_t c8 = ((end - start) / C8NUM) * C8NUM;
  float *variable_ptr = variable + start;
  float *mean_gradients_ptr = mean_gradients + start;
  float *mean_square_ptr = mean_square + start;
  float *moment_ptr = moment + start;
  float *gradients_ptr = gradients + start;

  __m256 decay_r = _mm256_set1_ps(1.0 - decay);
  __m256 momentum_r = _mm256_set1_ps(momentum);
  __m256 lr_r = _mm256_set1_ps(learning_rate);
  __m256 epsi_r = _mm256_set1_ps(epsilon);
  __m256 grad_r, mean_grad_r, mean_square_r, moment_r, variable_r;
  __m256 avx_r1, avx_r2;
  for (; c1 < start + c8; c1 += C8NUM) {
    grad_r = _mm256_loadu_ps(gradients_ptr);
    mean_square_r = _mm256_loadu_ps(mean_square_ptr);
    avx_r1 = _mm256_sub_ps(_mm256_mul_ps(grad_r, grad_r), mean_square_r);
    avx_r2 = _mm256_mul_ps(avx_r1, decay_r);
    mean_square_r = _mm256_add_ps(mean_square_r, avx_r2);
    _mm256_storeu_ps(mean_square_ptr, mean_square_r);

    mean_grad_r = _mm256_loadu_ps(mean_gradients_ptr);
    avx_r1 = _mm256_mul_ps(_mm256_sub_ps(grad_r, mean_grad_r), decay_r);
    mean_grad_r = _mm256_add_ps(mean_grad_r, avx_r1);
    _mm256_storeu_ps(mean_gradients_ptr, mean_grad_r);

    avx_r1 = _mm256_sub_ps(mean_square_r, _mm256_mul_ps(mean_grad_r, mean_grad_r));
    __m256 denom_r = _mm256_add_ps(avx_r1, epsi_r);
    __m256 cmp_r = _mm256_cmp_ps(denom_r, _mm256_setzero_ps(), _CMP_GE_OS);
    __m256 gt_zero_r = _mm256_blendv_ps(_mm256_set1_ps(1.0f), denom_r, cmp_r);

    avx_r1 = _mm256_mul_ps(grad_r, lr_r);
    avx_r2 = _mm256_div_ps(avx_r1, _mm256_sqrt_ps(gt_zero_r));
    moment_r = _mm256_loadu_ps(moment_ptr);
    avx_r1 = _mm256_mul_ps(moment_r, momentum_r);
    avx_r1 = _mm256_add_ps(avx_r1, avx_r2);
    moment_r = _mm256_blendv_ps(moment_r, avx_r1, cmp_r);
    _mm256_storeu_ps(moment_ptr, moment_r);

    variable_r = _mm256_loadu_ps(variable_ptr);
    avx_r1 = _mm256_sub_ps(variable_r, moment_r);
    variable_r = _mm256_blendv_ps(variable_r, avx_r1, cmp_r);
    _mm256_storeu_ps(variable_ptr, variable_r);

    variable_ptr += C8NUM;
    mean_gradients_ptr += C8NUM;
    mean_square_ptr += C8NUM;
    gradients_ptr += C8NUM;
    moment_ptr += C8NUM;
  }
#endif

  for (; c1 < end; c1++) {
    mean_square[c1] += (gradients[c1] * gradients[c1] - mean_square[c1]) * (1.0 - decay);
    mean_gradients[c1] += (gradients[c1] - mean_gradients[c1]) * (1.0 - decay);
    float denom = (mean_square[c1] - mean_gradients[c1] * mean_gradients[c1]) + epsilon;
    if (denom > 0) {
      moment[c1] = moment[c1] * momentum + (gradients[c1] * learning_rate) / sqrt(denom);
      variable[c1] -= moment[c1];
    }
  }
  return NNACL_OK;
}
