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
#ifdef ENABLE_SSE
#include <x86intrin.h>
#endif

#ifdef ENABLE_AVX
#include <immintrin.h>
#endif

#include <math.h>
#include "nnacl/fp32/rmsprop_fp32.h"

int RMSPropUnuseCenterFp32(float *variable, float *mean_square, float *moment, float *gradients, float momentum,
                           float learning_rate, float decay, float epsilon, size_t start, size_t end) {
  size_t c1 = start;
#ifdef ENABLE_AVX
  float *variable_ptr = variable;
  float *mean_square_ptr = mean_square;
  float *gradients_ptr = gradients;
  float *moment_ptr = moment;
  float decay_v = 1.0 - decay;

  size_t c8 = ((end - start) / C8NUM) * C8NUM;
  __m256 decay_r = _mm256_set1_ps(decay_v);
  __m256 moment_r = _mm256_set1_ps(momentum);
  __m256 lr_r = _mm256_set1_ps(learning_rate);
  __m256 gradient_r, mean_square_r, tmp_r1, tmp_r2, tmp_r3;
  for (; c1 < c8; c1 += C8NUM) {
    gradient_r = _mm256_loadu_ps(gradients_ptr);
    tmp_r1 = _mm256_mul_ps(gradient_r, gradient_r);
    tmp_r2 = _mm256_loadu_ps(mean_square_ptr);
    tmp_r3 = _mm256_sub_ps(tmp_r1, tmp_r2);
    tmp_r1 = _mm256_mul_ps(decay_r, tmp_r3);
    mean_square_r = _mm256_add_ps(tmp_r2, tmp_r1);
    _mm256_storeu_ps(mean_square_ptr, mean_square_r);

    tmp_r1 = _mm256_set1_ps(epsilon);
    tmp_r2 = _mm256_add_ps(mean_square_r, tmp_r1);
    tmp_r1 = _mm256_sqrt_ps(tmp_r2);
    tmp_r2 = _mm256_mul_ps(gradient_r, lr_r);
    tmp_r3 = _mm256_div_ps(tmp_r2, tmp_r1);
    tmp_r1 = _mm256_loadu_ps(moment_ptr);
    tmp_r2 = _mm256_mul_ps(tmp_r1, moment_r);
    tmp_r3 = _mm256_add_ps(tmp_r2, tmp_r3);
    _mm256_storeu_ps(moment_ptr, tmp_r3);

    tmp_r1 = _mm256_loadu_ps(variable_ptr);
    tmp_r2 = _mm256_sub_ps(tmp_r1, tmp_r3);
    _mm256_storeu_ps(variable_ptr, tmp_r2);

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
  float *variable_ptr = variable;
  float *mean_gradients_ptr = mean_gradients;
  float *mean_square_ptr = mean_square;
  float *moment_ptr = moment;
  const float *gradients_ptr = gradients;
  const float decay_v = 1.0f - decay;

  size_t c8 = (start - end / C8NUM) * C8NUM;
  __m256 gradient_r;
  __m256 var_r1, var_r2, var_r3, var_r4, var_r5, var_r6;
  for (; c1 < c8; c1 += C8NUM) {
    gradient_r = _mm256_loadu_ps(gradients_ptr);
    var_r1 = _mm256_mul_ps(gradient_r, gradient_r);
    var_r2 = _mm256_loadu_ps(mean_square_ptr);  //
    var_r1 = _mm256_sub_ps(var_r1, var_r2);
    var_r3 = _mm256_set1_ps(decay_v);  // 1 - decay ...
    var_r1 = _mm256_mul_ps(var_r1, var_r3);
    var_r1 = _mm256_add_ps(var_r2, var_r1);  // mean_squasre ...
    _mm256_storeu_ps(mean_square_ptr, var_r1);

    var_r2 = _mm256_loadu_ps(mean_gradients_ptr);
    var_r4 = _mm256_sub_ps(gradient_r, var_r2);
    var_r3 = _mm256_mul_ps(var_r4, var_r3);
    var_r2 = _mm256_add_ps(var_r2, var_r3);  // mean_gradients ..
    _mm256_storeu_ps(mean_gradients_ptr, var_r2);

    var_r3 = _mm256_mul_ps(var_r2, var_r2);
    var_r3 = _mm256_sub_ps(var_r1, var_r3);
    var_r4 = _mm256_set1_ps(epsilon);
    var_r3 = _mm256_add_ps(var_r3, var_r4);  // denom ...
    var_r5 = _mm256_setzero_ps();
    var_r1 = _mm256_cmp_ps(var_r3, var_r5, _CMP_GE_OS);  // mask_r

    var_r4 = _mm256_set1_ps(learning_rate);
    var_r5 = _mm256_mul_ps(gradient_r, var_r4);
    var_r4 = _mm256_sqrt_ps(var_r3);
    var_r6 = _mm256_div_ps(var_r5, var_r4);  // (gradients[i] * learning_rate[0]) / sqrt(denom)
    var_r4 = _mm256_loadu_ps(moment_ptr);    // ....
    var_r5 = _mm256_set1_ps(momentum);
    var_r2 = _mm256_mul_ps(var_r4, var_r5);  // moment[i] * momentum[i]
    var_r3 = _mm256_add_ps(var_r6, var_r2);
    var_r4 = _mm256_blendv_ps(var_r4, var_r3, var_r1);
    _mm256_storeu_ps(moment_ptr, var_r4);

    var_r2 = _mm256_loadu_ps(variable_ptr);
    var_r5 = _mm256_sub_ps(var_r2, var_r4);
    var_r6 = _mm256_blendv_ps(var_r2, var_r5, var_r1);
    _mm256_storeu_ps(variable_ptr, var_r6);

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
