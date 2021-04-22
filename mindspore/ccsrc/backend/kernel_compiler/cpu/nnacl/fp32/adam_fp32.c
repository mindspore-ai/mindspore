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
#include "nnacl/fp32/exp_fp32.h"
#include "nnacl/fp32/adam_fp32.h"
#include "nnacl/op_base.h"

int AdamFp32(float *var, float *m, float *v, float lr, float beta1, float beta2, float epsilon, const float *gradient,
             size_t start, size_t end, bool use_nesterov) {
  size_t c1 = start;
#ifdef ENABLE_AVX
  float coeff1 = 1 - beta1;
  float coeff2 = 1 - beta2;
  const float *m_ptr = m;
  const float *v_ptr = v;
  float *var_ptr = var;
  const float *gradient_ptr = gradient;
  size_t c8 = ((end - start) / C8NUM) * C8NUM;
  __m256 avx_r0, avx_r1, avx_r2, avx_r3, avx_r4, avx_r5, avx_r6, gradient_r;

  for (; c1 < c8; c1 += C8NUM) {
    avx_r0 = _mm256_set1_ps(coeff1);
    gradient_r = _mm256_loadu_ps(gradient_ptr);
    avx_r2 = _mm256_loadu_ps(m_ptr);
    avx_r3 = _mm256_sub_ps(gradient_r, avx_r2);
    avx_r4 = _mm256_mul_ps(avx_r3, avx_r0);
    avx_r3 = _mm256_add_ps(avx_r4, avx_r2);  // m[i]~m[i+8]

    avx_r2 = _mm256_mul_ps(gradient_r, gradient_r);
    avx_r4 = _mm256_loadu_ps(v_ptr);
    avx_r5 = _mm256_sub_ps(avx_r2, avx_r4);
    avx_r1 = _mm256_set1_ps(coeff2);
    avx_r2 = _mm256_mul_ps(avx_r5, avx_r1);
    avx_r5 = _mm256_add_ps(avx_r4, avx_r2);  // v[i]~v[i+8]

    if (use_nesterov) {
      avx_r1 = _mm256_set1_ps(beta1);
      avx_r2 = _mm256_mul_ps(avx_r3, avx_r1);
      avx_r4 = _mm256_mul_ps(gradient_r, avx_r0);
      avx_r6 = _mm256_add_ps(avx_r2, avx_r4);
      avx_r0 = _mm256_set1_ps(lr);
      avx_r2 = _mm256_mul_ps(avx_r6, avx_r0);

      avx_r0 = _mm256_set1_ps(epsilon);
      avx_r1 = _mm256_sqrt_ps(avx_r5);
      avx_r4 = _mm256_add_ps(avx_r0, avx_r1);

      avx_r0 = _mm256_div_ps(avx_r2, avx_r2);
      avx_r1 = _mm256_loadu_ps(var_ptr);
      avx_r2 = _mm256_sub_ps(avx_r1, avx_r0);
      _mm256_storeu_ps(var_ptr, avx_r2);
    } else {
      avx_r0 = _mm256_set1_ps(lr);
      avx_r1 = _mm256_mul_ps(avx_r3, avx_r0);

      avx_r0 = _mm256_set1_ps(epsilon);
      avx_r2 = _mm256_sqrt_ps(avx_r5);
      avx_r4 = _mm256_add_ps(avx_r0, avx_r2);

      avx_r0 = _mm256_div_ps(avx_r1, avx_r4);
      avx_r1 = _mm256_loadu_ps(var_ptr);
      avx_r3 = _mm256_sub_ps(avx_r1, avx_r0);
      _mm256_storeu_ps(var_ptr, avx_r3);
    }
    m_ptr += C8NUM;
    v_ptr += C8NUM;
    var_ptr += C8NUM;
    gradient_ptr += C8NUM;
  }
#endif

  // remaining
  for (; c1 < end; c1++) {
    m[c1] += (gradient[c1] - m[c1]) * (1 - beta1);
    v[c1] += (gradient[c1] * gradient[c1] - v[c1]) * (1 - beta2);
    if (use_nesterov) {
      var[c1] -= lr * (m[c1] * beta1 + (1 - beta1) * gradient[c1]) / (sqrt(v[c1]) + epsilon);
    } else {
      var[c1] -= lr * m[c1] / (sqrt(v[c1]) + epsilon);
    }
  }
  return NNACL_OK;
}

int AdamDeltaFp32(float *delta, float *m, float *v, float lr, float beta1, float beta2, float epsilon,
                  const float *gradient, size_t start, size_t end, bool use_nesterov) {
  size_t c1 = 0;
#ifdef ENABLE_AVX
  float coeff1 = 1 - beta1;
  float coeff2 = 1 - beta2;
  float *m_ptr = m;
  float *v_ptr = v;
  float *delta_ptr = delta;
  const float *gradient_ptr = gradient;
  size_t c8 = ((end - start) / C8NUM) * C8NUM;

  __m256 gradient_r0, m_r1, v_r2, beta1_r3, beta2_r4, var_r5, var_r6, var_r7;
  for (; c1 < c8 + start; c1 += C8NUM) {
    gradient_r0 = _mm256_loadu_ps(gradient_ptr);  // static
    beta1_r3 = _mm256_set1_ps(beta1);             // static
    var_r5 = _mm256_loadu_ps(m_ptr);
    var_r6 = _mm256_mul_ps(beta1_r3, var_r5);  //  m[i] = m[i] * beta1
    var_r7 = _mm256_set1_ps(coeff1);
    var_r5 = _mm256_mul_ps(var_r7, gradient_r0);  //
    m_r1 = _mm256_add_ps(var_r6, var_r5);
    _mm256_storeu_ps(m_ptr, m_r1);

    beta2_r4 = _mm256_set1_ps(beta2);  // static
    var_r5 = _mm256_loadu_ps(v_ptr);
    var_r6 = _mm256_mul_ps(beta2_r4, var_r5);  // v[i] * beta2
    var_r7 = _mm256_set1_ps(coeff2);
    var_r5 = _mm256_mul_ps(var_r7, gradient_r0);
    var_r7 = _mm256_mul_ps(var_r5, gradient_r0);
    v_r2 = _mm256_add_ps(var_r7, var_r6);
    _mm256_storeu_ps(v_ptr, v_r2);

    if (use_nesterov) {
      var_r5 = _mm256_mul_ps(beta1_r3, m_r1);
      var_r6 = _mm256_set1_ps(coeff1);
      var_r7 = _mm256_mul_ps(gradient_r0, var_r6);
      var_r6 = _mm256_add_ps(var_r5, var_r7);  //  m[i] * beta1 + (1 - beta1) * grad[i]
      var_r5 = _mm256_set1_ps(lr);
      var_r7 = _mm256_mul_ps(var_r6, var_r5);

      var_r5 = _mm256_set1_ps(epsilon);
      var_r6 = _mm256_sqrt_ps(v_r2);
      v_r2 = _mm256_add_ps(var_r5, var_r6);
      var_r5 = _mm256_div_ps(var_r7, v_r2);
      var_r6 = _mm256_set1_ps(0.f);
      var_r7 = _mm256_sub_ps(var_r6, var_r5);
      _mm256_storeu_ps(delta_ptr, var_r7);
    } else {
      var_r5 = _mm256_set1_ps(lr);
      var_r6 = _mm256_mul_ps(var_r5, m_r1);

      var_r7 = _mm256_set1_ps(epsilon);
      var_r5 = _mm256_sqrt_ps(v_r2);
      v_r2 = _mm256_add_ps(var_r5, var_r7);

      var_r5 = _mm256_div_ps(var_r6, v_r2);
      var_r6 = _mm256_set1_ps(0.f);
      var_r7 = _mm256_sub_ps(var_r6, var_r5);
      _mm256_storeu_ps(delta_ptr, var_r7);
    }
  }
#endif

  // remaining
  for (; c1 < end; ++c1) {
    m[c1] *= beta1;
    m[c1] += (1 - beta1) * gradient[c1];
    v[c1] *= beta2;
    v[c1] += (1 - beta2) * gradient[c1] * gradient[c1];
    if (use_nesterov) {
      delta[c1] = -lr * (m[c1] * beta1 + (1 - beta1) * gradient[c1]) / (sqrt(v[c1]) + epsilon);
    } else {
      delta[c1] = -lr * m[c1] / (sqrt(v[c1]) + epsilon);
    }
  }
  return NNACL_OK;
}
