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
#include <math.h>
#include "nnacl/fp32/exp_fp32.h"
#include "nnacl/fp32/adam_fp32.h"
#include "nnacl/intrinsics/ms_simd_instructions.h"
#ifdef ENABLE_AVX512
#include "nnacl/avx512/adam_fp32_avx512.h"
#endif

int AdamFp32(float *var, float *m, float *v, float lr, float beta1, float beta2, float epsilon, const float *gradient,
             size_t start, size_t end, bool use_nesterov) {
  size_t c1 = start;
#ifdef ENABLE_AVX
  size_t c8 = ((end - start) / C8NUM) * C8NUM;
  __m256 coeff1_r = _mm256_set1_ps(1 - beta1);
  __m256 coeff2_r = _mm256_set1_ps(1 - beta2);
  __m256 beta1_r = _mm256_set1_ps(beta1);
  __m256 lr_r = _mm256_set1_ps(lr);
  __m256 epsi_r = _mm256_set1_ps(epsilon);

  float *var_ptr = var + start;
  float *m_ptr = m + start;
  float *v_ptr = v + start;
  const float *grad_ptr = gradient + start;

  __m256 avx_r0, avx_r1;
  __m256 var_r, m_r, v_r, grad_r;

  for (; c1 < start + c8; c1 += C8NUM) {
    grad_r = _mm256_loadu_ps(grad_ptr);
    m_r = _mm256_loadu_ps(m_ptr);
    avx_r0 = _mm256_sub_ps(grad_r, m_r);
    avx_r1 = _mm256_mul_ps(avx_r0, coeff1_r);
    m_r = _mm256_add_ps(m_r, avx_r1);
    _mm256_storeu_ps(m_ptr, m_r);

    v_r = _mm256_loadu_ps(v_ptr);
    avx_r0 = _mm256_sub_ps(_mm256_mul_ps(grad_r, grad_r), v_r);
    v_r = _mm256_add_ps(v_r, _mm256_mul_ps(avx_r0, coeff2_r));
    _mm256_storeu_ps(v_ptr, v_r);

    if (use_nesterov) {
      avx_r0 = _mm256_add_ps(_mm256_mul_ps(m_r, beta1_r), _mm256_mul_ps(coeff1_r, grad_r));
      avx_r1 = _mm256_mul_ps(lr_r, avx_r0);
      avx_r0 = _mm256_add_ps(_mm256_sqrt_ps(v_r), epsi_r);
      __m256 avx_r2 = _mm256_div_ps(avx_r1, avx_r0);

      var_r = _mm256_loadu_ps(var_ptr);
      var_r = _mm256_sub_ps(var_r, avx_r2);
      _mm256_storeu_ps(var_ptr, var_r);
    } else {
      avx_r0 = _mm256_mul_ps(lr_r, m_r);
      avx_r1 = _mm256_add_ps(_mm256_sqrt_ps(v_r), epsi_r);
      __m256 avx_r2 = _mm256_div_ps(avx_r0, avx_r1);
      var_r = _mm256_loadu_ps(var_ptr);
      var_r = _mm256_sub_ps(var_r, avx_r2);
      _mm256_storeu_ps(var_ptr, var_r);
    }
    m_ptr += C8NUM;
    v_ptr += C8NUM;
    var_ptr += C8NUM;
    grad_ptr += C8NUM;
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
  size_t c1 = start;
#ifdef ENABLE_AVX
  size_t c8 = ((end - start) / C8NUM) * C8NUM;
  __m256 coeff1_r = _mm256_set1_ps(1.0f - beta1);
  __m256 coeff2_r = _mm256_set1_ps(1.0f - beta2);
  __m256 beta1_r = _mm256_set1_ps(beta1);
  __m256 beta2_r = _mm256_set1_ps(beta2);
  __m256 lr_r = _mm256_set1_ps(-lr);
  __m256 epsi_r = _mm256_set1_ps(epsilon);

  float *m_ptr = m + start;
  float *v_ptr = v + start;
  float *delta_ptr = delta + start;
  const float *gradient_ptr = gradient + start;

  __m256 m_r, v_r, delta_r, grad_r;
  __m256 avx_r0, avx_r1;
  for (; c1 < start + c8; c1 += C8NUM) {
    m_r = _mm256_loadu_ps(m_ptr);
    avx_r0 = _mm256_mul_ps(m_r, beta1_r);
    grad_r = _mm256_loadu_ps(gradient_ptr);
    m_r = _mm256_add_ps(avx_r0, _mm256_mul_ps(coeff1_r, grad_r));
    _mm256_storeu_ps(m_ptr, m_r);

    v_r = _mm256_loadu_ps(v_ptr);
    avx_r0 = _mm256_mul_ps(v_r, beta2_r);
    avx_r1 = _mm256_mul_ps(_mm256_mul_ps(coeff2_r, grad_r), grad_r);
    v_r = _mm256_add_ps(avx_r0, avx_r1);
    _mm256_storeu_ps(v_ptr, v_r);

    if (use_nesterov) {
      avx_r0 = _mm256_add_ps(_mm256_mul_ps(m_r, beta1_r), _mm256_mul_ps(coeff1_r, grad_r));
      avx_r0 = _mm256_mul_ps(lr_r, avx_r0);
      avx_r1 = _mm256_add_ps(_mm256_sqrt_ps(v_r), epsi_r);
      delta_r = _mm256_div_ps(avx_r0, avx_r1);
      _mm256_storeu_ps(delta_ptr, delta_r);
    } else {
      avx_r0 = _mm256_mul_ps(lr_r, m_r);
      avx_r1 = _mm256_add_ps(_mm256_sqrt_ps(v_r), epsi_r);
      delta_r = _mm256_div_ps(avx_r0, avx_r1);
      _mm256_storeu_ps(delta_ptr, delta_r);
    }
    m_ptr += C8NUM;
    v_ptr += C8NUM;
    delta_ptr += C8NUM;
    gradient_ptr += C8NUM;
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

int AdamWeightDecayFp32(float *var, float *m, float *v, float lr, float beta1, float beta2, float epsilon, float decay,
                        const float *gradient, size_t start, size_t end) {
  size_t c1 = start;
  SIMD_RUN_AVX512(AdamWeightDecayFp32, c1, var, m, v, lr, beta1, beta2, epsilon, decay, gradient, end);

  // remaining
  const float beta1_minus = 1 - beta1;
  const float beta2_minus = 1 - beta2;
  for (; c1 < end; c1++) {
    m[c1] += (gradient[c1] - m[c1]) * beta1_minus;
    v[c1] += (gradient[c1] * gradient[c1] - v[c1]) * beta2_minus;
    var[c1] -= lr * (m[c1] / (sqrt(v[c1]) + epsilon) + decay * var[c1]);
  }
  return NNACL_OK;
}

size_t FusedCastAdamFp32(float *var, float *m, float *v, float lr, float beta1, float beta2, float epsilon, float decay,
                         const int16_t *gradient16, float global_norm_reciprocal, size_t start, size_t end) {
  size_t c1 = start;

  SIMD_RUN_AVX512(FusedCastAdamFp32, c1, var, m, v, lr, beta1, beta2, epsilon, decay, gradient16,
                  global_norm_reciprocal, end);
  return c1;
}

size_t FusedCastAdamFp16(int16_t *var16, float *m, float *v, float lr, float beta1, float beta2, float epsilon,
                         float decay, const int16_t *gradient16, float global_norm_reciprocal, size_t start,
                         size_t end) {
  size_t c1 = start;
  SIMD_RUN_AVX512(FusedCastAdamFp16, c1, var16, m, v, lr, beta1, beta2, epsilon, decay, gradient16,
                  global_norm_reciprocal, end);
  return c1;
}
