/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/determinant_by_lu_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

template <typename T>
using Complex = mindspore::utils::Complex<T>;

__inline__ __device__ int PermutationOrder(int m, const int *per_batch_pivot) {
  int permutation_order = 0;
  for (int i = 0; i < m - 1; ++i) {
    permutation_order += per_batch_pivot[i] != (i + 1);
  }
  return permutation_order;
}

template <typename T>
__inline__ __device__ T CalInFiniteValue(const T sum_abs_log_det) {
  constexpr T template_zero = static_cast<T>(0);
  T result = sum_abs_log_det > template_zero ? -log(template_zero) : log(template_zero);
  return result;
}

template <>
__inline__ __device__ Complex<float> CalInFiniteValue(const Complex<float> sum_abs_log_det) {
  Complex<float> template_zero = static_cast<Complex<float>>(0);
  Complex<float> result = sum_abs_log_det.real() > template_zero.real() ? -log(template_zero) : log(template_zero);
  return result;
}

template <>
__inline__ __device__ Complex<double> CalInFiniteValue(const Complex<double> sum_abs_log_det) {
  Complex<double> template_zero = static_cast<Complex<double>>(0);
  Complex<double> result = sum_abs_log_det.real() > template_zero.real() ? -log(template_zero) : log(template_zero);
  return result;
}

template <typename T>
__global__ void CalculateDeterminantByLuKernel(const T *lu_input, const int *pivot, int m, int batch_size,
                                               bool is_sign_log_determinant, T *determinant_output, T *sign_output) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (batch_size); index += blockDim.x * gridDim.x) {
    const int permutation_order = PermutationOrder(m, pivot + index * m);
    T prod_sign = permutation_order % 2 ? (-1) : 1;
    T sum_abs_log_det = 0;
    int matrix_size = m * m;
    int stride = m + 1;
    size_t lu_i_index = matrix_size * index;
    // Get lu data's diagonal by stride.
    for (int i = 0; i < m; ++i, lu_i_index += stride) {
      const T abs_i = abs(lu_input[lu_i_index]);
      sum_abs_log_det += log(abs_i);
      prod_sign = prod_sign * (lu_input[lu_i_index] / abs_i);
    }
    if (!isfinite(sum_abs_log_det)) {
      prod_sign = 0;
      sum_abs_log_det = CalInFiniteValue(sum_abs_log_det);
    }
    if (is_sign_log_determinant) {
      sign_output[index] = prod_sign;
      determinant_output[index] = sum_abs_log_det;
    } else {
      determinant_output[index] = prod_sign * exp(sum_abs_log_det);
    }
  }
}

#ifdef _WIN32
template <>
__global__ void CalculateDeterminantByLuKernel(const Complex<float> *lu_input, const int *pivot, int m, int batch_size,
                                               bool is_sign_log_determinant, Complex<float> *determinant_output,
                                               Complex<float> *sign_output) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (batch_size); index += blockDim.x * gridDim.x) {
    const int permutation_order = PermutationOrder(m, pivot + index * m);
    Complex<float> prod_sign = permutation_order % 2 ? (-1) : 1;
    Complex<float> sum_abs_log_det = 0;
    int matrix_size = m * m;
    int stride = m + 1;
    size_t lu_i_index = matrix_size * index;
    // Get lu data's diagonal by stride.
    for (int i = 0; i < m; ++i, lu_i_index += stride) {
      const Complex<float> abs_i = abs(lu_input[lu_i_index]);
      sum_abs_log_det += log(abs_i);
      prod_sign = prod_sign * (lu_input[lu_i_index] / abs_i);
    }
    if (!mindspore::utils::isfinite(sum_abs_log_det)) {
      prod_sign = 0;
      sum_abs_log_det = CalInFiniteValue(sum_abs_log_det);
    }
    if (is_sign_log_determinant) {
      sign_output[index] = prod_sign;
      determinant_output[index] = sum_abs_log_det;
    } else {
      determinant_output[index] = prod_sign * exp(sum_abs_log_det);
    }
  }
}

template <>
__global__ void CalculateDeterminantByLuKernel(const Complex<double> *lu_input, const int *pivot, int m, int batch_size,
                                               bool is_sign_log_determinant, Complex<double> *determinant_output,
                                               Complex<double> *sign_output) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (batch_size); index += blockDim.x * gridDim.x) {
    const int permutation_order = PermutationOrder(m, pivot + index * m);
    Complex<double> prod_sign = permutation_order % 2 ? (-1) : 1;
    Complex<double> sum_abs_log_det = 0;
    int matrix_size = m * m;
    int stride = m + 1;
    size_t lu_i_index = matrix_size * index;
    // Get lu data's diagonal by stride.
    for (int i = 0; i < m; ++i, lu_i_index += stride) {
      const Complex<double> abs_i = abs(lu_input[lu_i_index]);
      sum_abs_log_det += log(abs_i);
      prod_sign = prod_sign * (lu_input[lu_i_index] / abs_i);
    }
    if (!mindspore::utils::isfinite(sum_abs_log_det)) {
      prod_sign = 0;
      sum_abs_log_det = CalInFiniteValue(sum_abs_log_det);
    }
    if (is_sign_log_determinant) {
      sign_output[index] = prod_sign;
      determinant_output[index] = sum_abs_log_det;
    } else {
      determinant_output[index] = prod_sign * exp(sum_abs_log_det);
    }
  }
}
#endif

template <typename T>
void CalculateDeterminantByLu(const T *lu_input, const int *pivot, int m, int batch_size,
                              bool is_sign_log_determinant, T *determinant_output, T *sign_output,
                              const uint32_t &device_id, cudaStream_t cuda_stream) {
  // Parallelization by batch_size.
  CalculateDeterminantByLuKernel<<<CUDA_BLOCKS(device_id, batch_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    lu_input, pivot, m, batch_size, is_sign_log_determinant, determinant_output, sign_output);
}

template CUDA_LIB_EXPORT void CalculateDeterminantByLu<float>(const float *lu_input, const int *pivot, int m,
                                                              int batch_size, bool is_sign_log_determinant,
                                                              float *determinant_output, float *sign_output,
                                                              const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalculateDeterminantByLu<double>(const double *lu_input, const int *pivot, int m,
                                                               int batch_size, bool is_sign_log_determinant,
                                                               double *determinant_output, double *sign_output,
                                                               const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalculateDeterminantByLu<Complex<float>>(
  const Complex<float> *lu_input, const int *pivot, int m, int batch_size, bool is_sign_log_determinant,
  Complex<float> *determinant_output, Complex<float> *sign_output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalculateDeterminantByLu<Complex<double>>(
  const Complex<double> *lu_input, const int *pivot, int m, int batch_size, bool is_sign_log_determinant,
  Complex<double> *determinant_output, Complex<double> *sign_output, const uint32_t &device_id,
  cudaStream_t cuda_stream);
