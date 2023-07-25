/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "sparse_ftrl_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__device__ __forceinline__ T PowFunc(T x, T y) {
  return pow(x, y);
}

template <>
__device__ __forceinline__ half PowFunc(half x, half y) {
  return __float2half(pow(__half2float(x), __half2float(y)));
}

template <typename T>
__device__ __forceinline__ bool CompareFunc(T x, T y) {
  return abs(x) > y;
}

template <>
__device__ __forceinline__ bool CompareFunc(half x, half y) {
  return abs(__half2float(x)) > __half2float(y);
}

template <typename T>
__device__ __forceinline__ T Sgn(T x) {
  return static_cast<T>(x != 0 ? (x > 0 ? 1 : -1) : 0);
}

template <>
__device__ __forceinline__ half Sgn(half x) {
  return __float2half(__half2float(x) != 0 ? (__half2float(x) > 0 ? 1 : -1) : 0);
}

template <typename T, typename S>
__global__ void SparseApplyFtrlKernel(const T *gradient, const S *indices, const int num_index, const size_t n_stride,
                                      const float learning_rate, const float l1_regularization,
                                      const float l2_regularization, const float learning_rate_power, T *variable,
                                      T *accumulation, T *linear) {
  const T two = static_cast<T>(2.0);
  const T learning_rate_val = static_cast<T>(learning_rate);
  const T l1_regularization_val = static_cast<T>(l1_regularization);
  const T l2_regularization_val = static_cast<T>(l2_regularization);
  const T learning_rate_power_val = static_cast<T>(-learning_rate_power);

  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (num_index * n_stride);
       pos += gridDim.x * blockDim.x) {
    const int posn = pos / n_stride;
    const int posi = pos % n_stride;
    const int indexed_n = indices[posn];
    const int i = indexed_n * n_stride + posi;
    const T cur_accumulation = accumulation[i] + gradient[pos] * gradient[pos];
    const T accumulation_power = PowFunc(accumulation[i], learning_rate_power_val);
    const T cur_accumulation_power = PowFunc(cur_accumulation, learning_rate_power_val);
    const T sigma = (cur_accumulation_power - accumulation_power) / learning_rate_val;

    linear[i] += gradient[pos] - sigma * variable[i];
    variable[i] = CompareFunc(linear[i], l1_regularization_val)
                    ? ((l1_regularization_val * Sgn(linear[i]) - linear[i]) /
                       (cur_accumulation_power / learning_rate_val + two * l2_regularization_val))
                    : static_cast<T>(0);
    accumulation[i] = cur_accumulation;
  }
}

template <typename T, typename S>
cudaError_t CalSparseApplyFtrl(const T *gradient, const S *indices, const int num_index, const size_t n_stride,
                               const float learning_rate, const float l1_regularization, const float l2_regularization,
                               const float learning_rate_power, const bool use_locking, T *variable, T *accumulation,
                               T *linear, cudaStream_t cuda_stream) {
  SparseApplyFtrlKernel<<<GET_BLOCKS(num_index * n_stride), GET_THREADS, 0, cuda_stream>>>(
    gradient, indices, num_index, n_stride, learning_rate, l1_regularization, l2_regularization, learning_rate_power,
    variable, accumulation, linear);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyFtrl<float, int>(
  const float *gradient, const int *indices, const int num_index, const size_t n_stride, const float learning_rate,
  const float l1_regularization, const float l2_regularization, const float learning_rate_power, const bool use_locking,
  float *variable, float *accumulation, float *linear, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyFtrl<float, int64_t>(
  const float *gradient, const int64_t *indices, const int num_index, const size_t n_stride, const float learning_rate,
  const float l1_regularization, const float l2_regularization, const float learning_rate_power, const bool use_locking,
  float *variable, float *accumulation, float *linear, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyFtrl<half, int>(
  const half *gradient, const int *indices, const int num_index, const size_t n_stride, const float learning_rate,
  const float l1_regularization, const float l2_regularization, const float learning_rate_power, const bool use_locking,
  half *variable, half *accumulation, half *linear, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyFtrl<half, int64_t>(
  const half *gradient, const int64_t *indices, const int num_index, const size_t n_stride, const float learning_rate,
  const float l1_regularization, const float l2_regularization, const float learning_rate_power, const bool use_locking,
  half *variable, half *accumulation, half *linear, cudaStream_t cuda_stream);
