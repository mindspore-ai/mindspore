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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/ftrl_impl.cuh"
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

template <typename T>
using Complex = mindspore::utils::Complex<T>;

using Complex64 = Complex<float>;
using Complex128 = Complex<double>;

template <typename T>
__device__ __forceinline__ T PowFunc(T x, T y) {
  return static_cast<T>(pow(static_cast<double>(x), static_cast<double>(y)));
}

template <>
__device__ __forceinline__ half PowFunc(half x, half y) {
  return __float2half(pow(__half2float(x), __half2float(y)));
}

template <typename T>
__device__ __forceinline__ bool CompareFunc(T x, T y) {
  return abs(static_cast<double>(x)) > static_cast<double>(y);
}

template <>
__device__ __forceinline__ bool CompareFunc(half x, half y) {
  return abs(__half2float(x)) > __half2float(y);
}

template <typename T>
__device__ __forceinline__ T Sgn(T x) {
  return static_cast<T>(static_cast<double>(x) != 0 ? (static_cast<double>(x) > 0 ? 1 : -1) : 0);
}

template <>
__device__ __forceinline__ half Sgn(half x) {
  return __float2half(__half2float(x) != 0 ? (__half2float(x) > 0 ? 1 : -1) : 0);
}

template <typename T>
__global__ void ApplyFtrlKernel(const size_t size, const T *gradient, const T *learning_rate,
                                const T *l1_regularization, const T *l2_regularization, const T *learning_rate_power,
                                T *variable, T *accumulation, T *linear) {
  const T two = static_cast<T>(2.0);
  const T learning_rate_power_val = -learning_rate_power[0];

  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    const T cur_accumulation = accumulation[i] + gradient[i] * gradient[i];
    const T accumulation_power = PowFunc(accumulation[i], learning_rate_power_val);
    const T cur_accumulation_power = PowFunc(cur_accumulation, learning_rate_power_val);
    const T sigma = (cur_accumulation_power - accumulation_power) / learning_rate[0];

    linear[i] += gradient[i] - sigma * variable[i];
    variable[i] = CompareFunc(linear[i], l1_regularization[0])
                    ? ((l1_regularization[0] * Sgn(linear[i]) - linear[i]) /
                       (cur_accumulation_power / learning_rate[0] + two * l2_regularization[0]))
                    : static_cast<T>(0);
    accumulation[i] = cur_accumulation;
  }
}

template <typename T>
cudaError_t ApplyFtrl(const size_t size, const T *gradient, const T *learning_rate, const T *l1_regularization,
                      const T *l2_regularization, const T *learning_rate_power, T *variable, T *accumulation, T *linear,
                      cudaStream_t cuda_stream) {
  ApplyFtrlKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, gradient, learning_rate, l1_regularization,
                                                                     l2_regularization, learning_rate_power, variable,
                                                                     accumulation, linear);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t ApplyFtrl<float>(const size_t size, const float *gradient,
                                                      const float *learning_rate, const float *l1_regularization,
                                                      const float *l2_regularization, const float *learning_rate_power,
                                                      float *variable, float *accumulation, float *linear,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ApplyFtrl<half>(const size_t size, const half *gradient, const half *learning_rate,
                                                     const half *l1_regularization, const half *l2_regularization,
                                                     const half *learning_rate_power, half *variable,
                                                     half *accumulation, half *linear, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ApplyFtrl<double>(const size_t size, const double *gradient,
                                                       const double *learning_rate, const double *l1_regularization,
                                                       const double *l2_regularization,
                                                       const double *learning_rate_power, double *variable,
                                                       double *accumulation, double *linear, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ApplyFtrl<int8_t>(const size_t size, const int8_t *gradient,
                                                       const int8_t *learning_rate, const int8_t *l1_regularization,
                                                       const int8_t *l2_regularization,
                                                       const int8_t *learning_rate_power, int8_t *variable,
                                                       int8_t *accumulation, int8_t *linear, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ApplyFtrl<int16_t>(const size_t size, const int16_t *gradient,
                                                        const int16_t *learning_rate, const int16_t *l1_regularization,
                                                        const int16_t *l2_regularization,
                                                        const int16_t *learning_rate_power, int16_t *variable,
                                                        int16_t *accumulation, int16_t *linear,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ApplyFtrl<int64_t>(const size_t size, const int64_t *gradient,
                                                        const int64_t *learning_rate, const int64_t *l1_regularization,
                                                        const int64_t *l2_regularization,
                                                        const int64_t *learning_rate_power, int64_t *variable,
                                                        int64_t *accumulation, int64_t *linear,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ApplyFtrl<uint8_t>(const size_t size, const uint8_t *gradient,
                                                        const uint8_t *learning_rate, const uint8_t *l1_regularization,
                                                        const uint8_t *l2_regularization,
                                                        const uint8_t *learning_rate_power, uint8_t *variable,
                                                        uint8_t *accumulation, uint8_t *linear,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ApplyFtrl<uint16_t>(
  const size_t size, const uint16_t *gradient, const uint16_t *learning_rate, const uint16_t *l1_regularization,
  const uint16_t *l2_regularization, const uint16_t *learning_rate_power, uint16_t *variable, uint16_t *accumulation,
  uint16_t *linear, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ApplyFtrl<uint32_t>(
  const size_t size, const uint32_t *gradient, const uint32_t *learning_rate, const uint32_t *l1_regularization,
  const uint32_t *l2_regularization, const uint32_t *learning_rate_power, uint32_t *variable, uint32_t *accumulation,
  uint32_t *linear, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ApplyFtrl<uint64_t>(
  const size_t size, const uint64_t *gradient, const uint64_t *learning_rate, const uint64_t *l1_regularization,
  const uint64_t *l2_regularization, const uint64_t *learning_rate_power, uint64_t *variable, uint64_t *accumulation,
  uint64_t *linear, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ApplyFtrl<Complex64>(
  const size_t size, const Complex64 *gradient, const Complex64 *learning_rate, const Complex64 *l1_regularization,
  const Complex64 *l2_regularization, const Complex64 *learning_rate_power, Complex64 *variable,
  Complex64 *accumulation, Complex64 *linear, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ApplyFtrl<Complex128>(
  const size_t size, const Complex128 *gradient, const Complex128 *learning_rate, const Complex128 *l1_regularization,
  const Complex128 *l2_regularization, const Complex128 *learning_rate_power, Complex128 *variable,
  Complex128 *accumulation, Complex128 *linear, cudaStream_t cuda_stream);
