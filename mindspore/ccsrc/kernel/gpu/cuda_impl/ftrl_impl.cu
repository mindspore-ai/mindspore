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

#include "kernel/gpu/cuda_impl/ftrl_impl.cuh"

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
void ApplyFtrl(const size_t size, const T *gradient, const T *learning_rate, const T *l1_regularization,
               const T *l2_regularization, const T *learning_rate_power, T *variable, T *accumulation, T *linear,
               cudaStream_t cuda_stream) {
  ApplyFtrlKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, gradient, learning_rate, l1_regularization,
                                                                     l2_regularization, learning_rate_power, variable,
                                                                     accumulation, linear);
}

template void ApplyFtrl<float>(const size_t size, const float *gradient, const float *learning_rate,
                               const float *l1_regularization, const float *l2_regularization,
                               const float *learning_rate_power, float *variable, float *accumulation, float *linear,
                               cudaStream_t cuda_stream);
template void ApplyFtrl<half>(const size_t size, const half *gradient, const half *learning_rate,
                              const half *l1_regularization, const half *l2_regularization,
                              const half *learning_rate_power, half *variable, half *accumulation, half *linear,
                              cudaStream_t cuda_stream);
