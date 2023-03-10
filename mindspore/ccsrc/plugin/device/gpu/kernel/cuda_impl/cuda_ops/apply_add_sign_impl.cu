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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/apply_add_sign_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__device__ __forceinline__ T Sgn(T x) {
  return static_cast<T>(x != 0 ? (x > 0 ? 1 : -1) : 0);
}

template <typename T, typename S, typename G>
__global__ void ApplyAddSignKernel(const size_t size, T *variable, T *accumulation, const S learning_rate,
                                   const S alpha, const S sign_decay, const S beta, const G *gradient) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    accumulation[i] = (beta * accumulation[i]) + ((static_cast<T>(1.) - beta) * gradient[i]);
    T update = (alpha + (sign_decay * Sgn(gradient[i]) * Sgn(accumulation[i]))) * gradient[i];
    variable[i] = variable[i] - (learning_rate * update);
  }
}

template <>
__global__ void ApplyAddSignKernel(const size_t size, half *variable, half *accumulation, const float learning_rate,
                                   const float alpha, const float sign_decay, const float beta, const half *gradient) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    accumulation[i] =
      (beta * __half2float(accumulation[i])) + ((static_cast<float>(1.) - beta) * __half2float(gradient[i]));
    float update = (alpha + (sign_decay * Sgn(__half2float(gradient[i])) * Sgn(__half2float(accumulation[i])))) *
                   __half2float(gradient[i]);
    variable[i] = __half2float(variable[i]) - (learning_rate * update);
    variable[i] = __float2half(variable[i]);
    accumulation[i] = __float2half(accumulation[i]);
  }
}

template <>
__global__ void ApplyAddSignKernel(const size_t size, float *variable, float *accumulation, const float learning_rate,
                                   const float alpha, const float sign_decay, const float beta, const half *gradient) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    accumulation[i] = (beta * accumulation[i]) + ((static_cast<float>(1.) - beta) * __half2float(gradient[i]));
    float update =
      (alpha + (sign_decay * Sgn(__half2float(gradient[i])) * Sgn(accumulation[i]))) * __half2float(gradient[i]);
    variable[i] = variable[i] - (learning_rate * update);
  }
}

template <>
__global__ void ApplyAddSignKernel(const size_t size, float *variable, float *accumulation, const half learning_rate,
                                   const half alpha, const half sign_decay, const half beta, const float *gradient) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    accumulation[i] =
      (__half2float(beta) * accumulation[i]) + ((static_cast<float>(1.) - __half2float(beta)) * gradient[i]);
    float update =
      (__half2float(alpha) + (__half2float(sign_decay) * Sgn(gradient[i]) * Sgn(accumulation[i]))) * gradient[i];
    variable[i] = variable[i] - (__half2float(learning_rate) * update);
  }
}

template <>
__global__ void ApplyAddSignKernel(const size_t size, float *variable, float *accumulation, const half learning_rate,
                                   const half alpha, const half sign_decay, const half beta, const half *gradient) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    accumulation[i] = (__half2float(beta) * accumulation[i]) +
                      ((static_cast<float>(1.) - __half2float(beta)) * __half2float(gradient[i]));
    float update =
      (__half2float(alpha) + (__half2float(sign_decay) * Sgn(__half2float(gradient[i])) * Sgn(accumulation[i]))) *
      __half2float(gradient[i]);
    variable[i] = variable[i] - __half2float(learning_rate) * update;
  }
}

template <>
__global__ void ApplyAddSignKernel(const size_t size, half *variable, half *accumulation, const half learning_rate,
                                   const half alpha, const half sign_decay, const half beta, const half *gradient) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    accumulation[i] = (__half2float(beta) * __half2float(accumulation[i])) +
                      ((static_cast<float>(1.) - __half2float(beta)) * __half2float(gradient[i]));
    float update = (__half2float(alpha) +
                    (__half2float(sign_decay) * Sgn(__half2float(gradient[i])) * Sgn(__half2float(accumulation[i])))) *
                   __half2float(gradient[i]);
    variable[i] = __float2half(__half2float(variable[i]) - __half2float(learning_rate) * update);
  }
}

template <typename T, typename S, typename G>
cudaError_t ApplyAddSign(const size_t size, T *variable, T *accumulation, const S learning_rate, const S alpha,
                         const S sign_decay, const S beta, const G *gradient, const uint32_t &device_id,
                         cudaStream_t cuda_stream) {
  ApplyAddSignKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    size, variable, accumulation, learning_rate, alpha, sign_decay, beta, gradient);
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template CUDA_LIB_EXPORT cudaError_t ApplyAddSign<double, double, double>(
  const size_t size, double *variable, double *accumulation, const double learning_rate, const double alpha,
  const double sign_decay, const double beta, const double *gradient, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAddSign<float, float, float>(
  const size_t size, float *variable, float *accumulation, const float learning_rate, const float alpha,
  const float sign_decay, const float beta, const float *gradient, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAddSign<float, float, half>(
  const size_t size, float *variable, float *accumulation, const float learning_rate, const float alpha,
  const float sign_decay, const float beta, const half *gradient, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAddSign<float, half, float>(
  const size_t size, float *variable, float *accumulation, const half learning_rate, const half alpha,
  const half sign_decay, const half beta, const float *gradient, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAddSign<float, half, half>(
  const size_t size, float *variable, float *accumulation, const half learning_rate, const half alpha,
  const half sign_decay, const half beta, const half *gradient, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAddSign<half, half, half>(
  const size_t size, half *variable, half *accumulation, const half learning_rate, const half alpha,
  const half sign_decay, const half beta, const half *gradient, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAddSign<half, float, half>(
  const size_t size, half *variable, half *accumulation, const float learning_rate, const float alpha,
  const float sign_decay, const float beta, const half *gradient, const uint32_t &device_id, cudaStream_t cuda_stream);
