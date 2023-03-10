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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/adadelta_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__device__ __forceinline__ T SqrtFunc(T input) {
  return sqrt(input);
}

template <>
__device__ __forceinline__ half SqrtFunc(half input) {
  return hsqrt(input);
}

template <typename T, typename S, typename G>
__global__ void ApplyAdadeltaKernal(const size_t size, const S *learning_rate, const S *rho, const S *epsilon,
                                    const G *gradient, T *variable, T *accumulation, T *accumulation_update) {
  const S one = static_cast<S>(1.0);
  T update = static_cast<T>(0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    accumulation[i] = accumulation[i] * rho[0] + (one - rho[0]) * gradient[i] * gradient[i];
    update = SqrtFunc(accumulation_update[i] + epsilon[0]) * gradient[i] / SqrtFunc(accumulation[i] + epsilon[0]);
    accumulation_update[i] = rho[0] * accumulation_update[i] + (one - rho[0]) * update * update;
    variable[i] -= learning_rate[0] * update;
  }
}

template <>
__global__ void ApplyAdadeltaKernal(const size_t size, const float *learning_rate, const float *rho,
                                    const float *epsilon, const half *gradient, half *variable, half *accumulation,
                                    half *accumulation_update) {
  const float one = static_cast<float>(1.0);
  half update = static_cast<half>(0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    accumulation[i] = accumulation[i] * __float2half(rho[0]) + __float2half(one - rho[0]) * gradient[i] * gradient[i];
    update = SqrtFunc(accumulation_update[i] + __float2half(epsilon[0])) * gradient[i] /
             SqrtFunc(accumulation[i] + __float2half(epsilon[0]));
    accumulation_update[i] =
      __float2half(rho[0]) * accumulation_update[i] + __float2half(one - rho[0]) * update * update;
    variable[i] -= __float2half(learning_rate[0]) * update;
  }
}

template <>
__global__ void ApplyAdadeltaKernal(const size_t size, const float *learning_rate, const float *rho,
                                    const float *epsilon, const half *gradient, float *variable, float *accumulation,
                                    float *accumulation_update) {
  const float one = static_cast<float>(1.0);
  float update = static_cast<float>(0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    accumulation[i] = accumulation[i] * rho[0] + (one - rho[0]) * __half2float(gradient[i]) * __half2float(gradient[i]);
    update = SqrtFunc(accumulation_update[i] + epsilon[0]) * __half2float(gradient[i]) /
             SqrtFunc(accumulation[i] + epsilon[0]);
    accumulation_update[i] = rho[0] * accumulation_update[i] + (one - rho[0]) * update * update;
    variable[i] -= learning_rate[0] * update;
  }
}

template <>
__global__ void ApplyAdadeltaKernal(const size_t size, const half *learning_rate, const half *rho, const half *epsilon,
                                    const float *gradient, float *variable, float *accumulation,
                                    float *accumulation_update) {
  const half one = static_cast<half>(1.0);
  float update = static_cast<float>(0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    accumulation[i] = accumulation[i] * __half2float(rho[0]) + __half2float(one - rho[0]) * gradient[i] * gradient[i];
    update = SqrtFunc(accumulation_update[i] + __half2float(epsilon[0])) * gradient[i] /
             SqrtFunc(accumulation[i] + __half2float(epsilon[0]));
    accumulation_update[i] =
      __half2float(rho[0]) * accumulation_update[i] + __half2float(one - rho[0]) * update * update;
    variable[i] -= __half2float(learning_rate[0]) * update;
  }
}

template <>
__global__ void ApplyAdadeltaKernal(const size_t size, const float *learning_rate, const float *rho,
                                    const float *epsilon, const float *gradient, half *variable, half *accumulation,
                                    half *accumulation_update) {
  const float one = static_cast<float>(1.0);
  half update = static_cast<half>(0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    accumulation[i] = accumulation[i] * __float2half(rho[0]) +
                      __float2half(one - rho[0]) * __float2half(gradient[i]) * __float2half(gradient[i]);
    update = SqrtFunc(accumulation_update[i] + __float2half(epsilon[0])) * __float2half(gradient[i]) /
             SqrtFunc(accumulation[i] + __float2half(epsilon[0]));
    accumulation_update[i] =
      __float2half(rho[0]) * accumulation_update[i] + __float2half(one - rho[0]) * update * update;
    variable[i] -= __float2half(learning_rate[0]) * update;
  }
}

template <>
__global__ void ApplyAdadeltaKernal(const size_t size, const half *learning_rate, const half *rho, const half *epsilon,
                                    const float *gradient, half *variable, half *accumulation,
                                    half *accumulation_update) {
  const half one = static_cast<half>(1.0);
  half update = static_cast<half>(0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    accumulation[i] = accumulation[i] * rho[0] + (one - rho[0]) * __float2half(gradient[i]) * __float2half(gradient[i]);
    update = SqrtFunc(accumulation_update[i] + epsilon[0]) * __float2half(gradient[i]) /
             SqrtFunc(accumulation[i] + epsilon[0]);
    accumulation_update[i] = rho[0] * accumulation_update[i] + (one - rho[0]) * update * update;
    variable[i] -= learning_rate[0] * update;
  }
}

template <>
__global__ void ApplyAdadeltaKernal(const size_t size, const half *learning_rate, const half *rho, const half *epsilon,
                                    const half *gradient, float *variable, float *accumulation,
                                    float *accumulation_update) {
  const half one = static_cast<half>(1.0);
  float update = static_cast<float>(0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    accumulation[i] = accumulation[i] * __half2float(rho[0]) +
                      __half2float(one - rho[0]) * __half2float(gradient[i]) * __half2float(gradient[i]);
    update = SqrtFunc(accumulation_update[i] + __half2float(epsilon[0])) * __half2float(gradient[i]) /
             SqrtFunc(accumulation[i] + __half2float(epsilon[0]));
    accumulation_update[i] =
      __half2float(rho[0]) * accumulation_update[i] + __half2float(one - rho[0]) * update * update;
    variable[i] -= __half2float(learning_rate[0]) * update;
  }
}

template <>
__global__ void ApplyAdadeltaKernal(const size_t size, const half *learning_rate, const half *rho, const half *epsilon,
                                    const half *gradient, double *variable, double *accumulation,
                                    double *accumulation_update) {
  const half one = static_cast<half>(1.0);
  double update = static_cast<double>(0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    accumulation[i] = accumulation[i] * __half2float(rho[0]) +
                      __half2float(one - rho[0]) * __half2float(gradient[i]) * __half2float(gradient[i]);
    update = SqrtFunc(accumulation_update[i] + __half2float(epsilon[0])) * __half2float(gradient[i]) /
             SqrtFunc(accumulation[i] + __half2float(epsilon[0]));
    accumulation_update[i] =
      __half2float(rho[0]) * accumulation_update[i] + __half2float(one - rho[0]) * update * update;
    variable[i] -= __half2float(learning_rate[0]) * update;
  }
}

template <>
__global__ void ApplyAdadeltaKernal(const size_t size, const float *learning_rate, const float *rho,
                                    const float *epsilon, const float *gradient, double *variable, double *accumulation,
                                    double *accumulation_update) {
  const float one = static_cast<float>(1.0);
  double update = static_cast<double>(0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    accumulation[i] = accumulation[i] * rho[0] + (one - rho[0]) * gradient[i] * gradient[i];
    update = SqrtFunc(accumulation_update[i] + epsilon[0]) * gradient[i] / SqrtFunc(accumulation[i] + epsilon[0]);
    accumulation_update[i] = rho[0] * accumulation_update[i] + (one - rho[0]) * update * update;
    variable[i] -= learning_rate[0] * update;
  }
}

template <>
__global__ void ApplyAdadeltaKernal(const size_t size, const float *learning_rate, const float *rho,
                                    const float *epsilon, const double *gradient, double *variable,
                                    double *accumulation, double *accumulation_update) {
  const float one = static_cast<float>(1.0);
  double update = static_cast<double>(0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    accumulation[i] = accumulation[i] * rho[0] + (one - rho[0]) * gradient[i] * gradient[i];
    update = SqrtFunc(accumulation_update[i] + epsilon[0]) * gradient[i] / SqrtFunc(accumulation[i] + epsilon[0]);
    accumulation_update[i] = rho[0] * accumulation_update[i] + (one - rho[0]) * update * update;
    variable[i] -= learning_rate[0] * update;
  }
}

template <>
__global__ void ApplyAdadeltaKernal(const size_t size, const half *learning_rate, const half *rho, const half *epsilon,
                                    const double *gradient, double *variable, double *accumulation,
                                    double *accumulation_update) {
  const half one = static_cast<half>(1.0);
  double update = static_cast<double>(0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    accumulation[i] = accumulation[i] * __half2float(rho[0]) + __half2float(one - rho[0]) * gradient[i] * gradient[i];
    update = SqrtFunc(accumulation_update[i] + __half2float(epsilon[0])) * gradient[i] /
             SqrtFunc(accumulation[i] + __half2float(epsilon[0]));
    accumulation_update[i] =
      __half2float(rho[0]) * accumulation_update[i] + __half2float(one - rho[0]) * update * update;
    variable[i] -= __half2float(learning_rate[0]) * update;
  }
}

template <typename T, typename S, typename G>
cudaError_t ApplyAdadelta(const size_t size, const S *learning_rate, const S *rho, const S *epsilon, const G *gradient,
                          T *variable, T *accumulation, T *accumulation_update, const uint32_t &device_id,
                          cudaStream_t cuda_stream) {
  ApplyAdadeltaKernal<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    size, learning_rate, rho, epsilon, gradient, variable, accumulation, accumulation_update);
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template CUDA_LIB_EXPORT cudaError_t ApplyAdadelta<double, double, double>(
  const size_t size, const double *learning_rate, const double *rho, const double *epsilon, const double *gradient,
  double *variable, double *accumulation, double *accumulation_update, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdadelta<float, float, float>(const size_t size, const float *learning_rate,
                                                                        const float *rho, const float *epsilon,
                                                                        const float *gradient, float *variable,
                                                                        float *accumulation, float *accumulation_update,
                                                                        const uint32_t &device_id,
                                                                        cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdadelta<half, half, half>(
  const size_t size, const half *learning_rate, const half *rho, const half *epsilon, const half *gradient,
  half *variable, half *accumulation, half *accumulation_update, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdadelta<half, float, half>(
  const size_t size, const float *learning_rate, const float *rho, const float *epsilon, const half *gradient,
  half *variable, half *accumulation, half *accumulation_update, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdadelta<float, float, half>(const size_t size, const float *learning_rate,
                                                                       const float *rho, const float *epsilon,
                                                                       const half *gradient, float *variable,
                                                                       float *accumulation, float *accumulation_update,
                                                                       const uint32_t &device_id,
                                                                       cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdadelta<half, half, float>(
  const size_t size, const half *learning_rate, const half *rho, const half *epsilon, const float *gradient,
  half *variable, half *accumulation, half *accumulation_update, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdadelta<float, half, float>(const size_t size, const half *learning_rate,
                                                                       const half *rho, const half *epsilon,
                                                                       const float *gradient, float *variable,
                                                                       float *accumulation, float *accumulation_update,
                                                                       const uint32_t &device_id,
                                                                       cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdadelta<half, float, float>(
  const size_t size, const float *learning_rate, const float *rho, const float *epsilon, const float *gradient,
  half *variable, half *accumulation, half *accumulation_update, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdadelta<float, half, half>(const size_t size, const half *learning_rate,
                                                                      const half *rho, const half *epsilon,
                                                                      const half *gradient, float *variable,
                                                                      float *accumulation, float *accumulation_update,
                                                                      const uint32_t &device_id,
                                                                      cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t
ApplyAdadelta<double, half, half>(const size_t size, const half *learning_rate, const half *rho, const half *epsilon,
                                  const half *gradient, double *variable, double *accumulation,
                                  double *accumulation_update, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t
ApplyAdadelta<double, half, double>(const size_t size, const half *learning_rate, const half *rho, const half *epsilon,
                                    const double *gradient, double *variable, double *accumulation,
                                    double *accumulation_update, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdadelta<double, float, double>(
  const size_t size, const float *learning_rate, const float *rho, const float *epsilon, const double *gradient,
  double *variable, double *accumulation, double *accumulation_update, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t
ApplyAdadelta<double, float, float>(const size_t size, const float *learning_rate, const float *rho,
                                    const float *epsilon, const float *gradient, double *variable, double *accumulation,
                                    double *accumulation_update, const uint32_t &device_id, cudaStream_t cuda_stream);
