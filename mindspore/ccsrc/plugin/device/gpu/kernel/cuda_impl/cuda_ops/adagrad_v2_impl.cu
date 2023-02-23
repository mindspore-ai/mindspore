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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/adagrad_v2_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__device__ __forceinline__ T SqrtFunc(T input) {
  return sqrt(input);
}

template <>
__device__ __forceinline__ half SqrtFunc(half input) {
  return hsqrt(input);
}

template <typename T, typename S>
__global__ void ApplyAdagradV2Kernel(const size_t size, const float epsilon, T *variable, T *accumulation,
                                     const S *learning_rate, const T *gradient) {
  T grad = static_cast<T>(0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    grad = gradient[i];
    accumulation[i] += grad * grad;
    variable[i] -= learning_rate[0] * grad / (SqrtFunc(accumulation[i] + epsilon));
  }
}

template <>
__global__ void ApplyAdagradV2Kernel(const size_t size, const float epsilon, half *variable, half *accumulation,
                                     const half *learning_rate, const half *gradient) {
  half grad = static_cast<half>(0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    grad = gradient[i];
    accumulation[i] += grad * grad;
    variable[i] -= learning_rate[0] * grad / (SqrtFunc(accumulation[i] + __float2half(epsilon)));
  }
}

template <>
__global__ void ApplyAdagradV2Kernel(const size_t size, const float epsilon, half *variable, half *accumulation,
                                     const float *learning_rate, const half *gradient) {
  half grad = static_cast<half>(0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    grad = gradient[i];
    accumulation[i] += grad * grad;
    variable[i] -= __float2half(learning_rate[0]) * grad / (SqrtFunc(accumulation[i] + __float2half(epsilon)));
  }
}

template <>
__global__ void ApplyAdagradV2Kernel(const size_t size, const float epsilon, half *variable, half *accumulation,
                                     const double *learning_rate, const half *gradient) {
  half grad = static_cast<half>(0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    grad = gradient[i];
    accumulation[i] += grad * grad;
    variable[i] -= __float2half(learning_rate[0]) * grad / (SqrtFunc(accumulation[i] + __float2half(epsilon)));
  }
}

template <>
__global__ void ApplyAdagradV2Kernel(const size_t size, const float epsilon, double *variable, double *accumulation,
                                     const half *learning_rate, const double *gradient) {
  double grad = static_cast<double>(0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    grad = gradient[i];
    accumulation[i] += grad * grad;
    variable[i] -= __half2float(learning_rate[0]) * grad / (SqrtFunc(accumulation[i] + epsilon));
  }
}

template <>
__global__ void ApplyAdagradV2Kernel(const size_t size, const float epsilon, double *variable, double *accumulation,
                                     const float *learning_rate, const double *gradient) {
  double grad = static_cast<double>(0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    grad = gradient[i];
    accumulation[i] += grad * grad;
    variable[i] -= learning_rate[0] * grad / (SqrtFunc(accumulation[i] + epsilon));
  }
}

template <>
__global__ void ApplyAdagradV2Kernel(const size_t size, const float epsilon, float *variable, float *accumulation,
                                     const half *learning_rate, const float *gradient) {
  float grad = static_cast<float>(0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    grad = gradient[i];
    accumulation[i] += grad * grad;
    variable[i] -= __half2float(learning_rate[0]) * grad / (SqrtFunc(accumulation[i] + epsilon));
  }
}

template <>
__global__ void ApplyAdagradV2Kernel(const size_t size, const float epsilon, float *variable, float *accumulation,
                                     const double *learning_rate, const float *gradient) {
  float grad = static_cast<float>(0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    grad = gradient[i];
    accumulation[i] += grad * grad;
    variable[i] -= learning_rate[0] * grad / (SqrtFunc(accumulation[i] + epsilon));
  }
}

template <typename T, typename S>
__global__ void ApplyAdagradV2Kernel_(const size_t size, const float epsilon, T *variable, T *accumulation,
                                      const S *learning_rate, const T *gradient) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    variable[i] -= learning_rate[0] * gradient[i] / (SqrtFunc(accumulation[i] + epsilon));
  }
}

template <>
__global__ void ApplyAdagradV2Kernel_(const size_t size, const float epsilon, half *variable, half *accumulation,
                                      const half *learning_rate, const half *gradient) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    variable[i] -= learning_rate[0] * gradient[i] / (SqrtFunc(accumulation[i] + __float2half(epsilon)));
  }
}

template <>
__global__ void ApplyAdagradV2Kernel_(const size_t size, const float epsilon, half *variable, half *accumulation,
                                      const float *learning_rate, const half *gradient) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    variable[i] -= __float2half(learning_rate[0]) * gradient[i] / (SqrtFunc(accumulation[i] + __float2half(epsilon)));
  }
}

template <>
__global__ void ApplyAdagradV2Kernel_(const size_t size, const float epsilon, half *variable, half *accumulation,
                                      const double *learning_rate, const half *gradient) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    variable[i] -= __float2half(learning_rate[0]) * gradient[i] / (SqrtFunc(accumulation[i] + __float2half(epsilon)));
  }
}

template <>
__global__ void ApplyAdagradV2Kernel_(const size_t size, const float epsilon, double *variable, double *accumulation,
                                      const half *learning_rate, const double *gradient) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    variable[i] -= __half2float(learning_rate[0]) * gradient[i] / (SqrtFunc(accumulation[i] + epsilon));
  }
}

template <>
__global__ void ApplyAdagradV2Kernel_(const size_t size, const float epsilon, double *variable, double *accumulation,
                                      const float *learning_rate, const double *gradient) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    variable[i] -= learning_rate[0] * gradient[i] / (SqrtFunc(accumulation[i] + epsilon));
  }
}

template <>
__global__ void ApplyAdagradV2Kernel_(const size_t size, const float epsilon, float *variable, float *accumulation,
                                      const half *learning_rate, const float *gradient) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    variable[i] -= __half2float(learning_rate[0]) * gradient[i] / (SqrtFunc(accumulation[i] + epsilon));
  }
}

template <>
__global__ void ApplyAdagradV2Kernel_(const size_t size, const float epsilon, float *variable, float *accumulation,
                                      const double *learning_rate, const float *gradient) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    variable[i] -= learning_rate[0] * gradient[i] / (SqrtFunc(accumulation[i] + epsilon));
  }
}

template <typename T, typename S>
cudaError_t ApplyAdagradV2(const size_t size, const float epsilon, const bool update_slots, T *variable,
                           T *accumulation, const S *learning_rate, const T *gradient, const uint32_t &device_id,
                           cudaStream_t cuda_stream) {
  if (update_slots) {
    ApplyAdagradV2Kernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      size, epsilon, variable, accumulation, learning_rate, gradient);
  } else {
    ApplyAdagradV2Kernel_<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      size, epsilon, variable, accumulation, learning_rate, gradient);
  }
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template CUDA_LIB_EXPORT cudaError_t ApplyAdagradV2<double, double>(const size_t size, const float epsilon,
                                                                    const bool update_slots, double *variable,
                                                                    double *accumulation, const double *learning_rate,
                                                                    const double *gradient, const uint32_t &device_id,
                                                                    cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdagradV2<float, float>(const size_t size, const float epsilon,
                                                                  const bool update_slots, float *variable,
                                                                  float *accumulation, const float *learning_rate,
                                                                  const float *gradient, const uint32_t &device_id,
                                                                  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdagradV2<half, half>(const size_t size, const float epsilon,
                                                                const bool update_slots, half *variable,
                                                                half *accumulation, const half *learning_rate,
                                                                const half *gradient, const uint32_t &device_id,
                                                                cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdagradV2<float, half>(const size_t size, const float epsilon,
                                                                 const bool update_slots, float *variable,
                                                                 float *accumulation, const half *learning_rate,
                                                                 const float *gradient, const uint32_t &device_id,
                                                                 cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdagradV2<half, float>(const size_t size, const float epsilon,
                                                                 const bool update_slots, half *variable,
                                                                 half *accumulation, const float *learning_rate,
                                                                 const half *gradient, const uint32_t &device_id,
                                                                 cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdagradV2<half, double>(const size_t size, const float epsilon,
                                                                  const bool update_slots, half *variable,
                                                                  half *accumulation, const double *learning_rate,
                                                                  const half *gradient, const uint32_t &device_id,
                                                                  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdagradV2<double, float>(const size_t size, const float epsilon,
                                                                   const bool update_slots, double *variable,
                                                                   double *accumulation, const float *learning_rate,
                                                                   const double *gradient, const uint32_t &device_id,
                                                                   cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdagradV2<double, half>(const size_t size, const float epsilon,
                                                                  const bool update_slots, double *variable,
                                                                  double *accumulation, const half *learning_rate,
                                                                  const double *gradient, const uint32_t &device_id,
                                                                  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdagradV2<float, double>(const size_t size, const float epsilon,
                                                                   const bool update_slots, float *variable,
                                                                   float *accumulation, const double *learning_rate,
                                                                   const float *gradient, const uint32_t &device_id,
                                                                   cudaStream_t cuda_stream);
