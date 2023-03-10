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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/adamax_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__device__ __forceinline__ T SqrtFunc(T input) {
  return sqrt(input);
}

template <>
__device__ __forceinline__ half SqrtFunc(half input) {
  return hsqrt(input);
}

template <typename T>
__device__ __forceinline__ T MaxFunc(T input1, T input2) {
  return (input1 > input2) ? input1 : input2;
}

template <typename T>
__device__ __forceinline__ T AbsFunc(T input) {
  const T zero = static_cast<T>(0);
  return (input >= zero) ? input : -input;
}

template <typename T, typename S, typename G>
__global__ void ApplyAdamaxKernal(const size_t size, const S *b1_power, const S *learning_rate, const S *b1,
                                  const S *b2, const S *eps, const G *gradient, T *variable, T *m, T *v) {
  const S one = static_cast<S>(1.0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    m[i] = b1[0] * m[i] + (one - b1[0]) * gradient[i];
    v[i] = MaxFunc(b2[0] * v[i], AbsFunc(gradient[i]));
    variable[i] -= learning_rate[0] * m[i] / (one - b1_power[0]) / (v[i] + eps[0]);
  }
}

template <>
__global__ void ApplyAdamaxKernal(const size_t size, const float *b1_power, const float *learning_rate, const float *b1,
                                  const float *b2, const float *eps, const half *gradient, half *variable, half *m,
                                  half *v) {
  const float one = static_cast<float>(1.0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    m[i] = __float2half(b1[0]) * m[i] + __float2half(one - b1[0]) * gradient[i];
    v[i] = MaxFunc(__float2half(b2[0]) * v[i], AbsFunc(gradient[i]));
    variable[i] -=
      __float2half(learning_rate[0]) * m[i] / __float2half(one - b1_power[0]) / (v[i] + __float2half(eps[0]));
  }
}

template <>
__global__ void ApplyAdamaxKernal(const size_t size, const float *b1_power, const float *learning_rate, const float *b1,
                                  const float *b2, const float *eps, const half *gradient, float *variable, float *m,
                                  float *v) {
  const float one = static_cast<float>(1.0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    m[i] = b1[0] * m[i] + (one - b1[0]) * __half2float(gradient[i]);
    v[i] = MaxFunc(b2[0] * v[i], AbsFunc(__half2float(gradient[i])));
    variable[i] -= learning_rate[0] * m[i] / (one - b1_power[0]) / (v[i] + eps[0]);
  }
}

template <>
__global__ void ApplyAdamaxKernal(const size_t size, const half *b1_power, const half *learning_rate, const half *b1,
                                  const half *b2, const half *eps, const float *gradient, float *variable, float *m,
                                  float *v) {
  const half one = static_cast<half>(1.0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    m[i] = __half2float(b1[0]) * m[i] + __half2float(one - b1[0]) * gradient[i];
    v[i] = MaxFunc(__half2float(b2[0]) * v[i], AbsFunc(gradient[i]));
    variable[i] -=
      __half2float(learning_rate[0]) * m[i] / __half2float(one - b1_power[0]) / (v[i] + __half2float(eps[0]));
  }
}

template <>
__global__ void ApplyAdamaxKernal(const size_t size, const half *b1_power, const half *learning_rate, const half *b1,
                                  const half *b2, const half *eps, const double *gradient, double *variable, double *m,
                                  double *v) {
  const half one = static_cast<half>(1.0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    m[i] = __half2float(b1[0]) * m[i] + __half2float(one - b1[0]) * gradient[i];
    v[i] = MaxFunc(__half2float(b2[0]) * v[i], AbsFunc(gradient[i]));
    variable[i] -=
      __half2float(learning_rate[0]) * m[i] / __half2float(one - b1_power[0]) / (v[i] + __half2float(eps[0]));
  }
}

template <>
__global__ void ApplyAdamaxKernal(const size_t size, const float *b1_power, const float *learning_rate, const float *b1,
                                  const float *b2, const float *eps, const double *gradient, double *variable,
                                  double *m, double *v) {
  const float one = static_cast<float>(1.0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    m[i] = b1[0] * m[i] + (one - b1[0]) * gradient[i];
    v[i] = MaxFunc(b2[0] * v[i], AbsFunc(gradient[i]));
    variable[i] -= learning_rate[0] * m[i] / (one - b1_power[0]) / (v[i] + eps[0]);
  }
}

template <>
__global__ void ApplyAdamaxKernal(const size_t size, const float *b1_power, const float *learning_rate, const float *b1,
                                  const float *b2, const float *eps, const float *gradient, half *variable, half *m,
                                  half *v) {
  const float one = static_cast<float>(1.0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    m[i] = __float2half(b1[0]) * m[i] + __float2half(one - b1[0]) * __float2half(gradient[i]);
    v[i] = MaxFunc(__float2half(b2[0]) * v[i], AbsFunc(__float2half(gradient[i])));
    variable[i] -=
      __float2half(learning_rate[0]) * m[i] / __float2half(one - b1_power[0]) / (v[i] + __float2half(eps[0]));
  }
}

template <>
__global__ void ApplyAdamaxKernal(const size_t size, const float *b1_power, const float *learning_rate, const float *b1,
                                  const float *b2, const float *eps, const float *gradient, double *variable, double *m,
                                  double *v) {
  const float one = static_cast<float>(1.0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    m[i] = b1[0] * m[i] + (one - b1[0]) * gradient[i];
    v[i] = MaxFunc(b2[0] * v[i], static_cast<double>(AbsFunc(gradient[i])));
    variable[i] -= learning_rate[0] * m[i] / (one - b1_power[0]) / (v[i] + eps[0]);
  }
}

template <>
__global__ void ApplyAdamaxKernal(const size_t size, const half *b1_power, const half *learning_rate, const half *b1,
                                  const half *b2, const half *eps, const half *gradient, double *variable, double *m,
                                  double *v) {
  const half one = static_cast<half>(1.0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    m[i] = __half2float(b1[0]) * m[i] + __half2float(one - b1[0]) * __half2float(gradient[i]);
    v[i] = MaxFunc(__half2float(b2[0]) * v[i], static_cast<double>(AbsFunc(__half2float(gradient[i]))));
    variable[i] -=
      __half2float(learning_rate[0]) * m[i] / __half2float(one - b1_power[0]) / (v[i] + __half2float(eps[0]));
  }
}

template <>
__global__ void ApplyAdamaxKernal(const size_t size, const half *b1_power, const half *learning_rate, const half *b1,
                                  const half *b2, const half *eps, const half *gradient, float *variable, float *m,
                                  float *v) {
  const half one = static_cast<half>(1.0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    m[i] = __half2float(b1[0]) * m[i] + __half2float(one - b1[0]) * __half2float(gradient[i]);
    v[i] = MaxFunc(__half2float(b2[0]) * v[i], AbsFunc(__half2float(gradient[i])));
    variable[i] -=
      __half2float(learning_rate[0]) * m[i] / __half2float(one - b1_power[0]) / (v[i] + __half2float(eps[0]));
  }
}

template <typename T, typename S, typename G>
cudaError_t ApplyAdamax(const size_t size, const S *b1_power, const S *learning_rate, const S *b1, const S *b2,
                        const S *eps, const G *gradient, T *variable, T *m, T *v, const uint32_t &device_id,
                        cudaStream_t cuda_stream) {
  ApplyAdamaxKernal<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    size, b1_power, learning_rate, b1, b2, eps, gradient, variable, m, v);
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template CUDA_LIB_EXPORT cudaError_t ApplyAdamax<double, double, double>(
  const size_t size, const double *b1_power, const double *learning_rate, const double *b1, const double *b2,
  const double *eps, const double *gradient, double *variable, double *m, double *v, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdamax<float, float, float>(const size_t size, const float *b1_power,
                                                                      const float *learning_rate, const float *b1,
                                                                      const float *b2, const float *eps,
                                                                      const float *gradient, float *variable, float *m,
                                                                      float *v, const uint32_t &device_id,
                                                                      cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdamax<half, half, half>(
  const size_t size, const half *b1_power, const half *learning_rate, const half *b1, const half *b2, const half *eps,
  const half *gradient, half *variable, half *m, half *v, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdamax<half, float, half>(const size_t size, const float *b1_power,
                                                                    const float *learning_rate, const float *b1,
                                                                    const float *b2, const float *eps,
                                                                    const half *gradient, half *variable, half *m,
                                                                    half *v, const uint32_t &device_id,
                                                                    cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdamax<float, float, half>(const size_t size, const float *b1_power,
                                                                     const float *learning_rate, const float *b1,
                                                                     const float *b2, const float *eps,
                                                                     const half *gradient, float *variable, float *m,
                                                                     float *v, const uint32_t &device_id,
                                                                     cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdamax<float, half, float>(
  const size_t size, const half *b1_power, const half *learning_rate, const half *b1, const half *b2, const half *eps,
  const float *gradient, float *variable, float *m, float *v, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdamax<half, float, float>(const size_t size, const float *b1_power,
                                                                     const float *learning_rate, const float *b1,
                                                                     const float *b2, const float *eps,
                                                                     const float *gradient, half *variable, half *m,
                                                                     half *v, const uint32_t &device_id,
                                                                     cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdamax<float, half, half>(
  const size_t size, const half *b1_power, const half *learning_rate, const half *b1, const half *b2, const half *eps,
  const half *gradient, float *variable, float *m, float *v, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdamax<double, half, half>(
  const size_t size, const half *b1_power, const half *learning_rate, const half *b1, const half *b2, const half *eps,
  const half *gradient, double *variable, double *m, double *v, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdamax<double, half, double>(
  const size_t size, const half *b1_power, const half *learning_rate, const half *b1, const half *b2, const half *eps,
  const double *gradient, double *variable, double *m, double *v, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdamax<double, float, double>(const size_t size, const float *b1_power,
                                                                        const float *learning_rate, const float *b1,
                                                                        const float *b2, const float *eps,
                                                                        const double *gradient, double *variable,
                                                                        double *m, double *v, const uint32_t &device_id,
                                                                        cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdamax<double, float, float>(const size_t size, const float *b1_power,
                                                                       const float *learning_rate, const float *b1,
                                                                       const float *b2, const float *eps,
                                                                       const float *gradient, double *variable,
                                                                       double *m, double *v, const uint32_t &device_id,
                                                                       cudaStream_t cuda_stream);
