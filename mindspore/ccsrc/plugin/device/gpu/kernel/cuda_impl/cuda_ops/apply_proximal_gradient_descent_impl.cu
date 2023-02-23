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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/apply_proximal_gradient_descent_impl.cuh"
#include <algorithm>
#include "include/cuda_fp16.h"

template <typename T>
__device__ __forceinline__ T RsqrtFunc(T x) {
  return rsqrt(x);
}

template <>
__device__ __forceinline__ half RsqrtFunc(half x) {
  return hrsqrt(x);
}

template <typename T>
__device__ __forceinline__ T AbsFunc(T x) {
  return abs(x);
}

template <>
__device__ __forceinline__ half AbsFunc(half x) {
  return abs(__half2float(x));
}

template <typename T>
__device__ __forceinline__ T MaxFunc(T x, T y) {
  return max(x, y);
}

template <>
__device__ __forceinline__ half MaxFunc(half x, half y) {
  return max(__half2float(x), __half2float(y));
}

template <typename T>
__device__ __forceinline__ T SgnFunc(T x) {
  return static_cast<T>(x != 0 ? (x > 0 ? 1 : -1) : 0);
}

template <>
__device__ __forceinline__ half SgnFunc(half x) {
  return __float2half(__half2float(x) != 0 ? (__half2float(x) > 0 ? 1 : -1) : 0);
}

template <typename T>
__global__ void CalApplyProximalGradientDescentKernel(const size_t input_elements, T *var, const T *alpha, const T *l1,
                                                      const T *l2, const T *delta, T *output) {
  if (l1[0] > static_cast<T>(0.0)) {
    for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < static_cast<int>(input_elements);
         pos += gridDim.x * blockDim.x) {
      auto prox_v = var[pos];
      prox_v -= delta[pos] * alpha[0];
      var[pos] = SgnFunc(prox_v) * MaxFunc(AbsFunc(prox_v) - alpha[0] * l1[0], static_cast<T>(0.0)) /
                 (static_cast<T>(1) + l2[0] * alpha[0]);
    }
  } else {
    for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < static_cast<int>(input_elements);
         pos += gridDim.x * blockDim.x) {
      auto prox_v = var[pos];
      prox_v -= delta[pos] * alpha[0];
      var[pos] = prox_v / (static_cast<T>(1) + l2[0] * alpha[0]);
    }
  }
}

template <typename T>
cudaError_t CalApplyProximalGradientDescent(const size_t input_elements, T *var, const T *alpha, const T *l1,
                                            const T *l2, const T *delta, T *output, const uint32_t &device_id,
                                            cudaStream_t cuda_stream) {
  CalApplyProximalGradientDescentKernel<<<CUDA_BLOCKS(device_id, input_elements), CUDA_THREADS(device_id), 0,
                                          cuda_stream>>>(input_elements, var, alpha, l1, l2, delta, output);
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template CUDA_LIB_EXPORT cudaError_t CalApplyProximalGradientDescent<float>(const size_t size, float *var,
                                                                            const float *alpha, const float *l1,
                                                                            const float *l2, const float *delta,
                                                                            float *output, const uint32_t &device_id,
                                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalApplyProximalGradientDescent<half>(const size_t size, half *var,
                                                                           const half *alpha, const half *l1,
                                                                           const half *l2, const half *delta,
                                                                           half *output, const uint32_t &device_id,
                                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalApplyProximalGradientDescent<double>(const size_t size, double *var,
                                                                             const double *alpha, const double *l1,
                                                                             const double *l2, const double *delta,
                                                                             double *output, const uint32_t &device_id,
                                                                             cudaStream_t cuda_stream);
