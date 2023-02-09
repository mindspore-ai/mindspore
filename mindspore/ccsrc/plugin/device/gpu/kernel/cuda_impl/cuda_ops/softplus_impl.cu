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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/softplus_impl.cuh"
#include <math.h>
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

// Refactor the softplus to improve numeric stable.
// The previous version result overflow when input slightly larger.
template <typename T>
__global__ void SoftplusKernel(const size_t size, const T threshold, const T *input_addr, T *output_addr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    T x = input_addr[pos];
    output_addr[pos] = x > -threshold ? x : (x < threshold ? exp(x) : log(1. + exp(x)));
  }
}

template <>
__global__ void SoftplusKernel(const size_t size, const half threshold, const half *input_addr, half *output_addr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    const half one = 1;
    const half threshold_in_half = threshold;
    half x = input_addr[pos];
    output_addr[pos] = x > -threshold_in_half ? x : (x < threshold_in_half ? hexp(x) : hlog(one + hexp(x)));
  }
}

template <typename T>
void Softplus(const size_t size, const T *input_addr, T *output_addr, cudaStream_t cuda_stream) {
  const T threshold = log(Epsilon<T>::value) + 2.0;
  SoftplusKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, threshold, input_addr, output_addr);
}

template <typename T>
__global__ void SoftplusGradKernel(const size_t size, const T threshold, const T *dy_addr, const T *x_addr,
                                   T *dx_addr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    const T one = 1;
    T x = x_addr[pos];
    dx_addr[pos] = dy_addr[pos] * (x > -threshold ? one : (x < threshold ? exp(x) : one / (one + exp(-x))));
  }
}

template <>
__global__ void SoftplusGradKernel(const size_t size, const half threshold, const half *dy_addr, const half *x_addr,
                                   half *dx_addr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    const half one = 1;
    const half threshold_in_half = threshold;
    half x = x_addr[pos];
    dx_addr[pos] = dy_addr[pos] * (x > -threshold ? one : (x < threshold ? hexp(x) : one / (one + hexp(-x))));
  }
}

template <typename T>
void SoftplusGrad(const size_t size, const T *dy_addr, const T *x_addr, T *dx_addr, cudaStream_t cuda_stream) {
  const T threshold = log(Epsilon<T>::value) + 2.0f;
  SoftplusGradKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, threshold, dy_addr, x_addr, dx_addr);
}
template CUDA_LIB_EXPORT void Softplus(const size_t size, const double *input_addr, double *output_addr,
                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Softplus(const size_t size, const float *input_addr, float *output_addr,
                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Softplus(const size_t size, const half *input_addr, half *output_addr,
                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SoftplusGrad(const size_t size, const double *dy_addr, const double *x_addr,
                                           double *dx_addr, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SoftplusGrad(const size_t size, const float *dy_addr, const float *x_addr, float *dx_addr,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SoftplusGrad(const size_t size, const half *dy_addr, const half *x_addr, half *dx_addr,
                                           cudaStream_t cuda_stream);
