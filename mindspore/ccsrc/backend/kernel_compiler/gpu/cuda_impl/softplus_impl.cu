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

#include "backend/kernel_compiler/gpu/cuda_impl/softplus_impl.cuh"
#include "runtime/device/gpu/cuda_common.h"

template <typename T>
__global__ void SoftplusKernel(const size_t size, const T *input_addr, T *output_addr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    float x = input_addr[pos];
    output_addr[pos] = logf(1. + exp(x));
  }
}

template <>
__global__ void SoftplusKernel(const size_t size, const half *input_addr, half *output_addr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    float x = __half2float(input_addr[pos]);
    output_addr[pos] = __float2half(logf(1. + exp(x)));
  }
}

template <typename T>
void Softplus(const size_t size, const T *input_addr, T *output_addr, cudaStream_t cuda_stream) {
  SoftplusKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input_addr, output_addr);
}

template <>
void Softplus(const size_t size, const half *input_addr, half *output_addr, cudaStream_t cuda_stream) {
  SoftplusKernel<half><<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input_addr, output_addr);
}

template <typename T>
__global__ void SoftplusGradKernel(const size_t size, const T *dy_addr, const T *x_addr, T *dx_addr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    T exp_x = exp(x_addr[pos]);
    dx_addr[pos] = dy_addr[pos] * exp_x / (1. + exp_x);
  }
}

template <typename T>
__global__ void SoftplusGradKernel(const size_t size, const half *dy_addr, const half *x_addr, half *dx_addr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    float x = __half2float(x_addr[pos]);
    float dy = __half2float(dy_addr[pos]);
    float exp_x = exp(x);
    dx_addr[pos] = __float2half(dy * exp_x / (1. + exp_x));
  }
}

template <typename T>
void SoftplusGrad(const size_t size, const T *dy_addr, const T *x_addr, T *dx_addr, cudaStream_t cuda_stream) {
  SoftplusGradKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, dy_addr, x_addr, dx_addr);
}

template <>
void SoftplusGrad(const size_t size, const half *dy_addr, const half *x_addr, half *dx_addr, cudaStream_t cuda_stream) {
  SoftplusGradKernel<half><<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, dy_addr, x_addr, dx_addr);
}

template void Softplus(const size_t size, const float *input_addr, float *output_addr, cudaStream_t cuda_stream);
template void Softplus(const size_t size, const half *input_addr, half *output_addr, cudaStream_t cuda_stream);
template void SoftplusGrad(const size_t size, const float *dy_addr, const float *x_addr, float *dx_addr,
                           cudaStream_t cuda_stream);
template void SoftplusGrad(const size_t size, const half *dy_addr, const half *x_addr, half *dx_addr,
                           cudaStream_t cuda_stream);
