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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/fast_gelu_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__global__ void FastGeluKernel(size_t size, T *input_addr, T *output_addr) {
  // formula:
  // fast_gelu(x) = x / (1 + exp(-1.702 * abs(x))) * (exp(0.851 * (x - abs(x))))
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    float x = input_addr[pos];
    float up = std::exp(0.851 * (x - std::abs(x)));
    float down = 1 + std::exp(-1.702 * std::abs(x));
    output_addr[pos] = x / down * up;
  }
}

template <>
__global__ void FastGeluKernel(size_t size, half *input_addr, half *output_addr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    half x = input_addr[pos];
    half up = hexp(half(0.851) * (x - half(std::abs(__half2float(x)))));
    half down = half(1) + hexp(half(-1.702) * half(std::abs(__half2float(x))));
    output_addr[pos] = x / down * up;
  }
}

template <>
__global__ void FastGeluKernel(size_t size, half2 *input_addr, half2 *output_addr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    half2 x = input_addr[pos];

    float2 float2_x = __half22float2(x);
    float2 abs_x_res;
    abs_x_res.x = std::abs(float2_x.x);
    abs_x_res.y = std::abs(float2_x.y);
    half2 half2_x_abs = __float22half2_rn(abs_x_res);

    half2 up = h2exp(half2(0.851, 0.851) * (x - half2_x_abs));
    half2 down = half2(1, 1) + h2exp(half2(-1.702, -1.702) * half2_x_abs);
    output_addr[pos] = x / down * up;
  }
}

template <typename T>
void FastGelu(size_t size, T *input_addr, T *output_addr, cudaStream_t cuda_stream) {
  FastGeluKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input_addr, output_addr);
}

template <>
void FastGelu(size_t size, half *input_addr, half *output_addr, cudaStream_t cuda_stream) {
  if (size % 2 == 0) {
    FastGeluKernel<half2><<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(
      size / 2, reinterpret_cast<half2 *>(input_addr), reinterpret_cast<half2 *>(output_addr));
  } else {
    FastGeluKernel<half><<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input_addr, output_addr);
  }
}

template <typename T>
__global__ void FastGeluGradKernel(size_t size, T *dy_addr, T *x_addr, T *dx_addr) {
  // formula:
  // dx = dy * y'
  // y' = div_up / div_down
  // div_up = exp(-1.702 * x) + 1.702 * x * exp(-1.702 * x) + exp(1.702 * (x - abs(x)))
  // div_down = (exp(-1.702 * x) + 1)^2
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    T x = x_addr[pos];
    T exp_res = std::exp(-1.702 * x);
    T div_up = exp_res + static_cast<T>(1.702) * x * exp_res + static_cast<T>(1);
    T div_down = (exp_res + static_cast<T>(1)) * (exp_res + static_cast<T>(1));
    T y_res = div_up / div_down;
    dx_addr[pos] = dy_addr[pos] * y_res;
  }
}

template <typename T>
__global__ void FastGeluGradKernel(size_t size, half2 *dy_addr, half2 *x_addr, half2 *dx_addr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    half2 x = x_addr[pos];
    half2 exp_res = h2exp(half2(-1.702, -1.702) * x);
    half2 div_up = exp_res + half2(1.702, 1.702) * x * exp_res + half2(1, 1);
    half2 div_down = (exp_res + half2(1, 1)) * (exp_res + half2(1, 1));
    half2 y_res = div_up / div_down;
    dx_addr[pos] = dy_addr[pos] * y_res;
  }
}

template <typename T>
__global__ void FastGeluGradKernel(size_t size, half *dy_addr, half *x_addr, half *dx_addr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    half x = x_addr[pos];
    half exp_res = hexp(half(-1.702) * x);
    half div_up = exp_res + half(1.702) * x * exp_res + half(1);
    half div_down = (exp_res + half(1)) * (exp_res + half(1));
    half y_res = div_up / div_down;
    dx_addr[pos] = dy_addr[pos] * y_res;
  }
}

template <typename T>
void FastGeluGradKernel(size_t size, T *dy_addr, T *x_addr, T *dx_addr, cudaStream_t cuda_stream) {
  FastGeluGradKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, dy_addr, x_addr, dx_addr);
}

template <>
void FastGeluGradKernel(size_t size, half *dy_addr, half *x_addr, half *dx_addr, cudaStream_t cuda_stream) {
  if (size % 2 == 0) {
    FastGeluGradKernel<half2><<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(
      size / 2, reinterpret_cast<half2 *>(dy_addr), reinterpret_cast<half2 *>(x_addr),
      reinterpret_cast<half2 *>(dx_addr));
  } else {
    FastGeluGradKernel<half><<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, dy_addr, x_addr, dx_addr);
  }
}

template CUDA_LIB_EXPORT void FastGelu(size_t size, float *input_addr, float *output_addr, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void FastGelu(size_t size, half *input_addr, half *output_addr, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void FastGeluGradKernel(size_t size, float *dy_addr, float *x_addr, float *dx_addr,
                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void FastGeluGradKernel(size_t size, half *dy_addr, half *x_addr, half *dx_addr,
                                                 cudaStream_t cuda_stream);
