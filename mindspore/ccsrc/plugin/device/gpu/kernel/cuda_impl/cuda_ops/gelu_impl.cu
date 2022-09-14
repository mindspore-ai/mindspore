/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/gelu_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__global__ void GeluKernel(size_t size, const T *input_addr, T *output_addr) {
  // formula:
  // gelu(x) = 0.5 * x * (1.0 + tanh(y))
  // tanh(y) = 2 / (1 + exp(-2y)) - 1)
  // y = sqrt(2/pi) * (x + 0.044715 * x^3)
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    float x = static_cast<float>(input_addr[pos]);
    float tanh_res = tanh(0.7978845608 * (x + 0.044715 * x * x * x));
    output_addr[pos] = 0.5 * x * (1.0 + tanh_res);
  }
}

template <>
__global__ void GeluKernel(size_t size, const half *input_addr, half *output_addr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    half x = input_addr[pos];
    float tanh_res = tanh(__half2float(half(0.7978845608) * (x + half(0.044715) * x * x * x)));
    output_addr[pos] = half(0.5) * x * (half(1.0) + __float2half(tanh_res));
  }
}

template <>
__global__ void GeluKernel(size_t size, const half2 *input_addr, half2 *output_addr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    half2 x = input_addr[pos];
    float2 tanh_param = __half22float2(half2(0.7978845608, 0.7978845608) * (x + half2(0.044715, 0.044715) * x * x * x));
    float2 tanh_res;
    tanh_res.x = tanh(tanh_param.x);
    tanh_res.y = tanh(tanh_param.y);
    output_addr[pos] = half2(0.5, 0.5) * x * (half2(1.0, 1.0) + __float22half2_rn(tanh_res));
  }
}

template <typename T>
void Gelu(size_t size, const T *input_addr, T *output_addr, cudaStream_t cuda_stream, const uint32_t device_id) {
  GeluKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input_addr, output_addr);
}

template <>
void Gelu(size_t size, const half *input_addr, half *output_addr, cudaStream_t cuda_stream, const uint32_t device_id) {
  if (size % 2 == 0) {
    GeluKernel<half2><<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      size / 2, reinterpret_cast<const half2 *>(input_addr), reinterpret_cast<half2 *>(output_addr));
  } else {
    GeluKernel<half>
      <<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input_addr, output_addr);
  }
}

template <typename T>
__global__ void GeluGradKernel(size_t size, T *dy_addr, T *x_addr, T *dx_addr) {
  // formula:
  // dx = dy * y'
  // y' = 0.5 * (1 + tanh(tanh_para)) +
  //      0.5 * x * (1 - tanh(tanh_para) * tanh(tanh_para)) * mul_right
  // tanh_para = sqrt(2/pi) * (x + 0.044715 * x^3)
  // mul_right = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2))
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    T x = x_addr[pos];
    T tanh_res = tanh(0.7978845608 * (x + 0.044715 * x * x * x));
    T mul_right = 0.7978845608 + 0.1070322244 * x * x;
    T y_res = 0.5 * (1.0 + tanh_res) + 0.5 * x * (1.0 - tanh_res * tanh_res) * mul_right;
    dx_addr[pos] = dy_addr[pos] * y_res;
  }
}

template <typename T>
__global__ void GeluGradKernel(size_t size, half2 *dy_addr, half2 *x_addr, half2 *dx_addr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    half2 x = x_addr[pos];
    float2 tanh_param = __half22float2(half2(0.7978845608, 0.7978845608) * (x + half2(0.044715, 0.044715) * x * x * x));
    float2 tanh_res;
    tanh_res.x = tanh(tanh_param.x);
    tanh_res.y = tanh(tanh_param.y);
    half2 tanh_res_half = __float22half2_rn(tanh_res);
    half2 mul_right = half2(0.7978845608, 0.7978845608) + half2(0.1070322244, 0.1070322244) * x * x;
    half2 y_res = half2(0.5, 0.5) * (half2(1.0, 1.0) + tanh_res_half) +
                  half2(0.5, 0.5) * x * (half2(1.0, 1.0) - tanh_res_half * tanh_res_half) * mul_right;
    dx_addr[pos] = dy_addr[pos] * y_res;
  }
}

template <typename T>
__global__ void GeluGradKernel(size_t size, half *dy_addr, half *x_addr, half *dx_addr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    half x = x_addr[pos];
    half tanh_param = half(0.7978845608) * (x + half(0.044715) * x * x * x);
    half tanh_res = __float2half_rn(tanh(__half2float(tanh_param)));
    half mul_right = half(0.7978845608) + half(0.1070322244) * x * x;
    half y_res = half(0.5) * (half(1.0) + tanh_res) + half(0.5) * x * (half(1.0) - tanh_res * tanh_res) * mul_right;
    dx_addr[pos] = dy_addr[pos] * y_res;
  }
}

template <typename T>
void GeluGradKernel(size_t size, T *dy_addr, T *x_addr, T *dx_addr, cudaStream_t cuda_stream,
                    const uint32_t device_id) {
  GeluGradKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, dy_addr, x_addr,
                                                                                            dx_addr);
}

template <>
void GeluGradKernel(size_t size, half *dy_addr, half *x_addr, half *dx_addr, cudaStream_t cuda_stream,
                    const uint32_t device_id) {
  if (size % 2 == 0) {
    GeluGradKernel<half2><<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      size / 2, reinterpret_cast<half2 *>(dy_addr), reinterpret_cast<half2 *>(x_addr),
      reinterpret_cast<half2 *>(dx_addr));
  } else {
    GeluGradKernel<half>
      <<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, dy_addr, x_addr, dx_addr);
  }
}

template CUDA_LIB_EXPORT void Gelu(size_t size, const double *input_addr, double *output_addr, cudaStream_t cuda_stream,
                                   uint32_t device_id);
template CUDA_LIB_EXPORT void Gelu(size_t size, const float *input_addr, float *output_addr, cudaStream_t cuda_stream,
                                   const uint32_t device_id);
template CUDA_LIB_EXPORT void Gelu(size_t size, const half *input_addr, half *output_addr, cudaStream_t cuda_stream,
                                   const uint32_t device_id);
template CUDA_LIB_EXPORT void GeluGradKernel(size_t size, float *dy_addr, float *x_addr, float *dx_addr,
                                             cudaStream_t cuda_stream, const uint32_t device_id);
template CUDA_LIB_EXPORT void GeluGradKernel(size_t size, half *dy_addr, half *x_addr, half *dx_addr,
                                             cudaStream_t cuda_stream, const uint32_t device_id);
