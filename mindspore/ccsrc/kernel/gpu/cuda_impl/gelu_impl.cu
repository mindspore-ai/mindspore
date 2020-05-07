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


#include "kernel/gpu/cuda_impl/gelu_impl.cuh"
#include "device/gpu/cuda_common.h"

template<typename T>
__global__ void GeluKernel(size_t size, T* input_addr, T* output_addr) {
  // formula:
  // gelu(x) = 0.5 * x * (1.0 + tanh(y))
  // tanh(y) = 2 / (1 + exp(-2y)) - 1)
  // y = sqrt(2/pi) * (x + 0.044715 * x^3)
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    float x = input_addr[pos];
    float tanh_res = tanh(0.7978845608 * (x + 0.044715 * x * x * x));
    output_addr[pos] = 0.5 * x * (1.0 + tanh_res);
  }
}

template<typename T>
void Gelu(size_t size, T* input_addr, T* output_addr, cudaStream_t cuda_stream) {
  GeluKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input_addr, output_addr);
  return;
}


template<typename T>
__global__ void GeluGradKernel(size_t size, T* dy_addr, T* x_addr, T* dx_addr) {
  // formula:
  // dx = dy * y'
  // y' = 0.5 * (1 + tanh(tanh_para)) +
  //      0.5 * x * (1 - tanh(tanh_para) * tanh(tanh_para)) * mul_right
  // tanh_para = sqrt(2/pi) * (x + 0.044715 * x^3)
  // mul_right = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2))
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    T x = x_addr[pos];
    T tanh_res = tanh(0.7978845608  * (x + 0.044715 * x * x * x));
    T mul_right = 0.7978845608  + 0.1070322244 * x * x;
    T y_res = 0.5 * (1 + tanh_res) + 0.5 * x * (1 - tanh_res * tanh_res) * mul_right;
    dx_addr[pos] = dy_addr[pos] * y_res;
  }
}

template<typename T>
void GeluGradKernel(size_t size, T* dy_addr, T* x_addr, T* dx_addr, cudaStream_t cuda_stream) {
  GeluGradKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, dy_addr, x_addr, dx_addr);
}


template void Gelu(size_t size, float* input_addr, float* output_addr, cudaStream_t cuda_stream);
template void GeluGradKernel(size_t size, float* dy_addr, float* x_addr, float* dx_addr, cudaStream_t cuda_stream);
