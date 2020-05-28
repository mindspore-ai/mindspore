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

#include "kernel/gpu/cuda_impl/tanh_impl.cuh"
#include <cuda_runtime.h>

template<typename T>
__global__ void TanhKernel(const size_t size, const T* x_addr, T* y_addr) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    y_addr[pos] = tanh(x_addr[pos]);
  }
}

template<typename T>
__global__ void TanhGradKernel(const size_t size, const T* y_addr, const T* dy_addr, T* dx_addr) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    dx_addr[pos] = dy_addr[pos] * (1 - y_addr[pos] * y_addr[pos]);
  }
}

template<typename T>
void Tanh(const size_t size, const T* x_addr, T* y_addr, cudaStream_t cuda_stream) {
  TanhKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, x_addr, y_addr);
}

template<typename T>
void TanhGrad(const size_t size, const T* y_addr, const T* dy_addr, T* dx_addr, cudaStream_t cuda_stream) {
  TanhGradKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, y_addr, dy_addr, dx_addr);
}

template void Tanh(const size_t size, const float* x_addr, float* y_addr, cudaStream_t cuda_stream);
template void TanhGrad(const size_t size, const float* y_addr, const float* dy_addr,
                       float* dx_addr, cudaStream_t cuda_stream);
