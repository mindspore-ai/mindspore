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

#include "backend/kernel_compiler/gpu/cuda_impl/square_sum_all_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/util.cuh"

template <typename T>
__global__ void SquareSumAllKernel(const size_t size, const T* input_addr_0, const T* input_addr_1,
                                   float* ws_addr_0, float* ws_addr_1) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    size_t split = size / 2;
    float power = 2.0;
    if (i < split) {
      float ret = powf(__half2float(input_addr_0[i]), power);
      MsAtomicAdd(ws_addr_0, ret);
    } else {
      float ret = powf(__half2float(input_addr_1[i - split]), power);
      MsAtomicAdd(ws_addr_1, ret);
    }
  }
  return;
}

template <>
__global__ void SquareSumAllKernel(const size_t size, const float* input_addr_0, const float* input_addr_1,
                                   float* ws_addr_0, float* ws_addr_1) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    size_t split = size / 2;
    float power = 2.0;
    if (i < split) {
      float ret = powf(input_addr_0[i], power);
      MsAtomicAdd(ws_addr_0, ret);
    } else {
      float ret = powf(input_addr_1[i - split], power);
      MsAtomicAdd(ws_addr_1, ret);
    }
  }
  return;
}

template <typename T>
__global__ void AssignKernel(const size_t size, T* output_addr_0, T* output_addr_1,
                                   float* ws_addr_0, float* ws_addr_1) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    output_addr_0[0] = __float2half(ws_addr_0[0]);
    output_addr_1[0] = __float2half(ws_addr_1[0]);
  }
  return;
}

template <>
__global__ void AssignKernel(const size_t size, float* output_addr_0, float* output_addr_1,
                             float* ws_addr_0, float* ws_addr_1) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    output_addr_0[0] = ws_addr_0[0];
    output_addr_1[0] = ws_addr_1[0];
  }
  return;
}

template <typename T>
__global__ void InitOutput(const size_t size, T *output) {
  T zero = 0;
  for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < size; id += blockDim.x * gridDim.x) {
    output[id] = zero;
  }
  return;
}

template <typename T>
void SquareSumAll(const size_t input_size_, const T* input_addr_0, const T* input_addr_1,
                  T* output_addr_0, T* output_addr_1,
                  float* ws_addr_0, float* ws_addr_1, cudaStream_t cuda_stream) {
  InitOutput<<<GET_BLOCKS(1), GET_THREADS, 0, cuda_stream>>>(1, ws_addr_0);
  InitOutput<<<GET_BLOCKS(1), GET_THREADS, 0, cuda_stream>>>(1, ws_addr_1);
  size_t size = input_size_ * 2;
  SquareSumAllKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input_addr_0, input_addr_1,
                                                                        ws_addr_0, ws_addr_1);
  AssignKernel<<<GET_BLOCKS(1), GET_THREADS, 0, cuda_stream>>>(1, output_addr_0, output_addr_1, ws_addr_0, ws_addr_1);
}

template void SquareSumAll(const size_t input_size_, const half* input_addr_0, const half* input_addr_1,
                  half* output_addr_0, half* output_addr_1, float* ws_addr_0, float* ws_addr_1,
                  cudaStream_t cuda_stream);
template void SquareSumAll(const size_t input_size_, const float* input_addr_0, const float* input_addr_1,
                  float* output_addr_0, float* output_addr_1, float* ws_addr_0, float* ws_addr_1,
                  cudaStream_t cuda_stream);
