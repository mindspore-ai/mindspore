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

#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "backend/kernel_compiler/gpu/cuda_impl/tensor_add_v2_impl.cuh"
template <typename T>
__global__ void TensorAddV2Kernel(const size_t element_num, const T* x1, const T* x2, T* y) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < element_num; i += blockDim.x * gridDim.x) {
   y[i] = x1[i] + x2[i];
  }
}

template <typename T>
void TensorAddV2(const size_t &element_num, const T* x1, const T* x2, T* y, cudaStream_t cuda_stream) {
  size_t thread_per_block = 256;
  size_t block_per_grid = (element_num + thread_per_block -1)/thread_per_block;
  TensorAddV2Kernel<<<block_per_grid, thread_per_block, 0, cuda_stream>>>(element_num,x1,x2,y);
  return;
}

template void TensorAddV2(const size_t &element_num, const float* x1, const float* x2, float* y, cudaStream_t cuda_stream);
