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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sparse_matrix_transpose_impl.cuh"
#include <stdint.h>
#include "include/cuda_runtime.h"
#include "include/cuda_fp16.h"

__global__ void ConjKernel(const size_t input_size, cuComplex *input_addr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < input_size; pos += blockDim.x * gridDim.x) {
    cuComplex *tar = input_addr + pos;
    *tar = cuConjf(*tar);
  }
}

__global__ void ConjKernel(const size_t input_size, cuDoubleComplex *input_addr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < input_size; pos += blockDim.x * gridDim.x) {
    cuDoubleComplex *tar = input_addr + pos;
    *tar = cuConj(*tar);
  }
}

cudaError_t Conj(const size_t input_size, cuComplex *input_addr, cudaStream_t stream) {
  ConjKernel<<<GET_BLOCKS(input_size), GET_THREADS, 0, stream>>>(input_size, input_addr);
  return GetCudaStatus();
}

cudaError_t Conj(const size_t input_size, cuDoubleComplex *input_addr, cudaStream_t stream) {
  ConjKernel<<<GET_BLOCKS(input_size), GET_THREADS, 0, stream>>>(input_size, input_addr);
  return GetCudaStatus();
}

CUDA_LIB_EXPORT cudaError_t Conj(const size_t input_size, cuComplex *input_addr, cudaStream_t stream);
CUDA_LIB_EXPORT cudaError_t Conj(const size_t input_size, cuDoubleComplex *input_addr, cudaStream_t stream);
