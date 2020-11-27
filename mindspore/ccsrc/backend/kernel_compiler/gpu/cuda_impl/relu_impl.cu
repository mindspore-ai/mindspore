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

#include "backend/kernel_compiler/gpu/cuda_impl/relu_impl.cuh"
#include "runtime/device/gpu/cuda_common.h"

template <typename T>
__global__ void CalReLUKernel(int size, T *input_addr, T *output_addr) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output_addr[pos] = input_addr[pos] > static_cast<T>(0) ? input_addr[pos] : static_cast<T>(0);
  }
}

template <typename T>
void CalReLU(int size, T *input_addr, T *output_addr, cudaStream_t cuda_stream) {
  CalReLUKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input_addr, output_addr);
  return;
}

template void CalReLU(int size, float *input_addr, float *output_addr, cudaStream_t cuda_stream);
template void CalReLU(int size, half *input_addr, half *output_addr, cudaStream_t cuda_stream);
template void CalReLU(int size, int32_t *input_addr, int32_t *output_addr, cudaStream_t cuda_stream);
template void CalReLU(int size, int64_t *input_addr, int64_t *output_addr, cudaStream_t cuda_stream);
