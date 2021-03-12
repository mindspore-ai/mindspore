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
#include <include/cuda_runtime.h>
#include "backend/kernel_compiler/gpu/cuda_impl/select_impl.cuh"

template <typename T>
__global__ void Select(const size_t size, const bool* cond, const T* input_x, const T* input_y, T* output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    output[pos] = cond[pos] ? input_x[pos] : input_y[pos];
  }
  return;
}

template <typename T>
void CalSelect(const size_t size, const bool* cond, const T* input_x, const T* input_y, T* output,
               cudaStream_t cuda_stream) {
  Select<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, cond, input_x, input_y, output);
  return;
}

template void CalSelect<double>(const size_t size, const bool* cond, const double* input_X, const double* input_y,
                                double* output, cudaStream_t cuda_stream);
template void CalSelect<float>(const size_t size, const bool* cond, const float* input_X, const float* input_y,
                               float* output, cudaStream_t cuda_stream);
template void CalSelect<int>(const size_t size, const bool* cond, const int* input_X, const int* input_y, int* output,
                             cudaStream_t cuda_stream);
template void CalSelect<half>(const size_t size, const bool* cond, const half* input_X, const half* input_y,
                              half* output, cudaStream_t cuda_stream);
template void CalSelect<int64_t>(const size_t size, const bool* cond, const int64_t* input_X, const int64_t* input_y,
                              int64_t* output, cudaStream_t cuda_stream);

