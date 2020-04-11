/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "kernel/gpu/cuda_impl/concatv2_impl.cuh"
template <typename T>
__global__ void ConcatV2(const size_t size, const int w1, const int w2, const T* input_1, const T* input_2, T* output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    int n = pos / (w1 + w2);
    int m = pos % (w1 + w2);
    output[pos] = m >= w1 ? input_2[n * w2 + m - w1] : input_1[n * w1 + m];
  }
  return;
}

template <typename T>
void CalConcatV2(const size_t size, const int w1, const int w2, const T* input_1, const T* input_2, T* output,
                 cudaStream_t cuda_stream) {
  ConcatV2<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, w1, w2, input_1, input_2, output);
  return;
}

template void CalConcatV2(const size_t size, const int w1, const int w2, const float* input_1, const float* input_2,
                          float* output, cudaStream_t cuda_stream);
template void CalConcatV2(const size_t size, const int w1, const int w2, const int* input_1, const int* input_2,
                          int* output, cudaStream_t cuda_stream);
template void CalConcatV2(const size_t size, const int w1, const int w2, const half* input_1, const half* input_2,
                          half* output, cudaStream_t cuda_stream);

