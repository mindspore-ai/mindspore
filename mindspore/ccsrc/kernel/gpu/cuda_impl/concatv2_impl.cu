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
__global__ void Concat(const size_t size, const int w1, const int w2, const T* input_1, const T* input_2, T* output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    int n = pos / (w1 + w2);
    int m = pos % (w1 + w2);
    output[pos] = m >= w1 ? input_2[n * w2 + m - w1] : input_1[n * w1 + m];
  }
  return;
}

template <typename T>
__global__ void Concat(const size_t size, const int w1, const int w2, const int w3,
                       const T* input_1, const T* input_2, const T* input_3, T* output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    int n = pos / (w1 + w2 + w3);
    int m = pos % (w1 + w2 + w3);
    output[pos] = m < w1 ? input_1[n * w1 + m] :
                    m < w1 + w2 ? input_2[n * w2 + m - w1] :
                      input_3[n * w3 + m - w1 - w2];
  }
  return;
}

template <typename T>
__global__ void Concat(const size_t size, const int w1, const int w2, const int w3, const int w4,
                       const T* input_1, const T* input_2, const T* input_3, const T* input_4, T* output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    int n = pos / (w1 + w2 + w3 + w4);
    int m = pos % (w1 + w2 + w3 + w4);
    output[pos] = m < w1 ? input_1[n * w1 + m] :
                    m < w1 + w2 ? input_2[n * w2 + m - w1]:
                      m < w1 + w2 + w3 ? input_3[n * w3 + m - w1 - w2]:
                        input_4[n * w4 + m - w1 - w2 - w3];
  }
  return;
}

template <typename T>
void ConcatKernel(const size_t size, const int w1, const int w2, const T* input_1, const T* input_2, T* output,
                 cudaStream_t cuda_stream) {
  Concat<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, w1, w2, input_1, input_2, output);
  return;
}

template <typename T>
void ConcatKernel(const size_t size, const int w1, const int w2, const int w3,
                  const T* input_1, const T* input_2, const T* input_3, T* output,
                  cudaStream_t cuda_stream) {
  Concat<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, w1, w2, w3, input_1, input_2, input_3, output);
  return;
}

template <typename T>
void ConcatKernel(const size_t size, const int w1, const int w2, const int w3, const int w4,
                  const T* input_1, const T* input_2, const T* input_3, const T* input_4, T* output,
                  cudaStream_t cuda_stream) {
  Concat<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, w1, w2, w3, w4, input_1,
                                                            input_2, input_3, input_4, output);
  return;
}

template void ConcatKernel(const size_t size, const int w1, const int w2, const float* input_1, const float* input_2,
                           float* output, cudaStream_t cuda_stream);
template void ConcatKernel(const size_t size, const int w1, const int w2, const int* input_1, const int* input_2,
                           int* output, cudaStream_t cuda_stream);
template void ConcatKernel(const size_t size, const int w1, const int w2, const half* input_1, const half* input_2,
                           half* output, cudaStream_t cuda_stream);

template void ConcatKernel(const size_t size, const int w1, const int w2, const int w3,
                           const float* input_1, const float* input_2, const float* input_3,
                           float* output, cudaStream_t cuda_stream);
template void ConcatKernel(const size_t size, const int w1, const int w2, const int w3,
                           const int* input_1, const int* input_2, const int* input_3,
                           int* output, cudaStream_t cuda_stream);
template void ConcatKernel(const size_t size, const int w1, const int w2, const int w3,
                           const half* input_1, const half* input_2, const half* input_3,
                           half* output, cudaStream_t cuda_stream);

template void ConcatKernel(const size_t size, const int w1, const int w2, const int w3, const int w4,
                           const float* input_1, const float* input_2, const float* input_3, const float* input_4,
                           float* output, cudaStream_t cuda_stream);
template void ConcatKernel(const size_t size, const int w1, const int w2, const int w3, const int w4,
                           const int* input_1, const int* input_2, const int* input_3, const int* input_4,
                           int* output, cudaStream_t cuda_stream);
template void ConcatKernel(const size_t size, const int w1, const int w2, const int w3, const int w4,
                           const half* input_1, const half* input_2, const half* input_3, const half* input_4,
                           half* output, cudaStream_t cuda_stream);

