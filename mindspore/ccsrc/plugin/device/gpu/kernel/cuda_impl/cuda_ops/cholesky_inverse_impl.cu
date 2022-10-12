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

#include <math.h>
#include "cholesky_inverse_impl.cuh"

template <typename T>
__global__ void CopyUpToLow(const size_t size, const T *input, const int64_t rank, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int row = pos / rank;
    int col = pos % rank;
    output[pos] = row <= col ? input[pos] : input[col * rank + row];
  }
  return;
}

template <typename T>
__global__ void CopyLowToUp(const size_t size, const T *input, const int64_t rank, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int row = pos / rank;
    int col = pos % rank;
    output[pos] = col <= row ? input[pos] : input[col * rank + row];
  }
  return;
}

template <typename T>
void CalCopyUpToLow(const size_t size, T *input, const int64_t rank, T *output, const uint32_t &device_id,
                    cudaStream_t cuda_stream) {
  CopyUpToLow<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input, rank,
                                                                                                output);
  return;
}

template <typename T>
void CalCopyLowToUp(const size_t size, T *input, const int64_t rank, T *output, const uint32_t &device_id,
                    cudaStream_t cuda_stream) {
  CopyLowToUp<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input, rank,
                                                                                                output);
  return;
}

template
CUDA_LIB_EXPORT void CalCopyUpToLow<float>(const size_t size, float *input, const int64_t rank,
                                           float *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalCopyUpToLow<double>(const size_t size, double *input, const int64_t rank,
                                            double *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalCopyLowToUp<float>(const size_t size, float *input, const int64_t rank,
                                           float *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalCopyLowToUp<double>(const size_t size, double *input, const int64_t rank,
                                            double *output, const uint32_t &device_id, cudaStream_t cuda_stream);
