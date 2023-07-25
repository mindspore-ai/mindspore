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
#include "search_sorted_impl.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

template <typename S, typename T>
__global__ void SearchSortedKernelUpper(const S *sequence, const S *values, T *output, size_t search_repeat,
                                        size_t search_len, size_t size) {
  for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < size / search_len * search_repeat;
       x += blockDim.x * gridDim.x) {
    T start = x / search_repeat * search_len;
    T start1 = start;
    T end = start + search_len;
    T index;
    S key = values[x];
    while (start < end) {
      index = start + ((end - start) >> 1);
      if (!(key < sequence[index])) {
        start = index + 1;
      } else {
        end = index;
      }
    }
    output[x] = start - start1;
  }
}

template <typename S, typename T>
__global__ void SearchSortedKernelLower(const S *sequence, const S *values, T *output, size_t search_repeat,
                                        size_t search_len, size_t size) {
  for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < size / search_len * search_repeat;
       x += blockDim.x * gridDim.x) {
    T start = x / search_repeat * search_len;
    T start1 = start;
    T end = start + search_len;
    size_t index;
    S key = values[x];
    while (start < end) {
      index = start + ((end - start) >> 1);
      if (!(key <= sequence[index])) {
        start = index + 1;
      } else {
        end = index;
      }
    }
    output[x] = start - start1;
  }
}

template <typename S, typename T>
cudaError_t CalSearchSorted(const size_t size, const S *sequence, const S *values, T *output, int *seq_dim,
                            size_t search_repeat, size_t search_len, bool right, const uint32_t &device_id,
                            cudaStream_t cuda_stream, int *count) {
  size_t mn = size / search_len * search_repeat;
  if (right) {
    SearchSortedKernelUpper<<<CUDA_BLOCKS(device_id, mn), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      sequence, values, output, search_repeat, search_len, size);
  } else {
    SearchSortedKernelLower<<<CUDA_BLOCKS(device_id, mn), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      sequence, values, output, search_repeat, search_len, size);
  }
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalSearchSorted<double, int32_t>(
  const size_t size, const double *sequence, const double *values, int32_t *output, int *seq_dim, size_t search_repeat,
  size_t search_len, bool right, const uint32_t &device_id, cudaStream_t cuda_stream, int *count);

template CUDA_LIB_EXPORT cudaError_t CalSearchSorted<float, int32_t>(const size_t size, const float *sequence,
                                                                     const float *values, int32_t *output, int *seq_dim,
                                                                     size_t search_repeat, size_t search_len,
                                                                     bool right, const uint32_t &device_id,
                                                                     cudaStream_t cuda_stream, int *count);

template CUDA_LIB_EXPORT cudaError_t CalSearchSorted<int64_t, int32_t>(
  const size_t size, const int64_t *sequence, const int64_t *values, int32_t *output, int *seq_dim,
  size_t search_repeat, size_t search_len, bool right, const uint32_t &device_id, cudaStream_t cuda_stream, int *count);

template CUDA_LIB_EXPORT cudaError_t CalSearchSorted<int32_t, int32_t>(
  const size_t size, const int32_t *sequence, const int32_t *values, int32_t *output, int *seq_dim,
  size_t search_repeat, size_t search_len, bool right, const uint32_t &device_id, cudaStream_t cuda_stream, int *count);

template CUDA_LIB_EXPORT cudaError_t CalSearchSorted<int16_t, int32_t>(
  const size_t size, const int16_t *sequence, const int16_t *values, int32_t *output, int *seq_dim,
  size_t search_repeat, size_t search_len, bool right, const uint32_t &device_id, cudaStream_t cuda_stream, int *count);

template CUDA_LIB_EXPORT cudaError_t CalSearchSorted<int8_t, int32_t>(
  const size_t size, const int8_t *sequence, const int8_t *values, int32_t *output, int *seq_dim, size_t search_repeat,
  size_t search_len, bool right, const uint32_t &device_id, cudaStream_t cuda_stream, int *count);

template CUDA_LIB_EXPORT cudaError_t CalSearchSorted<double, int64_t>(
  const size_t size, const double *sequence, const double *values, int64_t *output, int *seq_dim, size_t search_repeat,
  size_t search_len, bool right, const uint32_t &device_id, cudaStream_t cuda_stream, int *count);

template CUDA_LIB_EXPORT cudaError_t CalSearchSorted<float, int64_t>(const size_t size, const float *sequence,
                                                                     const float *values, int64_t *output, int *seq_dim,
                                                                     size_t search_repeat, size_t search_len,
                                                                     bool right, const uint32_t &device_id,
                                                                     cudaStream_t cuda_stream, int *count);

template CUDA_LIB_EXPORT cudaError_t CalSearchSorted<int64_t, int64_t>(
  const size_t size, const int64_t *sequence, const int64_t *values, int64_t *output, int *seq_dim,
  size_t search_repeat, size_t search_len, bool right, const uint32_t &device_id, cudaStream_t cuda_stream, int *count);

template CUDA_LIB_EXPORT cudaError_t CalSearchSorted<int32_t, int64_t>(
  const size_t size, const int32_t *sequence, const int32_t *values, int64_t *output, int *seq_dim,
  size_t search_repeat, size_t search_len, bool right, const uint32_t &device_id, cudaStream_t cuda_stream, int *count);

template CUDA_LIB_EXPORT cudaError_t CalSearchSorted<int16_t, int64_t>(
  const size_t size, const int16_t *sequence, const int16_t *values, int64_t *output, int *seq_dim,
  size_t search_repeat, size_t search_len, bool right, const uint32_t &device_id, cudaStream_t cuda_stream, int *count);

template CUDA_LIB_EXPORT cudaError_t CalSearchSorted<int8_t, int64_t>(
  const size_t size, const int8_t *sequence, const int8_t *values, int64_t *output, int *seq_dim, size_t search_repeat,
  size_t search_len, bool right, const uint32_t &device_id, cudaStream_t cuda_stream, int *count);
