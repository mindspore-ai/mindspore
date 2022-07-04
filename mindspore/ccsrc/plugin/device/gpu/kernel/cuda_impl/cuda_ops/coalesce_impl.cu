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

#include "coalesce_impl.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "include/cuda_fp16.h"

template <typename T, typename V>
__global__ void FlattenIndicesKernel(const T *input, T *output, V *shape, int indices_num) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < indices_num; i += gridDim.x * blockDim.x) {
    output[i] = input[i] * shape[1] + input[i + indices_num];
  }
}

template <typename T, typename V>
void FlattenIndices(const T *input, T *output, V *shape, int indices_num, cudaStream_t cuda_stream) {
  FlattenIndicesKernel<<<GET_BLOCKS(indices_num), GET_THREADS, 0, cuda_stream>>>(input, output, shape, indices_num);
  return;
}

template <typename T, typename V>
__global__ void ConvertTo2DIndicesKernel(T *input, T *output, V *shape, int indices_num) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < indices_num; i += gridDim.x * blockDim.x) {
    output[i] = input[i] / shape[1];
    output[i + indices_num] = input[i] % shape[1];
  }
}

template <typename T, typename V>
void ConvertTo2DIndices(T *input, T *output, V *shape, int indices_num, cudaStream_t cuda_stream) {
  ConvertTo2DIndicesKernel<<<GET_BLOCKS(indices_num), GET_THREADS, 0, cuda_stream>>>(input, output, shape, indices_num);
  return;
}

template <typename T, typename S>
__global__ void CalUniqueValuesdKernel(const T *indices, S *update, S *output, int values_num) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < values_num; i += gridDim.x * blockDim.x) {
    T index = indices[i];
    if (index < values_num) {
      MsAtomicAdd(&output[index], update[i]);
    }
  }
}

template <typename T, typename S>
void CalUniqueValues(const T *indices, S *update, S *output, int unique_num, int values_num, cudaStream_t stream) {
  cudaMemsetAsync(output, 0, sizeof(S) * unique_num, stream);
  CalUniqueValuesdKernel<<<GET_BLOCKS(values_num), GET_THREADS, 0, stream>>>(indices, update, output, values_num);
  return;
}

template CUDA_LIB_EXPORT void FlattenIndices<int, int64_t>(const int *input, int *output, int64_t *shape,
                                                           int indices_num, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void FlattenIndices<int64_t, int64_t>(const int64_t *input, int64_t *output, int64_t *shape,
                                                               int indices_num, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ConvertTo2DIndices<int64_t, int64_t>(int64_t *input, int64_t *output, int64_t *shape,
                                                                   int indices_num, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ConvertTo2DIndices<int, int64_t>(int *input, int *output, int64_t *shape, int indices_num,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalUniqueValues<int, half>(const int *indices, half *update, half *output, int unique_num,
                                                         int values_num, cudaStream_t stream);
template CUDA_LIB_EXPORT void CalUniqueValues<int64_t, half>(const int64_t *indices, half *update, half *output,
                                                             int unique_num, int values_num, cudaStream_t stream);
template CUDA_LIB_EXPORT void CalUniqueValues<int, float>(const int *indices, float *update, float *output,
                                                          int unique_num, int values_num, cudaStream_t stream);
template CUDA_LIB_EXPORT void CalUniqueValues<int64_t, float>(const int64_t *indices, float *update, float *output,
                                                              int unique_num, int values_num, cudaStream_t stream);
template CUDA_LIB_EXPORT void CalUniqueValues<int, double>(const int *indices, double *update, double *output,
                                                           int unique_num, int values_num, cudaStream_t stream);
template CUDA_LIB_EXPORT void CalUniqueValues<int64_t, double>(const int64_t *indices, double *update, double *output,
                                                               int unique_num, int values_num, cudaStream_t stream);
