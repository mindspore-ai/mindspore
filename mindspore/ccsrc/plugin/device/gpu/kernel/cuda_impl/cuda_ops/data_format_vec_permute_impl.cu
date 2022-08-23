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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/data_format_vec_permute_impl.cuh"
#include "include/cuda_runtime.h"

template <typename T>
__global__ void DataFormatVecPermuteKernel1D(const size_t size, const T *input, T *output, int32_t *index) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = input[index[pos]];
  }
  return;
}

template <typename T>
__global__ void DataFormatVecPermuteKernel2D(const size_t size, const T *input, T *output, int32_t *index) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int32_t dim = static_cast<int32_t>(2);
    int32_t i = static_cast<int32_t>(pos) / dim;
    output[dim * i] = input[dim * index[i]];
    output[dim * i + 1] = input[dim * index[i]+1];
  }
  return;
}

template <typename T>
void CalDataFormatVecPermute1D(const size_t size, const T *input, T *output, int32_t *index, const uint32_t &device_id,
                               cudaStream_t cuda_stream) {
  DataFormatVecPermuteKernel1D<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input,
                                                                                                          output,
                                                                                                          index);
}

template <typename T>
void CalDataFormatVecPermute2D(const size_t size, const T *input, T *output, int32_t *index, const uint32_t &device_id,
                               cudaStream_t cuda_stream) {
  DataFormatVecPermuteKernel2D<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input,
                                                                                                          output,
                                                                                                          index);
}

template CUDA_LIB_EXPORT void CalDataFormatVecPermute1D<int>(const size_t size, const int *input, int *output,
                                                             int32_t *index, const uint32_t &device_id,
                                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalDataFormatVecPermute1D<int64_t>(const size_t size, const int64_t *input,
                                                                 int64_t *output, int32_t *index,
                                                                 const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalDataFormatVecPermute2D<int>(const size_t size, const int *input, int *output,
                                                             int32_t *index, const uint32_t &device_id,
                                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalDataFormatVecPermute2D<int64_t>(const size_t size, const int64_t *input,
                                                                 int64_t *output, int32_t *index,
                                                                 const uint32_t &device_id, cudaStream_t cuda_stream);
