/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/data_format_dim_map_impl.cuh"
#include <assert.h>

template <typename T>
__global__ void DataFormatDimMapKernel(size_t size, T *input_addr, T *output_addr, int32_t *dim_map) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output_addr[pos] = static_cast<T>(dim_map[(input_addr[pos] % 4 + 4) % 4]);
  }
}

template <typename T>
cudaError_t DataFormatDimMap(size_t size, T *input_addr, T *output_addr, int32_t *dim_map, cudaStream_t cuda_stream) {
  DataFormatDimMapKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input_addr, output_addr, dim_map);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t DataFormatDimMap(size_t size, int32_t *input_addr, int32_t *output_addr,
                                                      int32_t *dim_map, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t DataFormatDimMap(size_t size, int64_t *input_addr, int64_t *output_addr,
                                                      int32_t *dim_map, cudaStream_t cuda_stream);
