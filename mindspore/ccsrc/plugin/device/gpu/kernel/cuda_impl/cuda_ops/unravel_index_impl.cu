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

#include "unravel_index_impl.cuh"

__device__ int64_t dims_prod;

template <typename T>
__global__ void UnravelIndex(T *input_indices, T *input_dims, const size_t indices_size, const size_t dims_size,
                             T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < indices_size; pos += blockDim.x * gridDim.x) {
    int output_start_idx = static_cast<int>(pos * dims_size);
    int output_end_idx = static_cast<int>((pos + 1) * dims_size - 1);
    T cur_indices = input_indices[pos];
    CUDA_KERNEL_ASSERT(cur_indices >= 0);
    CUDA_KERNEL_ASSERT(cur_indices < dims_prod);

    int count = dims_size - 1;
    for (int idx = output_end_idx; idx >= output_start_idx; --idx) {
      int r = pos;
      int c = idx % dims_size;
      output[r + c * indices_size] = cur_indices % input_dims[count];
      cur_indices /= input_dims[count];
      --count;
    }
  }
}

template <typename T>
__global__ void CheckInputDims(T *input_dims, const size_t dims_size) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < dims_size; pos += blockDim.x * gridDim.x) {
    CUDA_KERNEL_ASSERT(input_dims[pos] > 0);
  }
}

template <typename T>
__global__ void GetDimsProd(T *input_dims, const size_t dims_size) {
  int64_t dims_prod_val = 1;
  for (size_t i = 0; i < dims_size; i++) {
    dims_prod_val = dims_prod_val * input_dims[i];
  }
  dims_prod = dims_prod_val;
  CUDA_KERNEL_ASSERT(dims_prod == dims_prod);
}

template <typename T>
cudaError_t CalUnravelIndex(T *input_indices, T *input_dims, const size_t indices_size, const size_t dims_size,
                            T *output, const uint32_t &device_id, cudaStream_t cuda_stream) {
  GetDimsProd<<<1, 1, 0, cuda_stream>>>(input_dims, dims_size);
  CheckInputDims<<<CUDA_BLOCKS(device_id, dims_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(input_dims, dims_size);
  UnravelIndex<<<CUDA_BLOCKS(device_id, indices_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input_indices, input_dims, indices_size, dims_size, output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalUnravelIndex<int32_t>(int32_t *input_indices, int32_t *input_dims,
                                                              const size_t indices_size, const size_t dims_size,
                                                              int32_t *output, const uint32_t &device_id,
                                                              cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalUnravelIndex<int64_t>(int64_t *input_indices, int64_t *input_dims,
                                                              const size_t indices_size, const size_t dims_size,
                                                              int64_t *output, const uint32_t &device_id,
                                                              cudaStream_t cuda_stream);
