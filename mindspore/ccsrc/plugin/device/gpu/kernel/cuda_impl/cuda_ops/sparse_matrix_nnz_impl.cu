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

#include "sparse_matrix_nnz_impl.cuh"


template <typename T>
__global__ void SparseMatrixNNZ(const size_t size, const T *input, int32_t *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = input[pos + 1] - (pos == 0 ? 0 : input[pos]);
  }
  return;
}

template <typename T>
void CalSparseMatrixNNZ(const size_t size, const T *input, int32_t *output, const uint32_t &device_id,
                        cudaStream_t cuda_stream) {
  SparseMatrixNNZ<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input, output);
  return;
}

template
CUDA_LIB_EXPORT void CalSparseMatrixNNZ<int32_t>(const size_t size, const int32_t *input, int32_t *output,
                                                 const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalSparseMatrixNNZ<int64_t>(const size_t size, const int64_t *input, int32_t *output,
                                                 const uint32_t &device_id, cudaStream_t cuda_stream);
