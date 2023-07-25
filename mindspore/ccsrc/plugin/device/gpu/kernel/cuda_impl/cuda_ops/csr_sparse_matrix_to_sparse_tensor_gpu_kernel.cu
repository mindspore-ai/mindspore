/**
 * Copyright 2022 Huawei Sechnologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WISHOUS WARRANSIES OR CONDISIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <cuda_runtime.h>
#include "csr_sparse_matrix_to_sparse_tensor_gpu_kernel.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "include/cuda_fp16.h"

template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename S>
__global__ void StackIndices2D(const S *row_indices, const S *col_indices, S *indices, int size) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    indices[pos * 2] = static_cast<S>(__ldg(row_indices + pos));
    indices[pos * 2 + 1] = static_cast<S>(__ldg(col_indices + pos));
  }
}

template <typename S>
__device__ inline S BinarySearchRange(S *range, S n, S x) {
  S left = 0;
  S right = n - 1;
  while (left < right) {
    S mid = left + (right - left) / 2;
    if (x < range[mid]) {
      right = mid - 1;
    } else if (range[mid + 1] <= x) {
      left = mid + 1;
    } else {
      return mid;
    }
  }
  return left;
}

template <typename S>
__global__ void StackIndices3D(const S *batch_pointers, const S *row_indices, const S *col_indices, S *indices,
                               int batch_size, int total_nnz) {
  extern __shared__ S local_batch_ptr[];
  for (size_t i = threadIdx.x; i < batch_size + 1; i += blockDim.x) {
    local_batch_ptr[i] = batch_pointers[i];
  }
  __syncthreads();
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < total_nnz; pos += blockDim.x * gridDim.x) {
    S batch_idx = BinarySearchRange(local_batch_ptr, static_cast<S>(batch_size), static_cast<S>(pos));
    indices[pos * 3] = batch_idx;
    indices[pos * 3 + 1] = static_cast<S>(__ldg(row_indices + pos));
    indices[pos * 3 + 2] = static_cast<S>(__ldg(col_indices + pos));
  }
}

template <typename S>
cudaError_t CallStackIndices2D(const S *row_indices, const S *col_indices, S *indices, int size,
                               cudaStream_t cuda_stream) {
  StackIndices2D<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(row_indices, col_indices, indices, size);
  return GetCudaStatus();
}

template <typename S>
cudaError_t CallStackIndices3D(const S *batch_pointers, const S *row_indices, const S *col_indices, S *indices,
                               int batch_size, int total_nnz, size_t shared_memory_size, cudaStream_t cuda_stream) {
  StackIndices3D<<<GET_BLOCKS(total_nnz), GET_THREADS, shared_memory_size, cuda_stream>>>(
    batch_pointers, row_indices, col_indices, indices, batch_size, total_nnz);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CallStackIndices2D<int>(const int *row_indices, const int *col_indices,
                                                             int *indices, int size, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CallStackIndices3D<int>(const int *batch_pointers, const int *row_indices,
                                                             const int *col_indices, int *indices, int batch_size,
                                                             int total_nnz, size_t shared_memory_size,
                                                             cudaStream_t cuda_stream);
