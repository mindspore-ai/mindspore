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
#include "dense_to_csr_sparse_matrix_gpu_kernel.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "include/cuda_fp16.h"

template <typename S>
__global__ void SplitIndices2D(const S *indices, S *row_indices, S *col_indices, int size) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    row_indices[pos] = static_cast<S>(__ldg(indices + pos * 2));
    col_indices[pos] = static_cast<S>(__ldg(indices + pos * 2 + 1));
  }
}

template <typename S>
void CallSplitIndices2D(const S *indices, S *row_indices, S *col_indices, int size, cudaStream_t cuda_stream) {
  SplitIndices2D<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(indices, row_indices, col_indices, size);
  return;
}

template <typename S>
__global__ void SplitIndices3D(const S *indices, S *batch_indices, S *row_indices, S *col_indices, int size) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    batch_indices[pos] = static_cast<S>(__ldg(indices + pos * 3));
    row_indices[pos] = static_cast<S>(__ldg(indices + pos * 3 + 1));
    col_indices[pos] = static_cast<S>(__ldg(indices + pos * 3 + 2));
  }
}

template <typename S>
void CallSplitIndices3D(const S *indices, S *batch_indices, S *row_indices, S *col_indices, int size,
                        cudaStream_t cuda_stream) {
  SplitIndices3D<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(indices, batch_indices, row_indices, col_indices,
                                                                    size);
  return;
}

template <typename S>
__global__ void NNZPerBatch(const S *batch_indices, S *nnz_per_batch, int size) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x) {
    MsAtomicAdd(&nnz_per_batch[batch_indices[idx] + 1], 1);
  }
}

template <typename S>
void CallNNZPerBatch(const S *batch_indices, S *nnz_per_batch, int nnz, int batch_ptr_size, cudaStream_t cuda_stream) {
  NNZPerBatch<<<GET_BLOCKS(nnz), GET_THREADS, 0, cuda_stream>>>(batch_indices, nnz_per_batch, nnz);
  thrust::device_ptr<S> batch_ptr(nnz_per_batch);
  thrust::inclusive_scan(thrust::cuda::par.on(cuda_stream), batch_ptr, batch_ptr + batch_ptr_size, batch_ptr);
  return;
}

template CUDA_LIB_EXPORT void CallSplitIndices2D<int>(const int *indices, int *row_indices, int *col_indices, int size,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CallSplitIndices3D<int>(const int *indices, int *batch_indices, int *row_indices,
                                                      int *col_indices, int size, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CallNNZPerBatch<int>(const int *batch_indices, int *nnz_per_batch, int nnz,
                                                   int batch_ptr_size, cudaStream_t cuda_stream);
