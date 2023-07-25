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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sparse_matrix_softmax_impl.cuh"
#include <stdint.h>
#include <limits>
#include <algorithm>
#include "include/cuda_fp16.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"

template <typename DataType, typename IndexType>
__global__ void SparseMatrixSoftmaxKernel(int rows, IndexType *indptr, DataType *values, DataType *softmax) {
  for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < rows - 1; id += blockDim.x * gridDim.x) {
    IndexType begin = indptr[id];
    IndexType end = indptr[id + 1];

    DataType row_max = values[begin];
    for (int r_i = begin; r_i < end; ++r_i) {
      row_max = max(row_max, values[r_i]);
    }
    DataType sum_exp = 0;
    for (int r_i = begin; r_i < end; ++r_i) {
      DataType exp_i = exp(values[r_i] - row_max);
      softmax[r_i] = exp_i;
      sum_exp += exp_i;
    }
    for (int r_i = begin; r_i < end; ++r_i) {
      softmax[r_i] = softmax[r_i] / sum_exp;
    }
  }
}

template <typename IndexType>
__global__ void GpuCopy(IndexType *d_in, IndexType *d_out, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = idx; i < N / 2; i += blockDim.x * gridDim.x) {
    reinterpret_cast<int2 *>(d_out)[i] = reinterpret_cast<int2 *>(d_in)[i];
  }

  if (idx == N / 2 && N % 2 == 1) d_out[N - 1] = d_in[N - 1];
}

template <typename DataType, typename IndexType>
cudaError_t SparseMatrixSoftmax(int shape_size, int batch_pointers_size, int row_pointers_size, int col_indices_size,
                                IndexType *x_dense_shape, IndexType *x_batch_pointers, IndexType *x_row_pointers,
                                IndexType *x_col_indices, DataType *x_values, IndexType *y_dense_shape,
                                IndexType *y_batch_pointers, IndexType *y_row_pointers, IndexType *y_col_indices,
                                DataType *softmax, uint32_t device_id, cudaStream_t cuda_stream) {
  int threads_per_block = CUDA_THREADS(device_id);
  unsigned int grid_num = UP_DIV(row_pointers_size - 1, threads_per_block);

  // Copy identical Tensors
  GpuCopy<<<grid_num, threads_per_block, 0, cuda_stream>>>(x_dense_shape, y_dense_shape, shape_size);
  GpuCopy<<<grid_num, threads_per_block, 0, cuda_stream>>>(x_batch_pointers, y_batch_pointers, batch_pointers_size);
  GpuCopy<<<grid_num, threads_per_block, 0, cuda_stream>>>(x_row_pointers, y_row_pointers, row_pointers_size);
  GpuCopy<<<grid_num, threads_per_block, 0, cuda_stream>>>(x_col_indices, y_col_indices, col_indices_size);

  // Compute sparse matrix softmax
  SparseMatrixSoftmaxKernel<<<grid_num, threads_per_block, 0, cuda_stream>>>(row_pointers_size, x_row_pointers,
                                                                             x_values, softmax);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t SparseMatrixSoftmax<float, int32_t>(
  int shape_size, int batch_pointers_size, int row_pointers_size, int col_indices_size, int32_t *x_dense_shape,
  int32_t *x_batch_pointers, int32_t *x_row_pointers, int32_t *x_col_indices, float *x_values, int32_t *y_dense_shape,
  int32_t *y_batch_pointers, int32_t *y_row_pointers, int32_t *y_col_indices, float *softmax, uint32_t device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t SparseMatrixSoftmax<float, int64_t>(
  int shape_size, int batch_pointers_size, int row_pointers_size, int col_indices_size, int64_t *x_dense_shape,
  int64_t *x_batch_pointers, int64_t *x_row_pointers, int64_t *x_col_indices, float *x_values, int64_t *y_dense_shape,
  int64_t *y_batch_pointers, int64_t *y_row_pointers, int64_t *y_col_indices, float *softmax, uint32_t device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t SparseMatrixSoftmax<double, int32_t>(
  int shape_size, int batch_pointers_size, int row_pointers_size, int col_indices_size, int32_t *x_dense_shape,
  int32_t *x_batch_pointers, int32_t *x_row_pointers, int32_t *x_col_indices, double *x_values, int32_t *y_dense_shape,
  int32_t *y_batch_pointers, int32_t *y_row_pointers, int32_t *y_col_indices, double *softmax, uint32_t device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t SparseMatrixSoftmax<double, int64_t>(
  int shape_size, int batch_pointers_size, int row_pointers_size, int col_indices_size, int64_t *x_dense_shape,
  int64_t *x_batch_pointers, int64_t *x_row_pointers, int64_t *x_col_indices, double *x_values, int64_t *y_dense_shape,
  int64_t *y_batch_pointers, int64_t *y_row_pointers, int64_t *y_col_indices, double *softmax, uint32_t device_id,
  cudaStream_t cuda_stream);
