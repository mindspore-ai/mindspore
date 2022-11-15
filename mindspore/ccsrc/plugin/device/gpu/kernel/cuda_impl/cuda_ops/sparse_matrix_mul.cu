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

#include <algorithm>
#include <iostream>
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sparse_matrix_mul.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T, typename S>
__global__ void
CSRSparseMatrixMulKernel(const T *a_shape_addr, T *a_indptr_addr,
                         T *a_indices_addr, S *a_values_addr, S *b_dense_addr,
                         T *c_shape_addr, T *c_indptr_addr, T *c_indices_addr,
                         S *c_values_addr, int row_, int col_) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < col_) {
    int32_t col = a_indices_addr[i];
    int32_t row = -1;
    for (int j = 0; j < row_ - 1; j++) {
      if (a_indptr_addr[j] <= i && a_indptr_addr[j + 1] > i) {
        row = j;
        break;
      }
    }
    int32_t absIndex = row * a_shape_addr[1] + col;
    c_values_addr[i] = a_values_addr[i] * b_dense_addr[absIndex];
  }
}

template <typename T, typename S>
void CalSparseMatrixMul(const T *a_shape_addr, T *a_batch_pointers_addr,
                        T *a_indptr_addr, T *a_indices_addr, S *a_values_addr,
                        S *b_dense_addr, T *c_shape_addr,
                        T *c_batch_pointers_addr, T *c_indptr_addr,
                        T *c_indices_addr, S *c_values_addr, int row_, int col_,
                        uint32_t device_id, cudaStream_t cuda_stream) {
  CSRSparseMatrixMulKernel<<<1, CUDA_THREADS(device_id), 0, cuda_stream>>>(
      a_shape_addr, a_indptr_addr, a_indices_addr, a_values_addr, b_dense_addr,
      c_shape_addr, c_indptr_addr, c_indices_addr, c_values_addr, row_, col_);
  return;
}
template CUDA_LIB_EXPORT void CalSparseMatrixMul<int, float>(
    const int *a_shape_addr, int *a_batch_pointers_addr, int *a_indptr_addr,
    int *a_indices_addr, float *a_values_addr, float *b_dense_addr,
    int *c_shape_addr, int *c_batch_pointers_addr, int *c_indptr_addr,
    int *c_indices_addr, float *c_values_addr, int row_, int col_,
    uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalSparseMatrixMul<int, double>(
    const int *a_shape_addr, int *a_batch_pointers_addr, int *a_indptr_addr,
    int *a_indices_addr, double *a_values_addr, double *b_dense_addr,
    int *c_shape_addr, int *c_batch_pointers_addr, int *c_indptr_addr,
    int *c_indices_addr, double *c_values_addr, int row_, int col_,
    uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalSparseMatrixMul<int64_t, float>(
    const int64_t *a_shape_addr, int64_t *a_batch_pointers_addr,
    int64_t *a_indptr_addr, int64_t *a_indices_addr, float *a_values_addr,
    float *b_dense_addr, int64_t *c_shape_addr, int64_t *c_batch_pointers_addr,
    int64_t *c_indptr_addr, int64_t *c_indices_addr, float *c_values_addr,
    int row_, int col_, uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalSparseMatrixMul<int64_t, double>(
    const int64_t *a_shape_addr, int64_t *a_batch_pointers_addr,
    int64_t *a_indptr_addr, int64_t *a_indices_addr, double *a_values_addr,
    double *b_dense_addr, int64_t *c_shape_addr, int64_t *c_batch_pointers_addr,
    int64_t *c_indptr_addr, int64_t *c_indices_addr, double *c_values_addr,
    int row_, int col_, uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalSparseMatrixMul<int64_t, int>(
    const int64_t *a_shape_addr, int64_t *a_batch_pointers_addr,
    int64_t *a_indptr_addr, int64_t *a_indices_addr, int *a_values_addr,
    int *b_dense_addr, int64_t *c_shape_addr, int64_t *c_batch_pointers_addr,
    int64_t *c_indptr_addr, int64_t *c_indices_addr, int *c_values_addr,
    int row_, int col_, uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalSparseMatrixMul<int64_t, int64_t>(
    const int64_t *a_shape_addr, int64_t *a_batch_pointers_addr,
    int64_t *a_indptr_addr, int64_t *a_indices_addr, int64_t *a_values_addr,
    int64_t *b_dense_addr, int64_t *c_shape_addr,
    int64_t *c_batch_pointers_addr, int64_t *c_indptr_addr,
    int64_t *c_indices_addr, int64_t *c_values_addr, int row_, int col_,
    uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalSparseMatrixMul<int, int>(
    const int *a_shape_addr, int *a_batch_pointers_addr, int *a_indptr_addr,
    int *a_indices_addr, int *a_values_addr, int *b_dense_addr,
    int *c_shape_addr, int *c_batch_pointers_addr, int *c_indptr_addr,
    int *c_indices_addr, int *c_values_addr, int row_, int col_,
    uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalSparseMatrixMul<int, int64_t>(
    const int *a_shape_addr, int *a_batch_pointers_addr, int *a_indptr_addr,
    int *a_indices_addr, int64_t *a_values_addr, int64_t *b_dense_addr,
    int *c_shape_addr, int *c_batch_pointers_addr, int *c_indptr_addr,
    int *c_indices_addr, int64_t *c_values_addr, int row_, int col_,
    uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalSparseMatrixMul<int, int16_t>(
    const int *a_shape_addr, int *a_batch_pointers_addr, int *a_indptr_addr,
    int *a_indices_addr, int16_t *a_values_addr, int16_t *b_dense_addr,
    int *c_shape_addr, int *c_batch_pointers_addr, int *c_indptr_addr,
    int *c_indices_addr, int16_t *c_values_addr, int row_, int col_,
    uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalSparseMatrixMul<int64_t, int16_t>(
    const int64_t *a_shape_addr, int64_t *a_batch_pointers_addr,
    int64_t *a_indptr_addr, int64_t *a_indices_addr, int16_t *a_values_addr,
    int16_t *b_dense_addr, int64_t *c_shape_addr,
    int64_t *c_batch_pointers_addr, int64_t *c_indptr_addr,
    int64_t *c_indices_addr, int16_t *c_values_addr, int row_, int col_,
    uint32_t device_id, cudaStream_t cuda_stream);
