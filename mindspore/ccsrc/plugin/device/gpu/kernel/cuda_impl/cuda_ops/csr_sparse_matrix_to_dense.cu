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
#include "csr_sparse_matrix_to_dense.cuh"
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

template <typename T, typename S>
__global__ void CSRSparseMatrixToDenseKernel(const T *dense_shape_addr, T *batch_ptr_addr, T *row_ptr_addr,
                                             T *col_indices_addr, S *values_addr, S *output, size_t ndim, size_t rows) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < rows; i += blockDim.x * gridDim.x) {
    T cols = dense_shape_addr[ndim - 1];
    T batch_rows = dense_shape_addr[ndim - 2];
    int batch_index = i / (batch_rows + 1);
    T nnz = row_ptr_addr[i + 1] - row_ptr_addr[i];
    for (T j = 0; j < nnz; ++j) {
      T index = batch_ptr_addr[batch_index] + row_ptr_addr[i] + j;
      S value = values_addr[index];
      T col_index = col_indices_addr[index];
      T output_index = (i - batch_index) * cols + col_index;
      output[output_index] += value;
    }
  }
}

template <typename T, typename S>
cudaError_t CalCSRSparseMatrixToDense(const T *dense_shape_addr, T *batch_ptr_addr, T *row_ptr_addr,
                                      T *col_indices_addr, S *values_addr, S *output, size_t ndim, size_t rows,
                                      size_t nums, cudaStream_t cuda_stream) {
  cudaMemsetAsync(output, 0, nums * sizeof(S), cuda_stream);
  CSRSparseMatrixToDenseKernel<<<GET_BLOCKS(rows), GET_THREADS, 0, cuda_stream>>>(
    dense_shape_addr, batch_ptr_addr, row_ptr_addr, col_indices_addr, values_addr, output, ndim, rows);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalCSRSparseMatrixToDense<int, half>(const int *dense_shape_addr,
                                                                          int *batch_ptr_addr, int *row_ptr_addr,
                                                                          int *col_indices_addr, half *values_addr,
                                                                          half *output, size_t ndim, size_t rows,
                                                                          size_t nums, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalCSRSparseMatrixToDense<int, float>(const int *dense_shape_addr,
                                                                           int *batch_ptr_addr, int *row_ptr_addr,
                                                                           int *col_indices_addr, float *values_addr,
                                                                           float *output, size_t ndim, size_t rows,
                                                                           size_t nums, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalCSRSparseMatrixToDense<int, double>(const int *dense_shape_addr,
                                                                            int *batch_ptr_addr, int *row_ptr_addr,
                                                                            int *col_indices_addr, double *values_addr,
                                                                            double *output, size_t ndim, size_t rows,
                                                                            size_t nums, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalCSRSparseMatrixToDense<int, Complex<float>>(
  const int *dense_shape_addr, int *batch_ptr_addr, int *row_ptr_addr, int *col_indices_addr,
  Complex<float> *values_addr, Complex<float> *output, size_t ndim, size_t rows, size_t nums, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalCSRSparseMatrixToDense<int, Complex<double>>(
  const int *dense_shape_addr, int *batch_ptr_addr, int *row_ptr_addr, int *col_indices_addr,
  Complex<double> *values_addr, Complex<double> *output, size_t ndim, size_t rows, size_t nums,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalCSRSparseMatrixToDense<int64_t, half>(
  const int64_t *dense_shape_addr, int64_t *batch_ptr_addr, int64_t *row_ptr_addr, int64_t *col_indices_addr,
  half *values_addr, half *output, size_t ndim, size_t rows, size_t nums, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalCSRSparseMatrixToDense<int64_t, float>(
  const int64_t *dense_shape_addr, int64_t *batch_ptr_addr, int64_t *row_ptr_addr, int64_t *col_indices_addr,
  float *values_addr, float *output, size_t ndim, size_t rows, size_t nums, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalCSRSparseMatrixToDense<int64_t, double>(
  const int64_t *dense_shape_addr, int64_t *batch_ptr_addr, int64_t *row_ptr_addr, int64_t *col_indices_addr,
  double *values_addr, double *output, size_t ndim, size_t rows, size_t nums, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalCSRSparseMatrixToDense<int64_t, Complex<float>>(
  const int64_t *dense_shape_addr, int64_t *batch_ptr_addr, int64_t *row_ptr_addr, int64_t *col_indices_addr,
  Complex<float> *values_addr, Complex<float> *output, size_t ndim, size_t rows, size_t nums, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalCSRSparseMatrixToDense<int64_t, Complex<double>>(
  const int64_t *dense_shape_addr, int64_t *batch_ptr_addr, int64_t *row_ptr_addr, int64_t *col_indices_addr,
  Complex<double> *values_addr, Complex<double> *output, size_t ndim, size_t rows, size_t nums,
  cudaStream_t cuda_stream);
