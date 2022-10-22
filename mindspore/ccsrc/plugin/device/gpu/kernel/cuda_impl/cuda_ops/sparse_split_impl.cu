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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sparse_split_impl.cuh"
#include <complex>
#include <algorithm>
#include <iostream>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"

template <typename DataType, typename IndexType>
__global__ void SparseSplitKernel(IndexType *split_dim_ptr, IndexType *indices_ptr, DataType *values_ptr,
                                  IndexType *shape_ptr, IndexType num_split, IndexType **y_indices_ptr,
                                  DataType **y_values_ptr, IndexType *out_shape_ptr, int *sum_count_ptr,
                                  size_t input_nnz_, size_t num_dim_, IndexType *d_block_ptr) {
  // atomicADD
  for (size_t input_nz = blockIdx.x * blockDim.x + threadIdx.x; input_nz < input_nnz_;
       input_nz += blockDim.x * gridDim.x) {
    DataType value = values_ptr[input_nz];
    IndexType *index = indices_ptr + input_nz * 2;
    IndexType idx_i = index[*split_dim_ptr];
    IndexType block;
    for (IndexType i = 0; i < num_split; i++) {
      if (idx_i < d_block_ptr[i + 1] && idx_i >= d_block_ptr[i]) {
        block = i;
      }
    }
    int idx = atomicAdd(&sum_count_ptr[block], 1);
    if (split_dim_ptr == 0) {
      y_indices_ptr[block][idx * 2] = index[0] - d_block_ptr[block];
      y_indices_ptr[block][idx * 2 + 1] = index[1];
    } else {
      y_indices_ptr[block][idx * 2] = index[0];
      y_indices_ptr[block][idx * 2 + 1] = index[1] - d_block_ptr[block];
    }
    y_values_ptr[block][idx] = value;
  }
}

template <typename DataType, typename IndexType>
CUDA_LIB_EXPORT void SparseSplit(IndexType *split_dim_ptr, IndexType *indices_ptr, DataType *values_ptr,
                                 IndexType *shape_ptr, IndexType num_split, IndexType **y_indices_ptr,
                                 DataType **y_values_ptr, IndexType *out_shape_ptr, int *sum_count_ptr,
                                 size_t input_nnz_, size_t num_dim_, IndexType *d_block_ptr, cudaStream_t cuda_stream) {
  SparseSplitKernel<<<GET_BLOCKS(input_nnz_), GET_THREADS, 0, cuda_stream>>>(
    split_dim_ptr, indices_ptr, values_ptr, shape_ptr, num_split, y_indices_ptr, y_values_ptr, out_shape_ptr,
    sum_count_ptr, input_nnz_, num_dim_, d_block_ptr);
}

template CUDA_LIB_EXPORT void SparseSplit<uint8_t, int64_t>(int64_t *split_dim_ptr, int64_t *indices_ptr,
                                                            uint8_t *values_ptr, int64_t *shape_ptr, int64_t num_split,
                                                            int64_t **y_indices_ptr, uint8_t **y_values_ptr,
                                                            int64_t *out_shape_ptr, int *sum_count_ptr,
                                                            size_t input_nnz_, size_t num_dim_, int64_t *d_block_ptr,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSplit<uint16_t, int64_t>(int64_t *split_dim_ptr, int64_t *indices_ptr,
                                                             uint16_t *values_ptr, int64_t *shape_ptr,
                                                             int64_t num_split, int64_t **y_indices_ptr,
                                                             uint16_t **y_values_ptr, int64_t *out_shape_ptr,
                                                             int *sum_count_ptr, size_t input_nnz_, size_t num_dim_,
                                                             int64_t *d_block_ptr, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSplit<int64_t, int64_t>(int64_t *split_dim_ptr, int64_t *indices_ptr,
                                                            int64_t *values_ptr, int64_t *shape_ptr, int64_t num_split,
                                                            int64_t **y_indices_ptr, int64_t **y_values_ptr,
                                                            int64_t *out_shape_ptr, int *sum_count_ptr,
                                                            size_t input_nnz_, size_t num_dim_, int64_t *d_block_ptr,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSplit<int32_t, int64_t>(int64_t *split_dim_ptr, int64_t *indices_ptr,
                                                            int32_t *values_ptr, int64_t *shape_ptr, int64_t num_split,
                                                            int64_t **y_indices_ptr, int32_t **y_values_ptr,
                                                            int64_t *out_shape_ptr, int *sum_count_ptr,
                                                            size_t input_nnz_, size_t num_dim_, int64_t *d_block_ptr,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSplit<int16_t, int64_t>(int64_t *split_dim_ptr, int64_t *indices_ptr,
                                                            int16_t *values_ptr, int64_t *shape_ptr, int64_t num_split,
                                                            int64_t **y_indices_ptr, int16_t **y_values_ptr,
                                                            int64_t *out_shape_ptr, int *sum_count_ptr,
                                                            size_t input_nnz_, size_t num_dim_, int64_t *d_block_ptr,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSplit<int8_t, int64_t>(int64_t *split_dim_ptr, int64_t *indices_ptr,
                                                           int8_t *values_ptr, int64_t *shape_ptr, int64_t num_split,
                                                           int64_t **y_indices_ptr, int8_t **y_values_ptr,
                                                           int64_t *out_shape_ptr, int *sum_count_ptr,
                                                           size_t input_nnz_, size_t num_dim_, int64_t *d_block_ptr,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSplit<double, int64_t>(int64_t *split_dim_ptr, int64_t *indices_ptr,
                                                           double *values_ptr, int64_t *shape_ptr, int64_t num_split,
                                                           int64_t **y_indices_ptr, double **y_values_ptr,
                                                           int64_t *out_shape_ptr, int *sum_count_ptr,
                                                           size_t input_nnz_, size_t num_dim_, int64_t *d_block_ptr,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSplit<float, int64_t>(int64_t *split_dim_ptr, int64_t *indices_ptr,
                                                          float *values_ptr, int64_t *shape_ptr, int64_t num_split,
                                                          int64_t **y_indices_ptr, float **y_values_ptr,
                                                          int64_t *out_shape_ptr, int *sum_count_ptr, size_t input_nnz_,
                                                          size_t num_dim_, int64_t *d_block_ptr,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSplit<half, int64_t>(int64_t *split_dim_ptr, int64_t *indices_ptr, half *values_ptr,
                                                         int64_t *shape_ptr, int64_t num_split, int64_t **y_indices_ptr,
                                                         half **y_values_ptr, int64_t *out_shape_ptr,
                                                         int *sum_count_ptr, size_t input_nnz_, size_t num_dim_,
                                                         int64_t *d_block_ptr, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSplit<bool, int64_t>(int64_t *split_dim_ptr, int64_t *indices_ptr, bool *values_ptr,
                                                         int64_t *shape_ptr, int64_t num_split, int64_t **y_indices_ptr,
                                                         bool **y_values_ptr, int64_t *out_shape_ptr,
                                                         int *sum_count_ptr, size_t input_nnz_, size_t num_dim_,
                                                         int64_t *d_block_ptr, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSplit<Complex<float>, int64_t>(
  int64_t *split_dim_ptr, int64_t *indices_ptr, Complex<float> *values_ptr, int64_t *shape_ptr, int64_t num_split,
  int64_t **y_indices_ptr, Complex<float> **y_values_ptr, int64_t *out_shape_ptr, int *sum_count_ptr, size_t input_nnz_,
  size_t num_dim_, int64_t *d_block_ptr, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSplit<Complex<double>, int64_t>(
  int64_t *split_dim_ptr, int64_t *indices_ptr, Complex<double> *values_ptr, int64_t *shape_ptr, int64_t num_split,
  int64_t **y_indices_ptr, Complex<double> **y_values_ptr, int64_t *out_shape_ptr, int *sum_count_ptr,
  size_t input_nnz_, size_t num_dim_, int64_t *d_block_ptr, cudaStream_t cuda_stream);
