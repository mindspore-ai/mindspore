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

#include <complex>
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sparse_slice_impl.cuh"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename DataType, typename IndexType>
__global__ void SparseSliceKernel(const IndexType *indices_ptr, const DataType *values_ptr,
                                  const IndexType *x_ptr, IndexType *start_ptr, IndexType *size_ptr,
                                  IndexType *y_indices_ptr, DataType *y_values_ptr, IndexType *out_shape_ptr,
                                  int64_t *sum_count_ptr, size_t input_nnz_, size_t num_dim_, size_t out_size_) {
  int64_t addnum = 1;
  for (size_t input_nz = blockIdx.x * blockDim.x + threadIdx.x; input_nz < input_nnz_;
       input_nz += blockDim.x * gridDim.x) {
    size_t select = 1;
    DataType value = values_ptr[input_nz];
    for (int dim = 0; dim < num_dim_; dim += 1) {
      const IndexType start = start_ptr[dim];
      const IndexType end = size_ptr[dim] + start;
      const IndexType index = indices_ptr[input_nz * num_dim_ + dim];
      if (index < start || index >= end) {
        select = 0;
      }
    }
    if (select) {
      y_values_ptr[*sum_count_ptr] = value;
      for (int dim = 0; dim < num_dim_; dim += 1) {
        auto start = start_ptr[dim];
        IndexType index = indices_ptr[input_nz * num_dim_ + dim];
        IndexType new_index = index - start;
        y_indices_ptr[*sum_count_ptr * num_dim_ + dim] = new_index;
      }
      MsAtomicAdd(sum_count_ptr, addnum);
    }
  }
}

template <typename DataType, typename IndexType>
CUDA_LIB_EXPORT void SparseSlice(const IndexType *indices_ptr, const DataType *values_ptr,
                                 const IndexType *x_ptr, IndexType *start_ptr, IndexType *size_ptr,
                                 IndexType *y_indices_ptr, DataType *y_values_ptr, IndexType *out_shape_ptr,
                                 int64_t *sum_count_ptr, size_t input_nnz_, size_t num_dim_, size_t out_size_,
                                 uint32_t device_id, cudaStream_t cuda_stream) {
  SparseSliceKernel<<<1, 1, 0, cuda_stream>>>(
    indices_ptr, values_ptr, x_ptr, start_ptr, size_ptr, y_indices_ptr, y_values_ptr, out_shape_ptr, sum_count_ptr,
    input_nnz_, num_dim_, out_size_);
}

template CUDA_LIB_EXPORT void SparseSlice<uint8_t, int64_t>(const int64_t *indices_ptr, const uint8_t *values_ptr,
                                                            const int64_t *x_ptr, int64_t *start_ptr,
                                                            int64_t *size_ptr, int64_t *y_indices_ptr,
                                                            uint8_t *y_values_ptr, int64_t *out_shape_ptr,
                                                            int64_t *sum_count_ptr,
                                                            size_t input_nnz_, size_t num_dim_, size_t out_size_,
                                                            uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSlice<uint16_t, int64_t>(const int64_t *indices_ptr, const uint16_t *values_ptr,
                                                            const int64_t *x_ptr, int64_t *start_ptr,
                                                            int64_t *size_ptr, int64_t *y_indices_ptr,
                                                            uint16_t *y_values_ptr, int64_t *out_shape_ptr,
                                                            int64_t *sum_count_ptr,
                                                            size_t input_nnz_, size_t num_dim_, size_t out_size_,
                                                            uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSlice<uint32_t, int64_t>(const int64_t *indices_ptr, const uint32_t *values_ptr,
                                                            const int64_t *x_ptr, int64_t *start_ptr,
                                                            int64_t *size_ptr, int64_t *y_indices_ptr,
                                                            uint32_t *y_values_ptr, int64_t *out_shape_ptr,
                                                            int64_t *sum_count_ptr,
                                                            size_t input_nnz_, size_t num_dim_, size_t out_size_,
                                                            uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSlice<uint64_t, int64_t>(const int64_t *indices_ptr, const uint64_t *values_ptr,
                                                            const int64_t *x_ptr, int64_t *start_ptr,
                                                            int64_t *size_ptr, int64_t *y_indices_ptr,
                                                            uint64_t *y_values_ptr, int64_t *out_shape_ptr,
                                                            int64_t *sum_count_ptr,
                                                            size_t input_nnz_, size_t num_dim_, size_t out_size_,
                                                            uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSlice<int64_t, int64_t>(const int64_t *indices_ptr, const int64_t *values_ptr,
                                                            const int64_t *x_ptr, int64_t *start_ptr,
                                                            int64_t *size_ptr, int64_t *y_indices_ptr,
                                                            int64_t *y_values_ptr, int64_t *out_shape_ptr,
                                                            int64_t *sum_count_ptr,
                                                            size_t input_nnz_, size_t num_dim_, size_t out_size_,
                                                            uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSlice<int32_t, int64_t>(const int64_t *indices_ptr, const int32_t *values_ptr,
                                                            const int64_t *x_ptr, int64_t *start_ptr,
                                                            int64_t *size_ptr, int64_t *y_indices_ptr,
                                                            int32_t *y_values_ptr, int64_t *out_shape_ptr,
                                                            int64_t *sum_count_ptr,
                                                            size_t input_nnz_, size_t num_dim_, size_t out_size_,
                                                            uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSlice<int16_t, int64_t>(const int64_t *indices_ptr, const int16_t *values_ptr,
                                                            const int64_t *x_ptr, int64_t *start_ptr,
                                                            int64_t *size_ptr, int64_t *y_indices_ptr,
                                                            int16_t *y_values_ptr, int64_t *out_shape_ptr,
                                                            int64_t *sum_count_ptr,
                                                            size_t input_nnz_, size_t num_dim_, size_t out_size_,
                                                            uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSlice<int8_t, int64_t>(const int64_t *indices_ptr, const int8_t *values_ptr,
                                                            const int64_t *x_ptr, int64_t *start_ptr,
                                                            int64_t *size_ptr, int64_t *y_indices_ptr,
                                                            int8_t *y_values_ptr, int64_t *out_shape_ptr,
                                                            int64_t *sum_count_ptr,
                                                            size_t input_nnz_, size_t num_dim_, size_t out_size_,
                                                            uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSlice<double, int64_t>(const int64_t *indices_ptr, const double *values_ptr,
                                                          const int64_t *x_ptr, int64_t *start_ptr,
                                                          int64_t *size_ptr, int64_t *y_indices_ptr,
                                                          double *y_values_ptr, int64_t *out_shape_ptr,
                                                          int64_t *sum_count_ptr,
                                                          size_t input_nnz_, size_t num_dim_, size_t out_size_,
                                                          uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSlice<float, int64_t>(const int64_t *indices_ptr, const float *values_ptr,
                                                          const int64_t *x_ptr, int64_t *start_ptr,
                                                          int64_t *size_ptr, int64_t *y_indices_ptr,
                                                          float *y_values_ptr, int64_t *out_shape_ptr,
                                                          int64_t *sum_count_ptr,
                                                          size_t input_nnz_, size_t num_dim_, size_t out_size_,
                                                          uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSlice<half, int64_t>(const int64_t *indices_ptr, const half *values_ptr,
                                                         const int64_t *x_ptr, int64_t *start_ptr,
                                                         int64_t *size_ptr, int64_t *y_indices_ptr,
                                                         half *y_values_ptr, int64_t *out_shape_ptr,
                                                         int64_t *sum_count_ptr,
                                                         size_t input_nnz_, size_t num_dim_, size_t out_size_,
                                                         uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSlice<bool, int64_t>(const int64_t *indices_ptr, const bool *values_ptr,
                                                         const int64_t *x_ptr, int64_t *start_ptr,
                                                         int64_t *size_ptr, int64_t *y_indices_ptr,
                                                         bool *y_values_ptr, int64_t *out_shape_ptr,
                                                         int64_t *sum_count_ptr,
                                                         size_t input_nnz_, size_t num_dim_, size_t out_size_,
                                                         uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSlice<Complex<float>, int64_t>(const int64_t *indices_ptr,
                                                                   const Complex<float> *values_ptr,
                                                                   const int64_t *x_ptr, int64_t *start_ptr,
                                                                   int64_t *size_ptr, int64_t *y_indices_ptr,
                                                                   Complex<float> *y_values_ptr, int64_t *out_shape_ptr,
                                                                   int64_t *sum_count_ptr, size_t input_nnz_,
                                                                   size_t num_dim_, size_t out_size_,
                                                                   uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSlice<Complex<double>, int64_t>(const int64_t *indices_ptr,
                                                                    const Complex<double> *values_ptr,
                                                                    const int64_t *x_ptr, int64_t *start_ptr,
                                                                    int64_t *size_ptr, int64_t *y_indices_ptr,
                                                                    Complex<double> *y_values_ptr,
                                                                    int64_t *out_shape_ptr, int64_t *sum_count_ptr,
                                                                    size_t input_nnz_, size_t num_dim_,
                                                                    size_t out_size_,
                                                                    uint32_t device_id, cudaStream_t cuda_stream);
