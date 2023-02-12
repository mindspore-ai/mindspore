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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sparse_slice_grad_impl.cuh"
#include <algorithm>
#include <complex>
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

template <typename DataType, typename IndexType>
__global__ void SparseSliceGradKernel(const DataType *x_ptr, const IndexType *indices_ptr,
                                      const IndexType *start_ptr, const IndexType *new_indices_ptr,
                                      DataType *y_ptr, size_t *num_propagated, size_t input_nnz_,
                                      size_t output_nnz_, size_t num_dims_) {
  size_t input_nz = blockIdx.x * blockDim.x + threadIdx.x;
  if (input_nz < input_nnz_) {
    y_ptr[input_nz] = DataType(0);
    for (IndexType output_nz = 0; output_nz < output_nnz_; ++output_nz) {
      bool is_same = true;
       for (int d = 0; d < num_dims_; ++d) {
        const IndexType a = indices_ptr[input_nz * num_dims_ + d];
        const IndexType b = new_indices_ptr[output_nz * num_dims_ + d];
        const IndexType offset = start_ptr[d];
        if (b + offset != a) {
          is_same = false;
          break;
        }
       }
      if (is_same) {
        y_ptr[input_nz] = x_ptr[output_nz];
        MsAtomicAdd(num_propagated, size_t(1));
        break;
      }
    }
  }
}

template <typename DataType, typename IndexType>
CUDA_LIB_EXPORT void SparseSliceGrad(const DataType *x_ptr, const IndexType *indices_ptr, const IndexType *start_ptr,
                                     const IndexType *new_indices_ptr, DataType *y_ptr, size_t *num_propagated,
                                     size_t input_nnz_, size_t output_nnz_, size_t num_dim_, uint32_t device_id,
                                     cudaStream_t cuda_stream) {
  int threads_per_block = CUDA_THREADS(device_id);
  unsigned int grid_num = UP_DIV(input_nnz_ + 1, threads_per_block);
  SparseSliceGradKernel<<<grid_num, threads_per_block, 0, cuda_stream>>>(
    x_ptr, indices_ptr, start_ptr, new_indices_ptr, y_ptr, num_propagated, input_nnz_, output_nnz_, num_dim_);
}

template CUDA_LIB_EXPORT void SparseSliceGrad<int8_t, int64_t>(const int8_t *x_ptr, const int64_t *indices_ptr,
                                                               const int64_t *start_ptr, const int64_t *new_indices_ptr,
                                                               int8_t *y_ptr, size_t *num_propagated,
                                                               size_t input_nnz_, size_t output_nnz_, size_t num_dim_,
                                                               uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSliceGrad<int16_t, int64_t>(const int16_t *x_ptr, const int64_t *indices_ptr,
                                                                const int64_t *start_ptr,
                                                                const int64_t *new_indices_ptr,
                                                                int16_t *y_ptr, size_t *num_propagated,
                                                                size_t input_nnz_, size_t output_nnz_, size_t num_dim_,
                                                                uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSliceGrad<int32_t, int64_t>(const int32_t *x_ptr, const int64_t *indices_ptr,
                                                                const int64_t *start_ptr,
                                                                const int64_t *new_indices_ptr, int32_t *y_ptr,
                                                                size_t *num_propagated, size_t input_nnz_,
                                                                size_t output_nnz_, size_t num_dim_, uint32_t device_id,
                                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSliceGrad<int64_t, int64_t>(const int64_t *x_ptr, const int64_t *indices_ptr,
                                                               const int64_t *start_ptr,
                                                               const int64_t *new_indices_ptr, int64_t *y_ptr,
                                                               size_t *num_propagated, size_t input_nnz_,
                                                               size_t output_nnz_, size_t num_dim_, uint32_t device_id,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSliceGrad<uint8_t, int64_t>(const uint8_t *x_ptr, const int64_t *indices_ptr,
                                                                const int64_t *start_ptr,
                                                                const int64_t *new_indices_ptr, uint8_t *y_ptr,
                                                                size_t *num_propagated, size_t input_nnz_,
                                                                size_t output_nnz_, size_t num_dim_,
                                                                uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSliceGrad<uint16_t, int64_t>(const uint16_t *x_ptr, const int64_t *indices_ptr,
                                                                 const int64_t *start_ptr,
                                                                 const int64_t *new_indices_ptr, uint16_t *y_ptr,
                                                                 size_t *num_propagated, size_t input_nnz_,
                                                                 size_t output_nnz_, size_t num_dim_,
                                                                 uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSliceGrad<uint32_t, int64_t>(const uint32_t *x_ptr, const int64_t *indices_ptr,
                                                                 const int64_t *start_ptr,
                                                                 const int64_t *new_indices_ptr, uint32_t *y_ptr,
                                                                 size_t *num_propagated, size_t input_nnz_,
                                                                 size_t output_nnz_, size_t num_dim_,
                                                                 uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSliceGrad<uint64_t, int64_t>(const uint64_t *x_ptr, const int64_t *indices_ptr,
                                                                 const int64_t *start_ptr,
                                                                 const int64_t *new_indices_ptr, uint64_t *y_ptr,
                                                                 size_t *num_propagated, size_t input_nnz_,
                                                                 size_t output_nnz_, size_t num_dim_,
                                                                 uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSliceGrad<half, int64_t>(const half *x_ptr, const int64_t *indices_ptr,
                                                             const int64_t *start_ptr,
                                                             const int64_t *new_indices_ptr,
                                                             half *y_ptr, size_t *num_propagated, size_t input_nnz_,
                                                             size_t output_nnz_, size_t num_dim_, uint32_t device_id,
                                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSliceGrad<float, int64_t>(const float *x_ptr, const int64_t *indices_ptr,
                                                              const int64_t *start_ptr,
                                                              const int64_t *new_indices_ptr,
                                                              float *y_ptr, size_t *num_propagated, size_t input_nnz_,
                                                              size_t output_nnz_, size_t num_dim_, uint32_t device_id,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSliceGrad<double, int64_t>(const double *x_ptr, const int64_t *indices_ptr,
                                                               const int64_t *start_ptr,
                                                               const int64_t *new_indices_ptr, double *y_ptr,
                                                               size_t *num_propagated, size_t input_nnz_,
                                                               size_t output_nnz_, size_t num_dim_, uint32_t device_id,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSliceGrad<bool, int64_t>(const bool *x_ptr, const int64_t *indices_ptr,
                                                             const int64_t *start_ptr,
                                                             const int64_t *new_indices_ptr, bool *y_ptr,
                                                             size_t *num_propagated, size_t input_nnz_,
                                                             size_t output_nnz_, size_t num_dim_, uint32_t device_id,
                                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSliceGrad<Complex<float>, int64_t>(const Complex<float> *x_ptr,
                                                                        const int64_t *indices_ptr,
                                                                        const int64_t *start_ptr,
                                                                        const int64_t *new_indices_ptr,
                                                                        Complex<float> *y_ptr,
                                                                        size_t *num_propagated, size_t input_nnz_,
                                                                        size_t output_nnz_, size_t num_dim_,
                                                                        uint32_t device_id,
                                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseSliceGrad<Complex<double>, int64_t>(const Complex<double> *x_ptr,
                                                                        const int64_t *indices_ptr,
                                                                        const int64_t *start_ptr,
                                                                        const int64_t *new_indices_ptr,
                                                                        Complex<double> *y_ptr,
                                                                        size_t *num_propagated, size_t input_nnz_,
                                                                        size_t output_nnz_, size_t num_dim_,
                                                                        uint32_t device_id,
                                                                        cudaStream_t cuda_stream);
