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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sparse_addmm_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "include/cuda_runtime.h"

template <typename T, typename S>
__global__ void SparseAddDenseKernel(const S *input_indices, const T *input_values, int64_t input_values_num_,
                                     int64_t mat2_col_, const T *mat2, const T *mat3, const T *alpha, const T *beta,
                                     T *output, int64_t output_row_, int64_t output_col_) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < output_row_ * output_col_;
       pos += blockDim.x * gridDim.x) {
    output[pos] = (*beta) * mat3[pos];
  }

  return;
}

template <typename T, typename S>
__global__ void SparseMulDenseKernel(const S *input_indices, const T *input_values, int64_t input_values_num_,
                                     int64_t mat2_col_, const T *mat2, const T *mat3, const T *alpha, const T *beta,
                                     T *output, int64_t output_row_, int64_t output_col_) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < input_values_num_ * mat2_col_;
       pos += blockDim.x * gridDim.x) {
    int64_t indice_pos = static_cast<int64_t>(pos) / mat2_col_;
    int64_t j = static_cast<int64_t>(pos) % mat2_col_;
    S col = input_indices[2 * indice_pos + 1];  // col -> col of input value, row of mat2
    S idx = input_indices[2 * indice_pos];      // idx -> row of input value,
    T value = input_values[indice_pos] * mat2[col * mat2_col_ + j] * (*alpha);
    MsAtomicAdd(&output[idx * mat2_col_ + j], value);
  }
  return;
}

template <typename T, typename S>
cudaError_t SparseAddmm(const S *input_indices, const T *input_values, int64_t input_values_num_, const T *mat2,
                        const T *mat3, int64_t mat2_col_, const T *alpha, const T *beta, T *output, int64_t output_row_,
                        int64_t output_col_, const uint32_t &device_id, cudaStream_t cuda_stream) {
  SparseAddDenseKernel<<<CUDA_BLOCKS(device_id, output_row_ * output_col_), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input_indices, input_values, input_values_num_, mat2_col_, mat2, mat3, alpha, beta, output, output_row_,
    output_col_);
  SparseMulDenseKernel<<<CUDA_BLOCKS(device_id, input_values_num_ * mat2_col_), CUDA_THREADS(device_id), 0,
                         cuda_stream>>>(input_indices, input_values, input_values_num_, mat2_col_, mat2, mat3, alpha,
                                        beta, output, output_row_, output_col_);
  return GetCudaStatus();
}

#define GPU_SPARSE_ADDMM_EXPORT_REGISTER(T, S)                                                              \
  template CUDA_LIB_EXPORT cudaError_t SparseAddmm<T, S>(                                                   \
    const S *input_indices, const T *input_values, int64_t input_values_num_, const T *mat2, const T *mat3, \
    int64_t mat2_col_, const T *alpha, const T *beta, T *output, int64_t output_row_, int64_t output_col_,  \
    const uint32_t &device_id, cudaStream_t cuda_stream);

GPU_SPARSE_ADDMM_EXPORT_REGISTER(int8_t, int)
GPU_SPARSE_ADDMM_EXPORT_REGISTER(int16_t, int)
GPU_SPARSE_ADDMM_EXPORT_REGISTER(int, int)
GPU_SPARSE_ADDMM_EXPORT_REGISTER(int64_t, int)
GPU_SPARSE_ADDMM_EXPORT_REGISTER(uint8_t, int)
GPU_SPARSE_ADDMM_EXPORT_REGISTER(uint16_t, int)
GPU_SPARSE_ADDMM_EXPORT_REGISTER(uint32_t, int)
GPU_SPARSE_ADDMM_EXPORT_REGISTER(uint64_t, int)
GPU_SPARSE_ADDMM_EXPORT_REGISTER(float, int)
GPU_SPARSE_ADDMM_EXPORT_REGISTER(double, int)

GPU_SPARSE_ADDMM_EXPORT_REGISTER(int8_t, int64_t)
GPU_SPARSE_ADDMM_EXPORT_REGISTER(int16_t, int64_t)
GPU_SPARSE_ADDMM_EXPORT_REGISTER(int, int64_t)
GPU_SPARSE_ADDMM_EXPORT_REGISTER(int64_t, int64_t)
GPU_SPARSE_ADDMM_EXPORT_REGISTER(uint8_t, int64_t)
GPU_SPARSE_ADDMM_EXPORT_REGISTER(uint16_t, int64_t)
GPU_SPARSE_ADDMM_EXPORT_REGISTER(uint32_t, int64_t)
GPU_SPARSE_ADDMM_EXPORT_REGISTER(uint64_t, int64_t)
GPU_SPARSE_ADDMM_EXPORT_REGISTER(float, int64_t)
GPU_SPARSE_ADDMM_EXPORT_REGISTER(double, int64_t)
