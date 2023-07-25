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
#include "matrix_diag_part_v3_impl.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

__device__ inline int64_t ComputeOffset(const int64_t diag_idx, const int64_t num_rows, const int64_t num_cols,
                                        const int64_t max_diag_len, const bool left_align_super_diag,
                                        const bool left_align_sub_diag) {
  if ((diag_idx >= 0 && left_align_super_diag) || (diag_idx <= 0 && left_align_sub_diag)) {
    return 0;
  }
  int64_t diag_len1 = num_cols - max(diag_idx, int64_t(0));
  int64_t diag_len2 = num_rows + min(diag_idx, int64_t(0));
  return max_diag_len - min(diag_len1, diag_len2);
}

template <typename T>
__global__ void MatrixDiagPartV3Kernel(const T *matrix_ptr, const T *padding_value_ptr, T *diagnal_ptr,
                                       const int64_t num_rows, const int64_t num_cols, const int64_t upper_diag_index,
                                       const int64_t diag_size, const int64_t num_diag, const int64_t max_diag_len,
                                       const bool left_align_super_diag, const bool left_align_sub_diag) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < diag_size; idx += blockDim.x * gridDim.x) {
    const int64_t batch_diag_index = idx / max_diag_len;
    const int64_t ith_diag = idx % max_diag_len;
    const int64_t batch = batch_diag_index / num_diag;
    const int64_t diag_index = upper_diag_index - (batch_diag_index % num_diag);
    const int64_t offset =
      ComputeOffset(diag_index, num_rows, num_cols, max_diag_len, left_align_super_diag, left_align_sub_diag);
    const int64_t y_idx = ith_diag + max(static_cast<int64_t>(0), -diag_index) - offset;
    const int64_t x_idx = ith_diag + max(static_cast<int64_t>(0), diag_index) - offset;
    if ((0 <= y_idx && y_idx < num_rows) && (0 <= x_idx && x_idx < num_cols)) {
      diagnal_ptr[idx] = matrix_ptr[batch * num_rows * num_cols + y_idx * num_cols + x_idx];
    } else {
      diagnal_ptr[idx] = *padding_value_ptr;
    }
  }
}

template <typename T>
cudaError_t MatrixDiagPartV3(const T *matrix_ptr, const T *padding_value_ptr, T *diagnal_ptr, const int64_t num_rows,
                             const int64_t num_cols, const int64_t lower_diag_idx, const int64_t upper_diag_index,
                             const int64_t diag_size, const int64_t max_diag_len, const bool left_align_super_diag,
                             const bool left_align_sub_diag, uint32_t device_id, cudaStream_t cuda_stream) {
  const int64_t num_diag = upper_diag_index - lower_diag_idx + 1;
  MatrixDiagPartV3Kernel<<<CUDA_BLOCKS(device_id, diag_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    matrix_ptr, padding_value_ptr, diagnal_ptr, num_rows, num_cols, upper_diag_index, diag_size, num_diag, max_diag_len,
    left_align_super_diag, left_align_sub_diag);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t MatrixDiagPartV3<int8_t>(
  const int8_t *matrix_ptr, const int8_t *padding_value_ptr, int8_t *diagnal_ptr, const int64_t num_rows,
  const int64_t num_cols, const int64_t lower_diag_idx, const int64_t upper_diag_index, const int64_t diag_size,
  const int64_t max_diag_len, const bool left_align_super_diag, const bool left_align_sub_diag, uint32_t device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t MatrixDiagPartV3<uint8_t>(
  const uint8_t *matrix_ptr, const uint8_t *padding_value_ptr, uint8_t *diagnal_ptr, const int64_t num_rows,
  const int64_t num_cols, const int64_t lower_diag_idx, const int64_t upper_diag_index, const int64_t diag_size,
  const int64_t max_diag_len, const bool left_align_super_diag, const bool left_align_sub_diag, uint32_t device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t MatrixDiagPartV3<int16_t>(
  const int16_t *matrix_ptr, const int16_t *padding_value_ptr, int16_t *diagnal_ptr, const int64_t num_rows,
  const int64_t num_cols, const int64_t lower_diag_idx, const int64_t upper_diag_index, const int64_t diag_size,
  const int64_t max_diag_len, const bool left_align_super_diag, const bool left_align_sub_diag, uint32_t device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t MatrixDiagPartV3<uint16_t>(
  const uint16_t *matrix_ptr, const uint16_t *padding_value_ptr, uint16_t *diagnal_ptr, const int64_t num_rows,
  const int64_t num_cols, const int64_t lower_diag_idx, const int64_t upper_diag_index, const int64_t diag_size,
  const int64_t max_diag_len, const bool left_align_super_diag, const bool left_align_sub_diag, uint32_t device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t MatrixDiagPartV3<int32_t>(
  const int32_t *matrix_ptr, const int32_t *padding_value_ptr, int32_t *diagnal_ptr, const int64_t num_rows,
  const int64_t num_cols, const int64_t lower_diag_idx, const int64_t upper_diag_index, const int64_t diag_size,
  const int64_t max_diag_len, const bool left_align_super_diag, const bool left_align_sub_diag, uint32_t device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t MatrixDiagPartV3<uint32_t>(
  const uint32_t *matrix_ptr, const uint32_t *padding_value_ptr, uint32_t *diagnal_ptr, const int64_t num_rows,
  const int64_t num_cols, const int64_t lower_diag_idx, const int64_t upper_diag_index, const int64_t diag_size,
  const int64_t max_diag_len, const bool left_align_super_diag, const bool left_align_sub_diag, uint32_t device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t MatrixDiagPartV3<int64_t>(
  const int64_t *matrix_ptr, const int64_t *padding_value_ptr, int64_t *diagnal_ptr, const int64_t num_rows,
  const int64_t num_cols, const int64_t lower_diag_idx, const int64_t upper_diag_index, const int64_t diag_size,
  const int64_t max_diag_len, const bool left_align_super_diag, const bool left_align_sub_diag, uint32_t device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t MatrixDiagPartV3<uint64_t>(
  const uint64_t *matrix_ptr, const uint64_t *padding_value_ptr, uint64_t *diagnal_ptr, const int64_t num_rows,
  const int64_t num_cols, const int64_t lower_diag_idx, const int64_t upper_diag_index, const int64_t diag_size,
  const int64_t max_diag_len, const bool left_align_super_diag, const bool left_align_sub_diag, uint32_t device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t
MatrixDiagPartV3<half>(const half *matrix_ptr, const half *padding_value_ptr, half *diagnal_ptr, const int64_t num_rows,
                       const int64_t num_cols, const int64_t lower_diag_idx, const int64_t upper_diag_index,
                       const int64_t diag_size, const int64_t max_diag_len, const bool left_align_super_diag,
                       const bool left_align_sub_diag, uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t MatrixDiagPartV3<float>(
  const float *matrix_ptr, const float *padding_value_ptr, float *diagnal_ptr, const int64_t num_rows,
  const int64_t num_cols, const int64_t lower_diag_idx, const int64_t upper_diag_index, const int64_t diag_size,
  const int64_t max_diag_len, const bool left_align_super_diag, const bool left_align_sub_diag, uint32_t device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t MatrixDiagPartV3<double>(
  const double *matrix_ptr, const double *padding_value_ptr, double *diagnal_ptr, const int64_t num_rows,
  const int64_t num_cols, const int64_t lower_diag_idx, const int64_t upper_diag_index, const int64_t diag_size,
  const int64_t max_diag_len, const bool left_align_super_diag, const bool left_align_sub_diag, uint32_t device_id,
  cudaStream_t cuda_stream);
