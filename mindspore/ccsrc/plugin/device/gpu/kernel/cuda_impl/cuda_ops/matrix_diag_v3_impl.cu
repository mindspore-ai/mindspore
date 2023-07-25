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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/matrix_diag_v3_impl.cuh"
#include <algorithm>

__device__ inline int ComputeOffset(int diag_idx, int num_rows, int num_cols, int max_diag_len,
                                    bool left_align_super_diag, bool left_align_sub_diag) {
  bool left_align = (diag_idx >= 0 && left_align_super_diag) || (diag_idx <= 0 && left_align_sub_diag);
  if (left_align) {
    return 0;
  }
  int diag_len1 = num_cols - max(diag_idx, 0);
  int diag_len2 = num_rows + min(diag_idx, 0);
  return max_diag_len - min(diag_len1, diag_len2);
}

template <typename DataType>
__global__ void MatrixDiagV3Kernel(const DataType *x_ptr, const DataType *padding_value_ptr, DataType *y_ptr,
                                   int y_size, int num_rows, int num_cols, int lower_diag_index, int upper_diag_index,
                                   int diag_batch_len, int max_diag_len, bool left_align_super_diag,
                                   bool left_align_sub_diag) {
  DataType padding_value = *padding_value_ptr;
  int start_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  int step = static_cast<int>(blockDim.x * gridDim.x);
  for (int idx = start_idx; idx < y_size; idx += step) {
    int batch_row_idx = idx / num_cols;
    int col_idx = idx - batch_row_idx * num_cols;
    int batch_idx = batch_row_idx / num_rows;
    int row_idx = batch_row_idx - batch_idx * num_rows;
    int diag_idx = col_idx - row_idx;
    if (lower_diag_index <= diag_idx && diag_idx <= upper_diag_index) {
      int offset =
        ComputeOffset(diag_idx, num_rows, num_cols, max_diag_len, left_align_super_diag, left_align_sub_diag);
      int diag_row_idx = upper_diag_index - diag_idx;
      int diag_col_idx = col_idx - max(diag_idx, 0) + offset;
      y_ptr[idx] = x_ptr[batch_idx * diag_batch_len + diag_row_idx * max_diag_len + diag_col_idx];
    } else {
      y_ptr[idx] = padding_value;
    }
  }
}

template <typename DataType>
cudaError_t MatrixDiagV3(const DataType *x_ptr, const DataType *padding_value_ptr, DataType *y_ptr, int64_t y_size,
                         int64_t num_rows, int64_t num_cols, int64_t lower_diag_index, int64_t upper_diag_index,
                         int64_t max_diag_len, bool left_align_super_diag, bool left_align_sub_diag, uint32_t device_id,
                         cudaStream_t cuda_stream) {
  int64_t diag_batch_len = (upper_diag_index - lower_diag_index + 1) * max_diag_len;
  MatrixDiagV3Kernel<<<CUDA_BLOCKS(device_id, y_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    x_ptr, padding_value_ptr, y_ptr, static_cast<int>(y_size), static_cast<int>(num_rows), static_cast<int>(num_cols),
    static_cast<int>(lower_diag_index), static_cast<int>(upper_diag_index), static_cast<int>(diag_batch_len),
    static_cast<int>(max_diag_len), left_align_super_diag, left_align_sub_diag);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t MatrixDiagV3<int8_t>(const int8_t *x_ptr, const int8_t *padding_value_ptr,
                                                          int8_t *y_ptr, int64_t y_size, int64_t num_rows,
                                                          int64_t num_cols, int64_t lower_diag_index,
                                                          int64_t upper_diag_index, int64_t max_diag_len,
                                                          bool left_align_super_diag, bool left_align_sub_diag,
                                                          uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixDiagV3<int16_t>(const int16_t *x_ptr, const int16_t *padding_value_ptr,
                                                           int16_t *y_ptr, int64_t y_size, int64_t num_rows,
                                                           int64_t num_cols, int64_t lower_diag_index,
                                                           int64_t upper_diag_index, int64_t max_diag_len,
                                                           bool left_align_super_diag, bool left_align_sub_diag,
                                                           uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixDiagV3<int32_t>(const int32_t *x_ptr, const int32_t *padding_value_ptr,
                                                           int32_t *y_ptr, int64_t y_size, int64_t num_rows,
                                                           int64_t num_cols, int64_t lower_diag_index,
                                                           int64_t upper_diag_index, int64_t max_diag_len,
                                                           bool left_align_super_diag, bool left_align_sub_diag,
                                                           uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixDiagV3<int64_t>(const int64_t *x_ptr, const int64_t *padding_value_ptr,
                                                           int64_t *y_ptr, int64_t y_size, int64_t num_rows,
                                                           int64_t num_cols, int64_t lower_diag_index,
                                                           int64_t upper_diag_index, int64_t max_diag_len,
                                                           bool left_align_super_diag, bool left_align_sub_diag,
                                                           uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixDiagV3<uint8_t>(const uint8_t *x_ptr, const uint8_t *padding_value_ptr,
                                                           uint8_t *y_ptr, int64_t y_size, int64_t num_rows,
                                                           int64_t num_cols, int64_t lower_diag_index,
                                                           int64_t upper_diag_index, int64_t max_diag_len,
                                                           bool left_align_super_diag, bool left_align_sub_diag,
                                                           uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixDiagV3<uint16_t>(const uint16_t *x_ptr, const uint16_t *padding_value_ptr,
                                                            uint16_t *y_ptr, int64_t y_size, int64_t num_rows,
                                                            int64_t num_cols, int64_t lower_diag_index,
                                                            int64_t upper_diag_index, int64_t max_diag_len,
                                                            bool left_align_super_diag, bool left_align_sub_diag,
                                                            uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixDiagV3<uint32_t>(const uint32_t *x_ptr, const uint32_t *padding_value_ptr,
                                                            uint32_t *y_ptr, int64_t y_size, int64_t num_rows,
                                                            int64_t num_cols, int64_t lower_diag_index,
                                                            int64_t upper_diag_index, int64_t max_diag_len,
                                                            bool left_align_super_diag, bool left_align_sub_diag,
                                                            uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixDiagV3<uint64_t>(const uint64_t *x_ptr, const uint64_t *padding_value_ptr,
                                                            uint64_t *y_ptr, int64_t y_size, int64_t num_rows,
                                                            int64_t num_cols, int64_t lower_diag_index,
                                                            int64_t upper_diag_index, int64_t max_diag_len,
                                                            bool left_align_super_diag, bool left_align_sub_diag,
                                                            uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixDiagV3<half>(const half *x_ptr, const half *padding_value_ptr, half *y_ptr,
                                                        int64_t y_size, int64_t num_rows, int64_t num_cols,
                                                        int64_t lower_diag_index, int64_t upper_diag_index,
                                                        int64_t max_diag_len, bool left_align_super_diag,
                                                        bool left_align_sub_diag, uint32_t device_id,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixDiagV3<float>(const float *x_ptr, const float *padding_value_ptr,
                                                         float *y_ptr, int64_t y_size, int64_t num_rows,
                                                         int64_t num_cols, int64_t lower_diag_index,
                                                         int64_t upper_diag_index, int64_t max_diag_len,
                                                         bool left_align_super_diag, bool left_align_sub_diag,
                                                         uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixDiagV3<double>(const double *x_ptr, const double *padding_value_ptr,
                                                          double *y_ptr, int64_t y_size, int64_t num_rows,
                                                          int64_t num_cols, int64_t lower_diag_index,
                                                          int64_t upper_diag_index, int64_t max_diag_len,
                                                          bool left_align_super_diag, bool left_align_sub_diag,
                                                          uint32_t device_id, cudaStream_t cuda_stream);
