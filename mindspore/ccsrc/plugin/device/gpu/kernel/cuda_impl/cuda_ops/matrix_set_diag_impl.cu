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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/matrix_set_diag_impl.cuh"
#include <stdint.h>
#include <cuda_runtime.h>
#include <algorithm>

__inline__ __device__ int CalDiagOffset(int d, int max_diag_len, const int inner_row, const int inner_col,
                                        const bool right_align_super_diagonal, const bool right_align_sub_diagonal) {
  const bool right_align = (d >= 0 && right_align_super_diagonal) || (d <= 0 && right_align_sub_diagonal);
  const int diag_len = min(inner_row + min(0, d), inner_col - max(0, d));
  const int offset = (right_align) ? (max_diag_len - diag_len) : 0;
  return offset;
}

template <typename T>
__global__ void MatrixSetDiagKernel(const int outer_batch, const int inner_row, const int inner_col,
                                    const int num_diags, const int max_diag_len, const int lower_index,
                                    const int upper_index, const bool right_align_super_diagonal,
                                    const bool right_align_sub_diagonal, const bool is_single_diag, const T *diag_addr,
                                    T *output_addr) {
  int count = outer_batch * inner_row * inner_col;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    int batch = i / (inner_row * inner_col);
    int row = (i - batch * inner_row * inner_col) / inner_col;
    int col = (i - batch * inner_row * inner_col) % inner_col;
    int d = col - row;
    if (is_single_diag) {
      if (d == upper_index) {
        output_addr[i] = diag_addr[batch * max_diag_len + col - max(upper_index, 0)];
      }
    } else {
      int diag_index = upper_index - d;
      int offset =
        CalDiagOffset(d, max_diag_len, inner_row, inner_col, right_align_super_diagonal, right_align_sub_diagonal);
      int index_in_diag = col - max(d, 0) + offset;
      if (d >= lower_index && d <= upper_index) {
        output_addr[i] = diag_addr[batch * num_diags * max_diag_len + diag_index * max_diag_len + index_in_diag];
      }
    }
  }
}

template <typename T>
void MatrixSetDiag(const int outer_batch, const int inner_row, const int inner_col, const int num_diags,
                   const int max_diag_len, const int lower_index, const int upper_index,
                   const bool right_align_super_diagonal, const bool right_align_sub_diagonal,
                   const bool is_single_diag, const T *diag_addr, T *output_addr, const uint32_t &device_id,
                   cudaStream_t cuda_stream) {
  int count = outer_batch * inner_row * inner_col;
  MatrixSetDiagKernel<<<CUDA_BLOCKS(device_id, count), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    outer_batch, inner_row, inner_col, num_diags, max_diag_len, lower_index, upper_index, right_align_super_diagonal,
    right_align_sub_diagonal, is_single_diag, diag_addr, output_addr);
}

template CUDA_LIB_EXPORT void MatrixSetDiag<uint8_t>(const int outer_batch, const int inner_row, const int inner_col,
                                                     const int num_diags, const int max_diag_len, const int lower_index,
                                                     const int upper_index, const bool right_align_super_diagonal,
                                                     const bool right_align_sub_diagonal, const bool is_single_diag,
                                                     const uint8_t *diag_addr, uint8_t *output_addr,
                                                     const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void MatrixSetDiag<uint16_t>(const int outer_batch, const int inner_row, const int inner_col,
                                                      const int num_diags, const int max_diag_len,
                                                      const int lower_index, const int upper_index,
                                                      const bool right_align_super_diagonal,
                                                      const bool right_align_sub_diagonal, const bool is_single_diag,
                                                      const uint16_t *diag_addr, uint16_t *output_addr,
                                                      const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void MatrixSetDiag<uint32_t>(const int outer_batch, const int inner_row, const int inner_col,
                                                      const int num_diags, const int max_diag_len,
                                                      const int lower_index, const int upper_index,
                                                      const bool right_align_super_diagonal,
                                                      const bool right_align_sub_diagonal, const bool is_single_diag,
                                                      const uint32_t *diag_addr, uint32_t *output_addr,
                                                      const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void MatrixSetDiag<uint64_t>(const int outer_batch, const int inner_row, const int inner_col,
                                                      const int num_diags, const int max_diag_len,
                                                      const int lower_index, const int upper_index,
                                                      const bool right_align_super_diagonal,
                                                      const bool right_align_sub_diagonal, const bool is_single_diag,
                                                      const uint64_t *diag_addr, uint64_t *output_addr,
                                                      const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void MatrixSetDiag<int8_t>(const int outer_batch, const int inner_row, const int inner_col,
                                                    const int num_diags, const int max_diag_len, const int lower_index,
                                                    const int upper_index, const bool right_align_super_diagonal,
                                                    const bool right_align_sub_diagonal, const bool is_single_diag,
                                                    const int8_t *diag_addr, int8_t *output_addr,
                                                    const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void MatrixSetDiag<int16_t>(const int outer_batch, const int inner_row, const int inner_col,
                                                     const int num_diags, const int max_diag_len, const int lower_index,
                                                     const int upper_index, const bool right_align_super_diagonal,
                                                     const bool right_align_sub_diagonal, const bool is_single_diag,
                                                     const int16_t *diag_addr, int16_t *output_addr,
                                                     const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void MatrixSetDiag<int>(const int outer_batch, const int inner_row, const int inner_col,
                                                 const int num_diags, const int max_diag_len, const int lower_index,
                                                 const int upper_index, const bool right_align_super_diagonal,
                                                 const bool right_align_sub_diagonal, const bool is_single_diag,
                                                 const int *diag_addr, int *output_addr, const uint32_t &device_id,
                                                 cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void MatrixSetDiag<int64_t>(const int outer_batch, const int inner_row, const int inner_col,
                                                     const int num_diags, const int max_diag_len, const int lower_index,
                                                     const int upper_index, const bool right_align_super_diagonal,
                                                     const bool right_align_sub_diagonal, const bool is_single_diag,
                                                     const int64_t *diag_addr, int64_t *output_addr,
                                                     const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void MatrixSetDiag<half>(const int outer_batch, const int inner_row, const int inner_col,
                                                  const int num_diags, const int max_diag_len, const int lower_index,
                                                  const int upper_index, const bool right_align_super_diagonal,
                                                  const bool right_align_sub_diagonal, const bool is_single_diag,
                                                  const half *diag_addr, half *output_addr, const uint32_t &device_id,
                                                  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void MatrixSetDiag<float>(const int outer_batch, const int inner_row, const int inner_col,
                                                   const int num_diags, const int max_diag_len, const int lower_index,
                                                   const int upper_index, const bool right_align_super_diagonal,
                                                   const bool right_align_sub_diagonal, const bool is_single_diag,
                                                   const float *diag_addr, float *output_addr,
                                                   const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void MatrixSetDiag<double>(const int outer_batch, const int inner_row, const int inner_col,
                                                    const int num_diags, const int max_diag_len, const int lower_index,
                                                    const int upper_index, const bool right_align_super_diagonal,
                                                    const bool right_align_sub_diagonal, const bool is_single_diag,
                                                    const double *diag_addr, double *output_addr,
                                                    const uint32_t &device_id, cudaStream_t cuda_stream);
