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
#include "matrix_diag_part_impl.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include "utils/complex.h"

template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T>
__global__ void MatrixDiagPartKernel(const size_t size, const T *input_matrix_addr, const size_t m, const size_t n,
                                     const int64_t l, const int64_t u, const size_t num_diags,
                                     const size_t max_diag_len, const int64_t la, const int64_t ua, T *padding_value,
                                     T *output_addr, cudaStream_t cuda_stream) {
  int64_t dest_inner_matrix_len = num_diags * max_diag_len;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    const int64_t i = pos / dest_inner_matrix_len;
    const int64_t j = u - (pos % dest_inner_matrix_len) / max_diag_len;
    const int64_t k = (pos % dest_inner_matrix_len) % max_diag_len;
    int64_t current_diag_len = j >= 0 ? min(n - j, m) : min(m + j, n);
    int64_t current_pad_len = max_diag_len - current_diag_len;
    // Pad left by default (0:right, 1:left)
    bool pad_left = (la == 0 && j > 0) || (ua == 0 && j < 0);
    // Set none-padding values, l means current diag col index
    // Source pos, k offset, only effective when pad left
    int64_t k_offset = (pad_left && k >= current_pad_len) ? k - current_pad_len : k;

    // Calculate source offset row/col offset
    size_t row_index = j >= 0 ? j + k_offset : k_offset;
    size_t col_index = j >= 0 ? k_offset : k_offset - j;
    size_t source_offset = i * m * n + col_index * n + row_index;
    // If current pos need pad, then the value is pad value
    bool current_pad_flag = (pad_left && k < current_pad_len) || (!pad_left && k >= current_diag_len);
    T current_pad_value = current_pad_flag ? *padding_value : *(input_matrix_addr + source_offset);
    int64_t j_index = u - j;
    size_t dest_offset = dest_inner_matrix_len * i + j_index * max_diag_len + k;
    *(output_addr + dest_offset) = current_pad_value;
  }
}

template <typename T>
void MatrixDiagPart(const size_t size, const T *input_matrix_addr, const size_t m, const size_t n, const int64_t l,
                    const int64_t u, const size_t num_diags, const size_t max_diag_len, const int64_t la,
                    const int64_t ua, T *padding_value, T *output_addr, cudaStream_t cuda_stream) {
  MatrixDiagPartKernel<<<GET_BLOCKS(size), GET_THREADS_MAXSIZE(size), 0, cuda_stream>>>(
    size, input_matrix_addr, m, n, l, u, num_diags, max_diag_len, la, ua, padding_value, output_addr, cuda_stream);
}

template void MatrixDiagPart<int32_t>(const size_t size, const int32_t *input_matrix_addr, const size_t m,
                                      const size_t n, const int64_t l, const int64_t u, const size_t num_diags,
                                      const size_t max_diag_len, const int64_t la, const int64_t ua,
                                      int32_t *padding_value, int32_t *output_addr, cudaStream_t cuda_stream);
template void MatrixDiagPart<int64_t>(const size_t size, const int64_t *input_matrix_addr, const size_t m,
                                      const size_t n, const int64_t l, const int64_t u, const size_t num_diags,
                                      const size_t max_diag_len, const int64_t la, const int64_t ua,
                                      int64_t *padding_value, int64_t *output_addr, cudaStream_t cuda_stream);
template void MatrixDiagPart<float>(const size_t size, const float *input_matrix_addr, const size_t m, const size_t n,
                                    const int64_t l, const int64_t u, const size_t num_diags, const size_t max_diag_len,
                                    const int64_t la, const int64_t ua, float *padding_value, float *output_addr,
                                    cudaStream_t cuda_stream);
template void MatrixDiagPart<double>(const size_t size, const double *input_matrix_addr, const size_t m, const size_t n,
                                     const int64_t l, const int64_t u, const size_t num_diags,
                                     const size_t max_diag_len, const int64_t la, const int64_t ua,
                                     double *padding_value, double *output_addr, cudaStream_t cuda_stream);
