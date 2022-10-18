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

#include "tril_indices_impl.cuh"

template <typename T>
__global__ void TrilIndices(const int64_t row_offset, const int64_t m_first_row, const int64_t col,
                            const int64_t trapezoid_size, const size_t tril_size, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < tril_size; pos += blockDim.x * gridDim.x) {
    int64_t row_idx, col_idx;
    if (pos < trapezoid_size) {
      int64_t t_first_row = m_first_row << 1;
      auto t_bottom_row = t_first_row - 1;
      double t_sqrt = sqrt(static_cast<double>(t_bottom_row * t_bottom_row + (pos << 3)));
      row_idx = __double2ll_rd((-t_bottom_row + t_sqrt) / 2);
      col_idx = pos - ((t_first_row + row_idx - 1) * row_idx >> 1);
    } else {
      auto surplus = pos - trapezoid_size;
      row_idx = surplus / col + col - m_first_row + 1;
      col_idx = surplus % col;
    }
    row_idx += row_offset;

    output[pos] = static_cast<T>(row_idx);
    output[pos + tril_size] = static_cast<T>(col_idx);
  }
}

template <typename T>
void CalTrilIndices(const int64_t row_offset, const int64_t m_first_row, const int64_t col,
                    const int64_t trapezoid_size, const size_t tril_size, T *output, const uint32_t &device_id,
                    cudaStream_t cuda_stream) {
  TrilIndices<<<CUDA_BLOCKS(device_id, tril_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    row_offset, m_first_row, col, trapezoid_size, tril_size, output);
  return;
}

template CUDA_LIB_EXPORT void CalTrilIndices<int32_t>(const int64_t row_offset, const int64_t m_first_row,
                                                      const int64_t col, const int64_t trapezoid_size,
                                                      const size_t tril_size, int32_t *output,
                                                      const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalTrilIndices<int64_t>(const int64_t row_offset, const int64_t m_first_row,
                                                      const int64_t col, const int64_t trapezoid_size,
                                                      const size_t tril_size, int64_t *output,
                                                      const uint32_t &device_id, cudaStream_t cuda_stream);
