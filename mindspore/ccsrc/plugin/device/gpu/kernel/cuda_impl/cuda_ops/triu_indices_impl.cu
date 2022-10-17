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

#include "triu_indices_impl.cuh"

template <typename T>
__global__ void TriuIndices(const int64_t col_offset, const int64_t m_first_row, const int64_t col,
                            const int64_t rectangle_size, const size_t triu_size, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < triu_size; pos += blockDim.x * gridDim.x) {
    int64_t row_idx, col_idx;
    if (pos < rectangle_size) {
      row_idx = pos / col;
      col_idx = pos % col;
    } else {
      int64_t t_first_row = m_first_row << 1;
      auto t_bottom_row = -1 - t_first_row;
      int64_t idx = pos - rectangle_size;
      double t_sqrt = sqrt(static_cast<double>(t_bottom_row * t_bottom_row - (idx << 3)));
      row_idx = __double2ll_rd((-t_bottom_row - t_sqrt) / 2);
      col_idx = idx - ((t_first_row - row_idx + 1) * row_idx >> 1) + row_idx;
      row_idx += rectangle_size / col;
    }
    col_idx += col_offset;

    output[pos] = static_cast<T>(row_idx);
    output[pos + triu_size] = static_cast<T>(col_idx);
  }
}

template <typename T>
void CalTriuIndices(const int64_t col_offset, const int64_t m_first_row, const int64_t col,
                    const int64_t rectangle_size, const size_t triu_size, T *output, const uint32_t &device_id,
                    cudaStream_t cuda_stream) {
  TriuIndices<<<CUDA_BLOCKS(device_id, triu_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    col_offset, m_first_row, col, rectangle_size, triu_size, output);
  return;
}

template CUDA_LIB_EXPORT void CalTriuIndices<int32_t>(const int64_t row_offset, const int64_t m_first_row,
                                                      const int64_t col, const int64_t trapezoid_size,
                                                      const size_t triu_size, int32_t *output,
                                                      const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalTriuIndices<int64_t>(const int64_t row_offset, const int64_t m_first_row,
                                                      const int64_t col, const int64_t trapezoid_size,
                                                      const size_t triu_size, int64_t *output,
                                                      const uint32_t &device_id, cudaStream_t cuda_stream);
