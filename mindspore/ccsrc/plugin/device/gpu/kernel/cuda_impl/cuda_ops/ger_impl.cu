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

#include "ger_impl.cuh"

template <typename T>
__global__ void Ger(const size_t size, const T *row_input, const T *col_input, const size_t matrix_row,
                    const size_t matrix_col, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int matrix_size = matrix_row * matrix_col;
    int row = pos % matrix_size / matrix_col;
    int col = pos % matrix_size % matrix_col;
    output[pos] = static_cast<T>(row_input[col] * col_input[row]);
  }
  return;
}

template <typename T>
cudaError_t CalGer(const size_t size, const T *row_input, const T *col_input, const size_t matrix_row,
                   const size_t matrix_col, T *output, const uint32_t &device_id, cudaStream_t cuda_stream) {
  Ger<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, row_input, col_input, matrix_row,
                                                                                 matrix_col, output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalGer<half>(const size_t size, const half *row_input, const half *col_input,
                                                  const size_t matrix_row, const size_t matrix_col, half *output,
                                                  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalGer<float>(const size_t size, const float *row_input, const float *col_input,
                                                   const size_t matrix_row, const size_t matrix_col, float *output,
                                                   const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalGer<double>(const size_t size, const double *row_input, const double *col_input,
                                                    const size_t matrix_row, const size_t matrix_col, double *output,
                                                    const uint32_t &device_id, cudaStream_t cuda_stream);
