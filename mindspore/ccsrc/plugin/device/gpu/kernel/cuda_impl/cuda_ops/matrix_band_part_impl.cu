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
#include "matrix_band_part_impl.cuh"
#include <cuda_runtime.h>
#include <algorithm>

template <typename T>
__global__ void MatrixBandPartDiagonalKernel(const size_t size, const T *input_ptr, const size_t non_zero_len,
                                             const size_t m, const size_t n, const int64_t lower, const int64_t upper,
                                             T *output_ptr, cudaStream_t cuda_stream) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    const size_t i = pos / non_zero_len;
    const size_t j = pos % non_zero_len;
    const size_t offset = i * m * n + j * n;
    // Diagonal
    output_ptr[offset + j] = input_ptr[offset + j];
  }
}

template <typename T>
__global__ void MatrixBandPartKernel(const size_t size, const T *input_ptr, const size_t m, const size_t n,
                                     const int64_t lower, const int64_t upper, T *output_ptr,
                                     cudaStream_t cuda_stream) {
  auto zero = static_cast<T>(0.0);
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    const size_t last_two_dim_offset = pos % (m * n);
    int64_t i = static_cast<int64_t>(last_two_dim_offset / n);
    int64_t j = static_cast<int64_t>(last_two_dim_offset % n);
    // Note: the type of i or j can not be size_t.
    if ((i - j) <= lower && (j - i) <= upper) {
      output_ptr[pos] = input_ptr[pos];
    } else {
      output_ptr[pos] = zero;
    }
  }
}

template <typename T>
void MatrixBandPart(const size_t output_outer_size, const T *input_ptr, const size_t m, const size_t n,
                    const int64_t lower, const int64_t upper, T *output_ptr, const uint32_t &device_id,
                    cudaStream_t cuda_stream) {
  if (lower == 0 && upper == 0) {
    // The non_zero_len is the length of the non zero element along the -2 axis, so it can skip the position with 0.
    size_t non_zero_len = std::min(m, lower + n);
    int size = output_outer_size * non_zero_len;
    MatrixBandPartDiagonalKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      size, input_ptr, non_zero_len, m, n, lower, upper, output_ptr, cuda_stream);
  } else {
    int size = output_outer_size * m * n;
    MatrixBandPartKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      size, input_ptr, m, n, lower, upper, output_ptr, cuda_stream);
  }
}

template CUDA_LIB_EXPORT void MatrixBandPart<int32_t>(const size_t output_outer_size, const int32_t *input_ptr,
                                                      const size_t m, const size_t n, const int64_t lower,
                                                      const int64_t upper, int32_t *output_ptr,
                                                      const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPart<int64_t>(const size_t output_outer_size, const int64_t *input_ptr,
                                                      const size_t m, const size_t n, const int64_t lower,
                                                      const int64_t upper, int64_t *output_ptr,
                                                      const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPart<half>(const size_t output_outer_size, const half *input_ptr,
                                                   const size_t m, const size_t n, const int64_t lower,
                                                   const int64_t upper, half *output_ptr, const uint32_t &device_id,
                                                   cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPart<float>(const size_t output_outer_size, const float *input_ptr,
                                                    const size_t m, const size_t n, const int64_t lower,
                                                    const int64_t upper, float *output_ptr, const uint32_t &device_id,
                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPart<double>(const size_t output_outer_size, const double *input_ptr,
                                                     const size_t m, const size_t n, const int64_t lower,
                                                     const int64_t upper, double *output_ptr, const uint32_t &device_id,
                                                     cudaStream_t cuda_stream);
