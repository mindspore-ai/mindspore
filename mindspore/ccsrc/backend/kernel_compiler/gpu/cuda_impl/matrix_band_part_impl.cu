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
#include "utils/complex.h"

template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T>
__global__ void MatrixBandPartKernel(const size_t size, const T *input_matrix_addr, const size_t m, const size_t n,
                                     const int64_t l, const int64_t u, T *output_addr, cudaStream_t cuda_stream) {
  size_t diag_len = min(m, l + n);
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    const size_t i = pos / diag_len;
    const size_t j = pos % diag_len;
    const size_t s = j < l ? 0 : j - l;
    // When i = n - u, end is n -1, because end pos is start from 0
    const size_t e = j >= n - u ? n - 1 : j + u;
    const size_t offset = i * m * n + j * n;
    for (size_t x = s; x <= e; x++) {
      *(output_addr + offset + x) = *(input_matrix_addr + offset + x);
    }
  }
}

template <typename T>
void MatrixBandPart(const size_t size, const T *input_matrix_addr, const size_t m, const size_t n, const int64_t l,
                    const int64_t u, T *output_addr, cudaStream_t cuda_stream) {
  MatrixBandPartKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input_matrix_addr, m, n, l, u,
                                                                          output_addr, cuda_stream);
}

template void MatrixBandPart<int32_t>(const size_t size, const int32_t *input_matrix_addr, const size_t m,
                                      const size_t n, const int64_t l, const int64_t u, int32_t *output_addr,
                                      cudaStream_t cuda_stream);
template void MatrixBandPart<int64_t>(const size_t size, const int64_t *input_matrix_addr, const size_t m,
                                      const size_t n, const int64_t l, const int64_t u, int64_t *output_addr,
                                      cudaStream_t cuda_stream);
template void MatrixBandPart<float>(const size_t size, const float *input_matrix_addr, const size_t m, const size_t n,
                                    const int64_t l, const int64_t u, float *output_addr, cudaStream_t cuda_stream);
template void MatrixBandPart<double>(const size_t size, const double *input_matrix_addr, const size_t m, const size_t n,
                                     const int64_t l, const int64_t u, double *output_addr, cudaStream_t cuda_stream);
