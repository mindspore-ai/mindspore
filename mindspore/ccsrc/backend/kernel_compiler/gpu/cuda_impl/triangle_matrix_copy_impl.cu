/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "triangle_matrix_copy_impl.cuh"
template <typename T>
__global__ void TriangleMatrixCopyKernel(const T *input, T *output, cublasFillMode_t uplo,
                                         const size_t count, const size_t ldb, const size_t m) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    size_t batchIdx = i / (ldb * m);
    size_t row = (i - batchIdx * ldb * m) / m;
    size_t col = (i - batchIdx * ldb * m) % m;

    // If fill mode is 'CUBLAS_FILL_MODE_UPPER', the upper half of the matrix should be all 0;
    // If fill mode is 'CUBLAS_FILL_MODE_LOWER', the lower half of the matrix should be all 0;
    if (uplo == CUBLAS_FILL_MODE_UPPER) {
      if (col > row) {
        output[i] = 0;
      } else {
        output[i] = input[i];
      }
    } else if (uplo == CUBLAS_FILL_MODE_LOWER) {
      if (col < row) {
        output[i] = 0;
      } else {
        output[i] = input[i];
      }
    }
  }
}

template <typename T>
void TriangleMatrixCopy(const T *input, T *output, cublasFillMode_t uplo,
                        const size_t count, const size_t ldb, const size_t m, cudaStream_t cuda_stream) {
  TriangleMatrixCopyKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, uplo, count, ldb, m);
  return;
}

template void TriangleMatrixCopy<float>(const float *input, float *output, cublasFillMode_t uplo, const size_t count,
                                        const size_t ldb, const size_t m, cudaStream_t cuda_stream);
template void TriangleMatrixCopy<half>(const half *input, half *output, cublasFillMode_t uplo, const size_t count,
                                       const size_t ldb, const size_t m, cudaStream_t cuda_stream);
