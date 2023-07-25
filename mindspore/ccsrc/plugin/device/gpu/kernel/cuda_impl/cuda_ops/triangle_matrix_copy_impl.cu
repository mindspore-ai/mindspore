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
#include "include/cuda_fp16.h"
template <typename T>
__global__ void TriangleMatrixCopyKernel(const T *input, T *output, bool clean, cublasFillMode_t uplo,
                                         const size_t count, const size_t ldb, const size_t m) {
  // If fill mode is 'CUBLAS_FILL_MODE_LOWER', if clean is false, the upper half and the positive diagonal of the matrix
  // should not be assigned any value, otherwise they should be assigned to 0.
  // If fill mode is 'CUBLAS_FILL_MODE_UPPER',if clean is false,  the lower half and the positive diagonal of the matrix
  // should not be assigned any value, otherwise they should be assigned to 0.
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    size_t batchIdx = i / (ldb * m);
    size_t row = (i - batchIdx * ldb * m) / m;
    size_t col = (i - batchIdx * ldb * m) % m;
    if (uplo == CUBLAS_FILL_MODE_UPPER) {
      if (col > row && !clean) {
        output[i] = input[i];
      } else if (col > row && clean) {
        output[i] = 0;
      }
    } else {
      if (col < row && !clean) {
        output[i] = input[i];
      } else if (col < row && clean) {
        output[i] = 0;
      }
    }
  }
}

template <typename T>
__global__ void MatrixCopyKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = input[i];
  }
}

template <typename T>
cudaError_t TriangleMatrixCopy(const T *input, T *output, bool clean, cublasFillMode_t uplo, const size_t count,
                               const size_t ldb, const size_t m, cudaStream_t cuda_stream) {
  TriangleMatrixCopyKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, clean, uplo, count, ldb,
                                                                               m);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t TriangleMatrixCopy<float>(const float *input, float *output, bool clean,
                                                               cublasFillMode_t uplo, const size_t count,
                                                               const size_t ldb, const size_t m,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t TriangleMatrixCopy<half>(const half *input, half *output, bool clean,
                                                              cublasFillMode_t uplo, const size_t count,
                                                              const size_t ldb, const size_t m,
                                                              cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t TriangleMatrixCopy<double>(const double *input, double *output, bool clean,
                                                                cublasFillMode_t uplo, const size_t count,
                                                                const size_t ldb, const size_t m,
                                                                cudaStream_t cuda_stream);

template <typename T>
cudaError_t MatrixCopy(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  MatrixCopyKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t MatrixCopy<float>(const float *input, float *output, const size_t count,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixCopy<half>(const half *input, half *output, const size_t count,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixCopy<double>(const double *input, double *output, const size_t count,
                                                        cudaStream_t cuda_stream);
