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

#include "determinant_triangle_impl.cuh"
#include "include/cuda_fp16.h"
template <typename T>
__global__ void DetTriangleKernel(T *input, T *output, size_t matrix_n_, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = 1;
    for (int pos = 0; pos < matrix_n_ * matrix_n_; pos += matrix_n_ + 1) {
      output[i] *= input[i * matrix_n_ * matrix_n_ + pos];
    }
  }
  return;
}

template <typename T>
cudaError_t DetTriangle(T *input, T *output, size_t matrix_n_, size_t count, cudaStream_t cuda_stream) {
  DetTriangleKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, matrix_n_, count);
  return GetCudaStatus();
}

__device__ bool dev_error_res = false;

template <typename T>
__global__ void CheckTriangleKernel(T *input, int fill_mode_, size_t matrix_n_, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    size_t idx = 0;
    if (fill_mode_ == 0) {  // UPPER half
      for (size_t row = 0; row < matrix_n_; row++) {
        for (size_t col = row + 1; col < matrix_n_; col++) {
          idx = i * matrix_n_ * matrix_n_ + row * matrix_n_ + col;
          if (static_cast<float>(input[idx]) != 0) {
            dev_error_res = false;
            return;
          }
        }
      }
    } else if (fill_mode_ == 1) {  // LOWER half
      for (size_t row = 0; row < matrix_n_; row++) {
        for (size_t col = 0; col < row; col++) {
          idx = i * matrix_n_ * matrix_n_ + row * matrix_n_ + col;
          if (static_cast<float>(input[idx]) != 0) {
            dev_error_res = false;
            return;
          }
        }
      }
    } else {
      dev_error_res = false;
      return;
    }
  }
  dev_error_res = true;
  return;
}

template <typename T>
cudaError_t CheckTriangle(T *input, int fill_mode_, size_t matrix_n_, size_t count, cudaStream_t cuda_stream,
                          bool *host_error_res) {
  CheckTriangleKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, fill_mode_, matrix_n_, count);
  cudaMemcpyFromSymbol(host_error_res, dev_error_res, sizeof(bool));
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t DetTriangle<float>(float *input, float *output, size_t matrix_n_, size_t count,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t DetTriangle<half>(half *input, half *output, size_t matrix_n_, size_t count,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CheckTriangle<float>(float *input, int fill_mode_, size_t matrix_n_, size_t count,
                                                          cudaStream_t cuda_stream, bool *host_error_res);
template CUDA_LIB_EXPORT cudaError_t CheckTriangle<half>(half *input, int fill_mode_, size_t matrix_n_, size_t count,
                                                         cudaStream_t cuda_stream, bool *host_error_res);
