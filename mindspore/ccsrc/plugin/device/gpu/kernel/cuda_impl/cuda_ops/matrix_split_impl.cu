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

#include "matrix_split_impl.cuh"
#include <iostream>
template <typename T>
__global__ void MatrixSplitKernel(const size_t size, const size_t split_dim, const size_t dim, T *input_addr,
                                  T *output_addr) {
  for (size_t pointIdx = blockIdx.x * blockDim.x + threadIdx.x; pointIdx < (size); pointIdx += blockDim.x * gridDim.x) {
    size_t batchIdx = pointIdx / (split_dim * split_dim);
    size_t dst_x = (pointIdx - batchIdx * split_dim * split_dim) / split_dim;
    size_t dst_y = (pointIdx - batchIdx * split_dim * split_dim) % split_dim;
    size_t src_coordinate = (batchIdx * split_dim + dst_x) * dim + batchIdx * split_dim + dst_y;
    output_addr[pointIdx] = input_addr[src_coordinate];
  }
}

template <typename T>
__global__ void MatrixSplitKernel(const size_t size, const size_t split_dim, const size_t dim, const size_t res_dim,
                                  T *input_addr, T *output_addr) {
  for (size_t pointIdx = blockIdx.x * blockDim.x + threadIdx.x; pointIdx < (size); pointIdx += blockDim.x * gridDim.x) {
    size_t batchIdx = pointIdx / (split_dim * split_dim);
    size_t dst_x = (pointIdx - batchIdx * split_dim * split_dim) / split_dim;
    size_t dst_y = (pointIdx - batchIdx * split_dim * split_dim) % split_dim;
    size_t src_coordinate = (batchIdx * split_dim + dst_x) * dim + batchIdx * split_dim + dst_y;
    size_t batch_lower = dim / split_dim;
    if (batchIdx < batch_lower) {
      output_addr[pointIdx] = input_addr[src_coordinate];
    } else {
      if (dst_x < res_dim && dst_y < res_dim) {
        output_addr[pointIdx] = input_addr[src_coordinate];
      } else if (dst_x == dst_y) {
        output_addr[pointIdx] = 1;
      } else {
        output_addr[pointIdx] = 0;
      }
    }
  }
}

template <typename T>
cudaError_t MatrixSplit(const size_t size, const size_t split_dim, const size_t dim, T *input_addr, T *output_addr,
                        cudaStream_t cuda_stream) {
  size_t batch = dim / split_dim;
  size_t res_dim = dim - batch * split_dim;
  if (res_dim == 0) {
    MatrixSplitKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, split_dim, dim, input_addr, output_addr);
  } else {
    MatrixSplitKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, split_dim, dim, res_dim, input_addr,
                                                                         output_addr);
  }
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t MatrixSplit<float>(const size_t size, const size_t split_dim, const size_t dim,
                                                        float *input_addr, float *output_addr,
                                                        cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t MatrixSplit<double>(const size_t size, const size_t split_dim, const size_t dim,
                                                         double *input_addr, double *output_addr,
                                                         cudaStream_t cuda_stream);
