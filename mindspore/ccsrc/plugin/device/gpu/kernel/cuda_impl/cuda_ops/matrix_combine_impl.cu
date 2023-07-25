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

#include "matrix_combine_impl.cuh"
#include <iostream>
template <typename T>
__global__ void MatrixCombineKernel(const size_t size, const size_t src_height, const size_t src_width,
                                    const size_t dst_width, T *input_addr, T *output_addr) {
  for (size_t pointIdx = blockIdx.x * blockDim.x + threadIdx.x; pointIdx < (size); pointIdx += blockDim.x * gridDim.x) {
    size_t batchIdx = pointIdx / (src_height * src_width);
    size_t src_h = (pointIdx - batchIdx * src_height * src_width) / src_width;
    size_t src_w = (pointIdx - batchIdx * src_height * src_width) % src_width;
    size_t dst_h = src_height * batchIdx + src_h;
    size_t dst_w = src_width * batchIdx + src_w;
    output_addr[dst_h * dst_width + dst_w] = input_addr[pointIdx];
  }
}

template <typename T>
__global__ void MatrixCombineKernel(const size_t size, const size_t src_height, const size_t src_width,
                                    const size_t dst_width, const size_t res_width, const size_t batch, T *input_addr,
                                    T *output_addr) {
  for (size_t pointIdx = blockIdx.x * blockDim.x + threadIdx.x; pointIdx < (size); pointIdx += blockDim.x * gridDim.x) {
    size_t batchIdx = pointIdx / (src_height * src_width);
    if (batchIdx != (batch - 1)) {
      size_t src_h = (pointIdx - batchIdx * src_height * src_width) / src_width;
      size_t src_w = (pointIdx - batchIdx * src_height * src_width) % src_width;
      size_t dst_h = src_height * batchIdx + src_h;
      size_t dst_w = src_width * batchIdx + src_w;
      output_addr[dst_h * dst_width + dst_w] = input_addr[pointIdx];
    } else {
      size_t src_h = (pointIdx - (batch - 1) * src_height * src_width) / res_width;
      size_t src_w = (pointIdx - (batch - 1) * src_height * src_width) % res_width;
      size_t src_coordinate = (batch - 1) * src_height * src_width + src_h * src_width + src_w;
      size_t dst_h = src_height * (batch - 1) + src_h;
      size_t dst_w = src_width * (batch - 1) + src_w;
      output_addr[dst_h * dst_width + dst_w] = input_addr[src_coordinate];
    }
  }
}

template <typename T>
cudaError_t MatrixCombine(const size_t size, const size_t src_height, const size_t src_width, const size_t dst_width,
                          const size_t residual, const size_t res_width, const size_t batch, T *input_addr,
                          T *output_addr, cudaStream_t cuda_stream) {
  if (residual == 0) {
    MatrixCombineKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, src_height, src_width, dst_width,
                                                                           input_addr, output_addr);
  } else {
    MatrixCombineKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, src_height, src_width, dst_width,
                                                                           res_width, batch, input_addr, output_addr);
  }
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t MatrixCombine<float>(const size_t size, const size_t src_height,
                                                          const size_t src_width, const size_t dst_width,
                                                          const size_t residual, const size_t res_width,
                                                          const size_t batch, float *input_addr, float *output_addr,
                                                          cudaStream_t cuda_stream);
