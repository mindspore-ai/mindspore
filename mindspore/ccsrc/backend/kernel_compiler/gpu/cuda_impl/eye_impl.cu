/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "eye_impl.cuh"
#include <iostream>
template <typename T>
__global__ void EyeKernel(const size_t size, const size_t dim, T *output_addr) {
  for (size_t pointIdx = blockIdx.x * blockDim.x + threadIdx.x; pointIdx < (size); pointIdx += blockDim.x * gridDim.x) {
    size_t batchIdx = pointIdx / (dim * dim);
    size_t dst_x = (pointIdx - batchIdx * dim * dim) / dim;
    size_t dst_y = (pointIdx - batchIdx * dim * dim) % dim;
    if (dst_x == dst_y) {
      output_addr[pointIdx] = 1;
    } else {
      output_addr[pointIdx] = 0;
    }
  }
}

template <typename T>
void Eye(const size_t size, const size_t dim, T *output_addr, cudaStream_t cuda_stream) {
  EyeKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, dim, output_addr);
  return;
}

template void Eye<float>(const size_t size, const size_t dim, float *output_addr, cudaStream_t cuda_stream);
