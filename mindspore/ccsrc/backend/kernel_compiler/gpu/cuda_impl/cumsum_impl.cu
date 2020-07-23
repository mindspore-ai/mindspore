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

#include "cumsum_impl.cuh"
#include "runtime/device/gpu/cuda_common.h"

template <typename T>
__global__ void CumSumKernel(T *input, T *output, size_t dim0, size_t dim1, size_t dim2, size_t stride,
                             size_t stride2) {
  size_t num = dim0 * dim2;
  size_t i, k, offset;
  for (size_t write_index = blockIdx.x * blockDim.x + threadIdx.x; write_index < num;
       write_index += blockDim.x * gridDim.x) {
    i = write_index / dim2 % dim0;
    k = write_index % dim2;
    offset = i * stride + k;
    for (size_t j = 0; j < dim1; ++j) {
      size_t read_index = j * stride2 + offset;
      if (j == 0) {
        output[read_index] = input[read_index];
      } else {
        size_t read_index2 = (j - 1) * stride2 + offset;
        output[read_index] = output[read_index2] + input[read_index];
      }
    }
  }
}
template <typename T>
void CumSum(T *input, T *output, size_t dim0, size_t dim1, size_t dim2, size_t stride, size_t stride2,
            cudaStream_t stream) {
  int size = dim0 * dim2;
  CumSumKernel<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(input, output, dim0, dim1, dim2, stride, stride2);
  return;
}

template void CumSum<float>(float *input, float *output, size_t dim0, size_t dim1, size_t dim2, size_t stride,
                            size_t stride2, cudaStream_t stream);
