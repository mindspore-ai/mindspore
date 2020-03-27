/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include <iostream>
#include "kernel/gpu/cuda_impl/gather.cuh"
#include "device/gpu/cuda_common.h"
template <typename T, typename S>
__global__ void GatherKernel(T *input, S *indices, T *output, size_t output_dim0, size_t output_dim1,
                             size_t output_dim2, size_t input_dim1) {
  int num = output_dim0 * output_dim1 * output_dim2;
  int i, j, k;
  for (int write_index = blockIdx.x * blockDim.x + threadIdx.x; write_index < num;
       write_index += blockDim.x * gridDim.x) {
    i = write_index / (output_dim1 * output_dim2) % output_dim0;
    j = write_index / output_dim2 % output_dim1;
    k = write_index % output_dim2;

    if ((indices[j] >= 0) && (indices[j] < input_dim1)) {
      int read_index = i * input_dim1 * output_dim2 + indices[j] * output_dim2 + k;
      output[write_index] = input[read_index];
    } else {
      output[write_index] = 0;
    }
  }

  return;
}
template <typename T, typename S>
void Gather(T *input, S *indices, T *output, size_t output_dim0, size_t output_dim1, size_t output_dim2,
            size_t input_dim1, cudaStream_t stream) {
  int size = output_dim0 * output_dim1 * output_dim2;
  GatherKernel<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(input, indices, output, output_dim0, output_dim1,
                                                             output_dim2, input_dim1);
  return;
}

template void Gather<float, int>(float *input, int *indices, float *output, size_t output_dim0, size_t output_dim1,
                                 size_t output_dim2, size_t input_dim1, cudaStream_t stream);

template void Gather<half, int>(half *input, int *indices, half *output, size_t output_dim0, size_t output_dim1,
                                size_t output_dim2, size_t input_dim1, cudaStream_t stream);
