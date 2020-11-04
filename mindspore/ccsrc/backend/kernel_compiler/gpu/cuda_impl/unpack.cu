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

#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "backend/kernel_compiler/gpu/cuda_impl/unpack.cuh"
template <typename T>
__global__ void Unpack(const int size, const int output_num,
                       const int dims_after_axis, T** outputs, const T* input) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    int cycle = pos / (output_num * dims_after_axis);
    int cur_output_index = pos % (output_num * dims_after_axis) / dims_after_axis;
    int local_index = pos % (output_num * dims_after_axis) % dims_after_axis;
    outputs[cur_output_index][cycle * dims_after_axis + local_index] = input[pos];
  }
  return;
}

template <typename T>
void UnpackKernel(const int size, const int output_num,
                  const int dims_after_axis, T** outputs, const T* input,
                  cudaStream_t cuda_stream) {
  Unpack<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, output_num,
                                                            dims_after_axis, outputs, input);
  return;
}

template void UnpackKernel(const int size, const int output_num,
                           const int dims_after_axis, float** outputs, const float* input,
                           cudaStream_t cuda_stream);
template void UnpackKernel(const int size, const int output_num,
                           const int dims_after_axis, half** outputs, const half* input,
                           cudaStream_t cuda_stream);
template void UnpackKernel(const int size, const int output_num,
                           const int dims_after_axis, int** outputs, const int* input,
                           cudaStream_t cuda_stream);
template void UnpackKernel(const int size, const int output_num,
                           const int dims_after_axis, int16_t** outputs, const int16_t* input,
                           cudaStream_t cuda_stream);
template void UnpackKernel(const int size, const int output_num,
                           const int dims_after_axis, unsigned char** outputs, const unsigned char* input,
                           cudaStream_t cuda_stream);
template void UnpackKernel(const int size, const int output_num,
                           const int dims_after_axis, bool** outputs, const bool* input,
                           cudaStream_t cuda_stream);
