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
#include "backend/kernel_compiler/gpu/cuda_impl/pack.cuh"
template <typename T>
__global__ void Pack(const int size, const int input_num, const int dims_behind_axis, T** inputs, T* output) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
      int cycle = pos / (input_num * dims_behind_axis);
      int cur_input_index = pos % (input_num * dims_behind_axis) / dims_behind_axis;
      int local_index = pos % (input_num * dims_behind_axis) % dims_behind_axis;
      output[pos] = inputs[cur_input_index][cycle * dims_behind_axis + local_index];
  }
  return;
}

template <typename T>
void PackKernel(const int size, const int input_num,
                const int dims_behind_axis, T** inputs, T* output,
                cudaStream_t cuda_stream) {
  Pack<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input_num, dims_behind_axis, inputs, output);
  return;
}

template void PackKernel(const int size, const int input_num,
                           const int dims_behind_axis, float** inputs, float* output,
                           cudaStream_t cuda_stream);
template void PackKernel(const int size, const int input_num,
                           const int dims_behind_axis, int** inputs, int* output,
                           cudaStream_t cuda_stream);
template void PackKernel(const int size, const int input_num,
                           const int dims_behind_axis, half** inputs, half* output,
                           cudaStream_t cuda_stream);
template void PackKernel(const int size, const int input_num,
                           const int dims_behind_axis, short** inputs, short* output,  // NOLINT
                           cudaStream_t cuda_stream);
template void PackKernel(const int size, const int input_num,
                           const int dims_behind_axis, unsigned char** inputs, unsigned char* output,
                           cudaStream_t cuda_stream);
template void PackKernel(const int size, const int input_num,
                           const int dims_behind_axis, bool** inputs, bool* output,
                           cudaStream_t cuda_stream);
