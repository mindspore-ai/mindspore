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
__global__ void Unpack(const size_t size, const size_t output_num,
                       const size_t dims_after_axis, T** outputs, const T* input) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
      size_t cur_input_index = pos / dims_after_axis % output_num;
      size_t cycle_len = output_num * dims_after_axis;
      size_t local_index = pos / cycle_len * dims_after_axis + pos % cycle_len % dims_after_axis;
      outputs[cur_input_index][local_index] = input[pos];
  }
  return;
}

template <typename T>
void UnpackKernel(const size_t size, const size_t output_num,
                  const size_t dims_after_axis, T** outputs, const T* input,
                  cudaStream_t cuda_stream) {
  Unpack<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, output_num,
                                                            dims_after_axis, outputs, input);
  return;
}

template void UnpackKernel(const size_t size, const size_t output_num,
                           const size_t dims_after_axis, int8_t** outputs, const int8_t* input,
                           cudaStream_t cuda_stream);
template void UnpackKernel(const size_t size, const size_t output_num,
                           const size_t dims_after_axis, int16_t** outputs, const int16_t* input,
                           cudaStream_t cuda_stream);
template void UnpackKernel(const size_t size, const size_t output_num,
                           const size_t dims_after_axis, int** outputs, const int* input,
                           cudaStream_t cuda_stream);
template void UnpackKernel(const size_t size, const size_t output_num,
                           const size_t dims_after_axis, int64_t** outputs, const int64_t* input,
                           cudaStream_t cuda_stream);
template void UnpackKernel(const size_t size, const size_t output_num,
                           const size_t dims_after_axis, uint8_t** outputs, const uint8_t* input,
                           cudaStream_t cuda_stream);
template void UnpackKernel(const size_t size, const size_t output_num,
                           const size_t dims_after_axis, uint16_t** outputs, const uint16_t* input,
                           cudaStream_t cuda_stream);
template void UnpackKernel(const size_t size, const size_t output_num,
                           const size_t dims_after_axis, uint32_t** outputs, const uint32_t* input,
                           cudaStream_t cuda_stream);
template void UnpackKernel(const size_t size, const size_t output_num,
                           const size_t dims_after_axis, uint64_t** outputs, const uint64_t* input,
                           cudaStream_t cuda_stream);
template void UnpackKernel(const size_t size, const size_t output_num,
                           const size_t dims_after_axis, half** outputs, const half* input,
                           cudaStream_t cuda_stream);
template void UnpackKernel(const size_t size, const size_t output_num,
                           const size_t dims_after_axis, float** outputs, const float* input,
                           cudaStream_t cuda_stream);
template void UnpackKernel(const size_t size, const size_t output_num,
                           const size_t dims_after_axis, bool** outputs, const bool* input,
                           cudaStream_t cuda_stream);
