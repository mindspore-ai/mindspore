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
#include "backend/kernel_compiler/gpu/cuda_impl/split_impl.cuh"
template <typename T>
__global__ void Split(const size_t size, const int axis_step, const int all_size_before_axis,
                      const int all_size_axis, const T* input, T** outputs) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int num = pos % all_size_before_axis / all_size_axis;
    int block = num / axis_step;
    int block_pos = pos / all_size_before_axis * axis_step * all_size_axis +
                    num % axis_step * all_size_axis + pos % all_size_axis;
    outputs[block][block_pos] = input[pos];
  }
  return;
}

template <typename T>
void SplitKernel(const size_t size, const int axis_step, const int all_size_before_axis,
                 const int all_size_axis, const T* input, T** outputs, cudaStream_t cuda_stream) {
  Split<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, axis_step, all_size_before_axis,
                                                           all_size_axis, input, outputs);
  return;
}

template void SplitKernel(const size_t size, const int axis_step, const int all_size_before_axis,
                          const int all_size_axis, const half* input, half** outputs,
                          cudaStream_t cuda_stream);
template void SplitKernel(const size_t size, const int axis_step, const int all_size_before_axis,
                          const int all_size_axis, const float* input, float** outputs,
                          cudaStream_t cuda_stream);
template void SplitKernel(const size_t size, const int axis_step, const int all_size_before_axis,
                          const int all_size_axis, const double* input, double** outputs,
                          cudaStream_t cuda_stream);
template void SplitKernel(const size_t size, const int axis_step, const int all_size_before_axis,
                          const int all_size_axis, const int* input, int** outputs,
                          cudaStream_t cuda_stream);
template void SplitKernel(const size_t size, const int axis_step, const int all_size_before_axis,
                          const int all_size_axis, const uint32_t* input, uint32_t** outputs,
                          cudaStream_t cuda_stream);
template void SplitKernel(const size_t size, const int axis_step, const int all_size_before_axis,
                          const int all_size_axis, const int64_t* input, int64_t** outputs,
                          cudaStream_t cuda_stream);
template void SplitKernel(const size_t size, const int axis_step, const int all_size_before_axis,
                          const int all_size_axis, const bool* input, bool** outputs,
                          cudaStream_t cuda_stream);
