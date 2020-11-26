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

#include "backend/kernel_compiler/gpu/cuda_impl/linspace.cuh"
#include <iostream>

template <typename T>
__global__ void LinSpaceKernel(const T *start, const T *stop, const size_t value_count, T *output) {
  T add_value = ((*stop - *start) / (value_count - 1));
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < value_count; i += gridDim.x * blockDim.x) {
    output[i] = *start + (add_value * i);
  }
}
template <typename T>
void calLinSpace(const T *start, const T *stop, const size_t value_count, T *output, cudaStream_t cuda_stream) {
  LinSpaceKernel<<<GET_BLOCKS(value_count), GET_THREADS, 0, cuda_stream>>>(start, stop, value_count, output);
}
template void calLinSpace<float>(const float *start, const float *stop, const size_t value_count, float *output,
                                 cudaStream_t cuda_stream);
