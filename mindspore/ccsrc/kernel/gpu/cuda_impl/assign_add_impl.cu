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

#include "assign_add_impl.cuh"
#include "device/gpu/cuda_common.h"
#include "include/cuda_fp16.h"
template <typename T>
__global__ void AssignAdd(const size_t size, T* ref, const T* value, T* output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    output[pos] = ref[pos] + value[pos];
    ref[pos] = output[pos];
  }
  return;
}

template <typename T>
void CalAssignAdd(const size_t size, T* ref, const T* value, T* output, cudaStream_t cuda_stream) {
  AssignAdd<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, ref, value, output);

  return;
}

template void CalAssignAdd<float>(const size_t size, float* ref, const float* value, float* output,
                                  cudaStream_t cuda_stream);
template void CalAssignAdd<half>(const size_t size, half* ref, const half* value, half* output,
                                 cudaStream_t cuda_stream);
template void CalAssignAdd<int>(const size_t size, int* ref, const int* value, int* output, cudaStream_t cuda_stream);
