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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cast_all_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T, typename S>
__global__ void CastAll(T **inputs, S **output, const size_t num, const size_t *size) {
  for (size_t i = 0; i < num; i++) {
    for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size[i]; pos += blockDim.x * gridDim.x) {
      output[i][pos] = static_cast<S>(inputs[i][pos]);
    }
  }
}

template <typename T, typename S>
cudaError_t CastAllKernel(T **inputs, S **output, const size_t max, const size_t num, const size_t *size,
                          cudaStream_t stream) {
  CastAll<<<GET_BLOCKS(max), GET_THREADS, 0, stream>>>(inputs, output, num, size);
  return GetCudaStatus();
}
template CUDA_LIB_EXPORT cudaError_t CastAllKernel(half **inputs, float **output, const size_t max, const size_t num,
                                                   const size_t *size, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t CastAllKernel(float **inputs, half **output, const size_t max, const size_t num,
                                                   const size_t *size, cudaStream_t stream);
