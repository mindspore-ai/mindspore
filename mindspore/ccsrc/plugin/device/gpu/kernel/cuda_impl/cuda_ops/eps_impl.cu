/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#include "eps_impl.cuh"

#include <cmath>
#include <iostream>
#include <limits>

#include "base/float16.h"

template <typename T>
__global__ void EpsKernel(const size_t size, T *input, T *output, T min_val) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    output[pos] = min_val;
  }
}

template <typename T>
void CalEps(const size_t size, T *input, T *output, T min_val, const uint32_t &device_id, cudaStream_t cuda_stream) {
  int thread_num = size > 512 ? 512 : size;
  EpsKernel<<<CUDA_BLOCKS_CAL(device_id, size, thread_num), thread_num, 0, cuda_stream>>>(size, input, output, min_val);
  return;
}

template CUDA_LIB_EXPORT void CalEps<float16>(const size_t size, float16 *input, float16 *output, float16 min_val,
                                              const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalEps<float>(const size_t size, float *input, float *output, float min_val,
                                            const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalEps<double>(const size_t size, double *input, double *output, double min_val,
                                             const uint32_t &device_id, cudaStream_t cuda_stream);
