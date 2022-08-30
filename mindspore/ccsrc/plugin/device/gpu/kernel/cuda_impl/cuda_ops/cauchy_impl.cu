/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <curand_kernel.h>
#include "include/cuda_fp16.h"
#include "cauchy_impl.cuh"
__global__ void CauchyKernel(float *output, uint64_t seed, const float median, const float sigma, const size_t count) {
  curandStatePhilox4_32_10_t state;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x * 4) {
    curand_init(seed, 0, i, &state);
    auto rand = curand_uniform4(&state);
    float randu[4];
    randu[0] = static_cast<float>(rand.x);
    randu[1] = static_cast<float>(rand.y);
    randu[2] = static_cast<float>(rand.z);
    randu[3] = static_cast<float>(rand.w);
#pragma unroll
    for (int j = 0; j < 4; ++j, i += blockDim.x * gridDim.x) {
      output[i] = median + sigma * tanf(static_cast<float>(M_PI) * (randu[j] - static_cast<float>(0.5)));
    }
  }
}

void Cauchy(float *output, uint64_t seed, const float median, const float sigma, const size_t count, uint32_t device_id,
            cudaStream_t cuda_stream) {
  CauchyKernel<<<CUDA_BLOCKS(device_id, count), CUDA_THREADS(device_id), 0, cuda_stream>>>(output, seed, median, sigma,
                                                                                           count);
  return;
}
