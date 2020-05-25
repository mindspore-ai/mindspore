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

#include <stdint.h>
#include "dropout_impl.cuh"
#include "include/cuda_runtime.h"

__global__ void DropoutForwardKernel(const float *input, float *mask, float *output, size_t num_count,
                                     float drop_prob) {
  float scale = 1.f / drop_prob;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_count; i += blockDim.x * gridDim.x) {
    mask[i] = mask[i] > drop_prob;
    output[i] = scale * input[i] * mask[i];
  }
}

void DropoutForward(const float *input, float *mask, float *output, size_t num_count, float drop_prob,
                    cudaStream_t cuda_stream) {
  DropoutForwardKernel<<<GET_BLOCKS(num_count), GET_THREADS, 0, cuda_stream>>>(input, mask, output, num_count,
                                                                               drop_prob);
}

__global__ void DropoutBackwardKernel(const float *dy, const float *mask, float *dx, size_t num_count,
                                      float drop_prob) {
  float scale = 1.f / (1.f - drop_prob);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_count; i += blockDim.x * gridDim.x) {
    dx[i] = scale * dy[i] * mask[i];
  }
}

void DropoutBackward(const float *dy, const float *mask, float *dx, size_t num_count, float drop_prob,
                     cudaStream_t cuda_stream) {
  DropoutBackwardKernel<<<GET_BLOCKS(num_count), GET_THREADS, 0, cuda_stream>>>(dy, mask, dx, num_count, drop_prob);
}
