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

#include "src/delegate/tensorrt/cuda_impl/activation.cuh"
#include <stdio.h>
#include <math.h>
#include "src/delegate/tensorrt/cuda_impl/cuda_helper.h"

template <typename T>
__global__ void SigmoidKernel(const T *input1, T *output, int element_cnt) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < element_cnt; pos += blockDim.x * gridDim.x) {
    output[pos] = static_cast<T>(1) / (static_cast<T>(1) + exp(-input1[pos]));
  }
}

template <typename T>
void Sigmoid(const T *input1, T *output, int element_cnt, cudaStream_t stream) {
  SigmoidKernel<<<GET_BLOCKS(element_cnt), GET_THREADS, 0, stream>>>(input1, output, element_cnt);
  return;
}

template void Sigmoid(const float *input1, float *output, int element_cnt, cudaStream_t stream);
