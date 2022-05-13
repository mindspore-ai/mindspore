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

#include "src/extendrt/delegate/tensorrt/cuda_impl/activation.cuh"
#include <stdio.h>
#include <math.h>
#include "src/extendrt/delegate/tensorrt/cuda_impl/cuda_helper.h"

template <typename T>
__global__ void SigmoidKernel(const T *input1, T *output, int element_cnt) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < element_cnt; pos += blockDim.x * gridDim.x) {
    output[pos] = static_cast<T>(1) / (static_cast<T>(1) + exp(-input1[pos]));
  }
}

template <typename T>
__global__ void GeluKernel(const T *input_addr, T *output_addr, int size) {
  // formula:
  // gelu(x) = 0.5 * x * (1.0 + tanh(y))
  // tanh(y) = 2 / (1 + exp(-2y)) - 1)
  // y = sqrt(2/pi) * (x + 0.044715 * x^3)
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    float x = input_addr[pos];
    float tanh_res = tanh(0.7978845608f * (x + 0.044715f * x * x * x));
    output_addr[pos] = 0.5f * x * (1.0f + tanh_res);
  }
}

template <typename T>
void Sigmoid(const T *input1, T *output, int element_cnt, cudaStream_t stream) {
  SigmoidKernel<<<GET_BLOCKS(element_cnt), GET_THREADS, 0, stream>>>(input1, output, element_cnt);
  return;
}

template <typename T>
void Gelu(const T *input1, T *output, int element_cnt, cudaStream_t stream) {
  GeluKernel<<<GET_BLOCKS(element_cnt), GET_THREADS, 0, stream>>>(input1, output, element_cnt);
  return;
}

template void Sigmoid(const float *input1, float *output, int element_cnt, cudaStream_t stream);

template void Gelu(const float *input1, float *output, int element_cnt, cudaStream_t stream);
