/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/extendrt/delegate/tensorrt/cuda_impl/equal.cuh"
#include <stdio.h>
#include "src/extendrt/delegate/tensorrt/cuda_impl/cuda_helper.h"

template <typename T>
__global__ void EqualKernel(const T *input1, const T *input2, T *output, int element_cnt) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < element_cnt; pos += blockDim.x * gridDim.x) {
    output[pos] = (input1[pos] - input2[pos] < 1e-6 && input1[pos] - input2[pos] > -1e-6);
  }
}

template <typename T>
void Equal(const T *input1, const T *input2, T *output, int element_cnt, cudaStream_t stream) {
  EqualKernel<<<GET_BLOCKS(element_cnt), GET_THREADS, 0, stream>>>(input1, input2, output, element_cnt);
  return;
}

template void Equal(const float *input1, const float *input2, float *output, int element_cnt, cudaStream_t stream);
template void Equal(const int *input1, const int *input2, int *output, int element_cnt, cudaStream_t stream);
