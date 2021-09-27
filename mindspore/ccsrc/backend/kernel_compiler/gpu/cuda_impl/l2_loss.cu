/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "l2_loss.cuh"
#include "runtime/device/gpu/cuda_common.h"
#include "backend/kernel_compiler/gpu/cuda_impl/util.cuh"

template <typename T>
__global__ void L2LossKernel(const size_t input_size, const T *input , T *output) {
  T ret = 0;
  for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < input_size; id += blockDim.x * gridDim.x) {
      ret = input[id] * input[id];
      ret /= static_cast<T>(2);
      MsAtomicAdd(output, ret);
  }
}

template <typename T>
__global__ void ClearOutputMem(T *output) {
    output[0] = static_cast<T>(0);
}

template <typename T>
void L2Loss(const size_t input_size, const T *input , T *output, cudaStream_t stream) {
  ClearOutputMem<<<GET_BLOCKS(1), GET_THREADS, 0, stream>>>(output);
  L2LossKernel<<<GET_BLOCKS(input_size), GET_THREADS, 0, stream>>>(input_size, input, output);
}

template void L2Loss<float>(const size_t input_size, const float *input , float *output, cudaStream_t stream);
template void L2Loss<half>(const size_t input_size, const half *input , half *output, cudaStream_t stream);
