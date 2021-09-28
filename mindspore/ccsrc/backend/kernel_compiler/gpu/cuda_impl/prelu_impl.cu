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

#include "backend/kernel_compiler/gpu/cuda_impl/prelu_impl.cuh"

template <typename T>
__global__ void CalPReLUKernel(size_t size, size_t weight_size, size_t per_channel_size,
                               const T *input, const T *weight, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    size_t channel_id = weight_size == 1 ? 0 : (pos / per_channel_size) % weight_size;
    output[pos] = input[pos] < static_cast<T>(0) ? weight[channel_id] * input[pos] :input[pos];
  }
}

template <typename T>
void CalPReLU(size_t size, size_t weight_size, size_t per_channel_size,
              const T *input, const T *weight, T *output, cudaStream_t cuda_stream) {
  CalPReLUKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, weight_size, per_channel_size,
                                                                    input, weight, output);
}

template void CalPReLU(size_t, size_t, size_t, const float *, const float *, float *, cudaStream_t);
template void CalPReLU(size_t, size_t, size_t, const half *, const half *, half *, cudaStream_t);
