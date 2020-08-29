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

#include <cuda_runtime.h>
#include "range_impl.cuh"
#include "runtime/device/gpu/cuda_common.h"

template <typename T>
__global__ void Range(const int size, const float start, const float limit, const float delta, const T *input,
                      T *output) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = input[pos] * delta + start;
  }
}

template <typename T>
void CalRange(const int size, const float start, const float limit, const float delta, const T *input, T *output,
              cudaStream_t cuda_stream) {
  Range<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, start, limit, delta, input, output);
  return;
}
template void CalRange<float>(const int size, const float start, const float limit, const float delta,
                              const float *input, float *output, cudaStream_t cuda_stream);

template void CalRange<int>(const int size, const float start, const float limit, const float delta, const int *input,
                            int *output, cudaStream_t cuda_stream);
