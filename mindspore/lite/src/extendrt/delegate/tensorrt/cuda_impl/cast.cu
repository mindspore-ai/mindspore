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

#include "src/extendrt/delegate/tensorrt/cuda_impl/cast.cuh"
#include "src/extendrt/delegate/tensorrt/cuda_impl/cuda_helper.h"

// Generic cast
template <typename S, typename T>
__device__ __forceinline__ void CastBase(const S *input_addr, T *output_addr) {
  *output_addr = static_cast<T>((*input_addr));
}

template <typename S, typename T>
__global__ void CastKernel(const int input_size, const S *input_addr, T *output_addr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < input_size; pos += blockDim.x * gridDim.x) {
    CastBase(input_addr + pos, output_addr + pos);
  }
}

template <typename S, typename T>
void Cast(const int input_size, const S *input_addr, T *output_addr, cudaStream_t stream) {
  CastKernel<<<GET_BLOCKS(input_size), GET_THREADS, 0, stream>>>(input_size, input_addr, output_addr);
}

template void Cast(const int input_size, const int8_t *input_addr, int8_t *output_addr, cudaStream_t stream);
template void Cast(const int input_size, const int8_t *input_addr, int32_t *output_addr, cudaStream_t stream);
template void Cast(const int input_size, const int8_t *input_addr, float *output_addr, cudaStream_t stream);

template void Cast(const int input_size, const int32_t *input_addr, int8_t *output_addr, cudaStream_t stream);
template void Cast(const int input_size, const int32_t *input_addr, int32_t *output_addr, cudaStream_t stream);
template void Cast(const int input_size, const int32_t *input_addr, float *output_addr, cudaStream_t stream);
template void Cast(const int input_size, const int32_t *input_addr, bool *output_addr, cudaStream_t stream);

template void Cast(const int input_size, const float *input_addr, int8_t *output_addr, cudaStream_t stream);
template void Cast(const int input_size, const float *input_addr, int32_t *output_addr, cudaStream_t stream);
template void Cast(const int input_size, const float *input_addr, float *output_addr, cudaStream_t stream);
