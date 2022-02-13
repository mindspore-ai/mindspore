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
template <typename T>
__global__ void DropoutForwardKernel(const T *input, T *mask, T *output, float *mask_f, size_t num_count,
                                     float keep_prob) {
  float scale = 1.f / keep_prob;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_count; i += blockDim.x * gridDim.x) {
    mask_f[i] = mask_f[i] <= keep_prob;
    output[i] = scale * input[i] * mask_f[i];
    mask[i] = mask_f[i];
  }
}
template <>
__global__ void DropoutForwardKernel(const half *input, half *mask, half *output, float *mask_f,
                                     size_t num_count, float keep_prob) {
  half scale = __float2half(1.f / keep_prob);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_count; i += blockDim.x * gridDim.x) {
    mask_f[i] = mask_f[i] <= keep_prob;
    output[i] = scale * input[i] * __float2half(mask_f[i]);
    mask[i] = __float2half(mask_f[i]);
  }
}
template <typename T>
void DropoutForward(const T *input, T *mask, T *output, float *mask_f, size_t num_count, float drop_prob,
                    cudaStream_t cuda_stream) {
  DropoutForwardKernel<<<GET_BLOCKS(num_count), GET_THREADS, 0, cuda_stream>>>(input, mask, output, mask_f,
                                                                               num_count, drop_prob);
}
template <typename T>
__global__ void DropoutBackwardKernel(const T *dy, const T *mask, T *dx, size_t num_count,
                                      float keep_prob) {
  float scale = 1.f / keep_prob;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_count; i += blockDim.x * gridDim.x) {
    dx[i] = scale * dy[i] * mask[i];
  }
}
template <>
__global__ void DropoutBackwardKernel(const half *dy, const half *mask, half *dx, size_t num_count,
                                      float keep_prob) {
  half scale = __float2half(1.f / keep_prob);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_count; i += blockDim.x * gridDim.x) {
    dx[i] = scale * dy[i] * mask[i];
  }
}
template <typename T>
void DropoutBackward(const T *dy, const T *mask, T *dx, size_t num_count, float drop_prob,
                     cudaStream_t cuda_stream) {
  DropoutBackwardKernel<<<GET_BLOCKS(num_count), GET_THREADS, 0, cuda_stream>>>(dy, mask, dx, num_count, drop_prob);
}

template void DropoutForward<float>(const float *input, float *mask, float *output, float *mask_f,
                                    size_t num_count, float drop_prob, cudaStream_t cuda_stream);
template void DropoutForward<half>(const half *input, half *mask, half *output, float *mask_f,
                                    size_t num_count, float drop_prob, cudaStream_t cuda_stream);
template void DropoutBackward<float>(const float *dy, const float *mask, float *dx, size_t num_count,
                                     float drop_prob, cudaStream_t cuda_stream);
template void DropoutBackward<half>(const half *dy, const half *mask, half *dx, size_t num_count,
                                    float drop_prob, cudaStream_t cuda_stream);
