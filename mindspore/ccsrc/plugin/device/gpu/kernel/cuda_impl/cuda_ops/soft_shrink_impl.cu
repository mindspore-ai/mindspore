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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/soft_shrink_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__global__ void SoftShrinkComp(size_t size, const T *input, const float lambd, T *output) {
  const T positive_lambd = static_cast<T>(lambd);
  const T negative_lambd = static_cast<T>(-1 * lambd);
  const T zero = static_cast<T>(0);
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = (input[pos] > positive_lambd)
                    ? (input[pos] - positive_lambd)
                    : ((input[pos] < negative_lambd) ? (input[pos] + positive_lambd) : (zero));
  }
}

template <typename T>
__global__ void SoftShrinkGradComp(size_t size, const T *dy_addr, const T *x_addr, const float lambd, T *dx_addr) {
  const T positive_lambd = static_cast<T>(lambd);
  const T negative_lambd = static_cast<T>(-1 * lambd);
  const T zero = static_cast<T>(0);
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    dx_addr[pos] = (x_addr[pos] >= negative_lambd && x_addr[pos] <= positive_lambd) ? zero : dy_addr[pos];
  }
}

template <typename T>
cudaError_t SoftShrink(const size_t &size, const T *input, const float lambd, T *output, const uint32_t &device_id,
                       cudaStream_t cuda_stream) {
  SoftShrinkComp<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input, lambd, output);
  return GetCudaStatus();
}

template <typename T>
cudaError_t SoftShrinkGrad(const size_t &size, const T *dy_addr, const T *x_addr, const float lambd, T *dx_addr,
                           const uint32_t &device_id, cudaStream_t cuda_stream) {
  SoftShrinkGradComp<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, dy_addr, x_addr,
                                                                                                lambd, dx_addr);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t SoftShrink(const size_t &size, const half *input, const float lambd, half *output,
                                                const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t SoftShrink(const size_t &size, const float *input, const float lambd,
                                                float *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t SoftShrink(const size_t &size, const int *input, const float lambd, int *output,
                                                const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t SoftShrink(const size_t &size, const int64_t *input, const float lambd,
                                                int64_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t SoftShrinkGrad(const size_t &size, const half *dy_addr, const half *x_addr,
                                                    const float lambd, half *dx_addr, const uint32_t &device_id,
                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t SoftShrinkGrad(const size_t &size, const float *dy_addr, const float *x_addr,
                                                    const float lambd, float *dx_addr, const uint32_t &device_id,
                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t SoftShrinkGrad(const size_t &size, const int *dy_addr, const int *x_addr,
                                                    const float lambd, int *dx_addr, const uint32_t &device_id,
                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t SoftShrinkGrad(const size_t &size, const int64_t *dy_addr, const int64_t *x_addr,
                                                    const float lambd, int64_t *dx_addr, const uint32_t &device_id,
                                                    cudaStream_t cuda_stream);
