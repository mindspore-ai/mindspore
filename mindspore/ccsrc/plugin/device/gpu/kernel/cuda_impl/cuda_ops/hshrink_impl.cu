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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/hshrink_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__global__ void HShrinkKernel(size_t size, const T *input, const float lambd, T *output) {
  const T positive_lambd = static_cast<T>(lambd);
  const T negative_lambd = static_cast<T>(-1 * lambd);
  const T zero = static_cast<T>(0);
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = (input[pos] >= negative_lambd && input[pos] <= positive_lambd) ? zero : input[pos];
  }
}

template <typename T>
__global__ void HShrinkGradKernel(size_t size, const T *dout, const T *x, const float lambd, T *output) {
  const T positive_lambd = static_cast<T>(lambd);
  const T negative_lambd = static_cast<T>(-1 * lambd);
  const T zero = static_cast<T>(0);
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = (x[pos] >= negative_lambd && x[pos] <= positive_lambd) ? zero : dout[pos];
  }
}

template <typename T>
cudaError_t CalHShrink(const size_t &size, const T *input, const float lambd, T *output, const uint32_t &device_id,
                       cudaStream_t cuda_stream) {
  HShrinkKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input, lambd, output);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalHShrinkGrad(const size_t &size, const T *dout, const T *x, const float lambd, T *output,
                           const uint32_t &device_id, cudaStream_t cuda_stream) {
  HShrinkGradKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, dout, x, lambd,
                                                                                               output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalHShrink<half>(const size_t &size, const half *input, const float lambd,
                                                      half *output, const uint32_t &device_id,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalHShrink<float>(const size_t &size, const float *input, const float lambd,
                                                       float *output, const uint32_t &device_id,
                                                       cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalHShrinkGrad<half>(const size_t &size, const half *dout, const half *x,
                                                          const float lambd, half *output, const uint32_t &device_id,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalHShrinkGrad<float>(const size_t &size, const float *dout, const float *x,
                                                           const float lambd, float *output, const uint32_t &device_id,
                                                           cudaStream_t cuda_stream);
