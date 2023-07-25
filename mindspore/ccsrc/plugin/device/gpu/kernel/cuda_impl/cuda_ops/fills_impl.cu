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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/fills_impl.cuh"
#include "include/cuda_runtime.h"
#include "include/cuda_fp16.h"

template <typename T>
__global__ void FillsKernel(const size_t n, const float *input, T *output) {
  const T value = static_cast<T>(*input);
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < n; pos += blockDim.x * gridDim.x) {
    output[pos] = value;
  }
}

template <typename T>
cudaError_t FillsForward(const size_t &n, const float *input, T *output, const uint32_t &device_id,
                         cudaStream_t stream) {
  FillsKernel<<<CUDA_BLOCKS(device_id, n), CUDA_THREADS(device_id), 0, stream>>>(n, input, output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t FillsForward<float>(const size_t &n, const float *input, float *output,
                                                         const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t FillsForward<half>(const size_t &n, const float *input, half *output,
                                                        const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t FillsForward<int8_t>(const size_t &n, const float *input, int8_t *output,
                                                          const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t FillsForward<int16_t>(const size_t &n, const float *input, int16_t *output,
                                                           const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t FillsForward<int32_t>(const size_t &n, const float *input, int32_t *output,
                                                           const uint32_t &device_id, cudaStream_t stream);
