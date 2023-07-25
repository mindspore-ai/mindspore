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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/nextafter_impl.cuh"
#include <cmath>

template <typename T>
__global__ void NextAfterKernel(const size_t size, const T *input1, const T *input2, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = nextafter(input1[pos], input2[pos]);
  }
  return;
}

template <typename T>
cudaError_t NextAfter(const size_t size, const T *input1, const T *input2, T *output, const uint32_t &device_id,
                      cudaStream_t cuda_stream) {
  NextAfterKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input1, input2,
                                                                                             output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t NextAfter<float>(const size_t size, const float *input1, const float *input2,
                                                      float *output, const uint32_t &device_id,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t NextAfter<double>(const size_t size, const double *input1, const double *input2,
                                                       double *output, const uint32_t &device_id,
                                                       cudaStream_t cuda_stream);
