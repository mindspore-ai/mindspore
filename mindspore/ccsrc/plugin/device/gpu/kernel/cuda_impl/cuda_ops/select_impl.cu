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

#include <stdio.h>
#include <stdint.h>
#include <include/cuda_runtime.h>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/select_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__global__ void Select(const size_t size, const bool *cond, const T *input_x, const T *input_y, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    output[pos] = cond[pos] ? input_x[pos] : input_y[pos];
  }
  return;
}
__global__ void Select(const size_t size, const int *cond, const float *input_x, const float *input_y, float *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    output[pos] = (cond[pos] - 1 < 1e-6 && cond[pos] - 1 > -1e-6) ? input_x[pos] : input_y[pos];
  }
  return;
}
__global__ void Select(const size_t size, const int *cond, const int *input_x, const int *input_y, int *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    output[pos] = (cond[pos] == 1) ? input_x[pos] : input_y[pos];
  }
  return;
}

template <typename T>
void CalSelect(const size_t size, const bool *cond, const T *input_x, const T *input_y, T *output,
               const uint32_t &device_id, cudaStream_t cuda_stream) {
  Select<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, cond, input_x, input_y,
                                                                                    output);
  return;
}
void CalSelect(const size_t size, const int *cond, const float *input_x, const float *input_y, float *output,
               const uint32_t &device_id, cudaStream_t cuda_stream) {
  Select<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, cond, input_x, input_y,
                                                                                    output);
  return;
}

void CalSelect(const size_t size, const int *cond, const int *input_x, const int *input_y, int *output,
               const uint32_t &device_id, cudaStream_t cuda_stream) {
  Select<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, cond, input_x, input_y,
                                                                                    output);
  return;
}

template CUDA_LIB_EXPORT void CalSelect<double>(const size_t size, const bool *cond, const double *input_X,
                                                const double *input_y, double *output, const uint32_t &device_id,
                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSelect<float>(const size_t size, const bool *cond, const float *input_X,
                                               const float *input_y, float *output, const uint32_t &device_id,
                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSelect<int>(const size_t size, const bool *cond, const int *input_X,
                                             const int *input_y, int *output, const uint32_t &device_id,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSelect<half>(const size_t size, const bool *cond, const half *input_X,
                                              const half *input_y, half *output, const uint32_t &device_id,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSelect<int64_t>(const size_t size, const bool *cond, const int64_t *input_X,
                                                 const int64_t *input_y, int64_t *output, const uint32_t &device_id,
                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSelect<bool>(const size_t size, const bool *cond, const bool *input_X,
                                              const bool *input_y, bool *output, const uint32_t &device_id,
                                              cudaStream_t cuda_stream);
CUDA_LIB_EXPORT void CalSelect(const size_t size, const int *cond, const float *input_x, const float *input_y,
                               float *output, const uint32_t &device_id, cudaStream_t stream);
CUDA_LIB_EXPORT void CalSelect(const size_t size, const int *cond, const float *input_x, const float *input_y,
                               float *output, const uint32_t &device_id, cudaStream_t stream);
