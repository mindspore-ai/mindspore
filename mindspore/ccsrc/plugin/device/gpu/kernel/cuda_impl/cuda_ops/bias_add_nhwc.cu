/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include <math.h>
#include <stdint.h>
#include <vector>
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/bias_add_nhwc.cuh"

template <typename T>
__global__ void CalBiasAdd(const size_t num_value, const size_t num_bias, const T *src, const T *bias, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < num_value; pos += blockDim.x * gridDim.x) {
    size_t j = pos % num_bias;
    output[pos] = src[pos] + bias[j];
  }
  return;
}

template <typename T>
cudaError_t CalBiasAddNHWC(const size_t num_value, const size_t num_bias, const T *src, const T *bias, T *output,
                           const uint32_t &device_id, cudaStream_t cuda_stream) {
  size_t thread_num = num_value > 256 ? 256 : num_value;
  CalBiasAdd<<<CUDA_BLOCKS_CAL(device_id, num_value, thread_num), thread_num, 0, cuda_stream>>>(num_value, num_bias,
                                                                                                src, bias, output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalBiasAddNHWC<half>(const size_t num_value, const size_t num_bias,
                                                          const half *src, const half *bias, half *output,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalBiasAddNHWC<float>(const size_t num_value, const size_t num_bias,
                                                           const float *src, const float *bias, float *output,
                                                           const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalBiasAddNHWC<int8_t>(const size_t num_value, const size_t num_bias,
                                                            const int8_t *src, const int8_t *bias, int8_t *output,
                                                            const uint32_t &device_id, cudaStream_t cuda_stream);
