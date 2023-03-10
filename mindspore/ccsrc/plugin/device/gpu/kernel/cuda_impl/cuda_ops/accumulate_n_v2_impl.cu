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
#include <math.h>
#include <stdint.h>
#include <vector>
#include "include/cuda_fp16.h"
#include "accumulate_n_v2_impl.cuh"

template <typename T>
__global__ void AccumulateNV2(const size_t size, const size_t n, T **inputs, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    T temp = 0;
    for (size_t num = 0; num < n; num++) {
      temp += inputs[num][pos];
    }
    output[pos] = temp;
  }
  return;
}

template <typename T>
cudaError_t CalAccumulateNV2(const size_t size, const size_t n, T **inputs, T *output, const uint32_t &device_id,
                             cudaStream_t cuda_stream) {
  AccumulateNV2<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, n, inputs, output);
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template CUDA_LIB_EXPORT cudaError_t CalAccumulateNV2<uint8_t>(const size_t size, const size_t n, uint8_t **inputs,
                                                               uint8_t *output, const uint32_t &device_id,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAccumulateNV2<int8_t>(const size_t size, const size_t n, int8_t **inputs,
                                                              int8_t *output, const uint32_t &device_id,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAccumulateNV2<int32_t>(const size_t size, const size_t n, int32_t **inputs,
                                                               int32_t *output, const uint32_t &device_id,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAccumulateNV2<half>(const size_t size, const size_t n, half **inputs,
                                                            half *output, const uint32_t &device_id,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAccumulateNV2<float>(const size_t size, const size_t n, float **inputs,
                                                             float *output, const uint32_t &device_id,
                                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAccumulateNV2<double>(const size_t size, const size_t n, double **inputs,
                                                              double *output, const uint32_t &device_id,
                                                              cudaStream_t cuda_stream);
