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
#include "check_numerics_impl.cuh"

template <typename T>
__global__ void CheckNumerics(const size_t size, const T *input, int32_t *flag_device) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    if (std::isnan(input[pos])) {
       flag_device[0] = 1;
    } else if (std::isinf(input[pos])) {
       flag_device[1] = 1;
    }
  }
  return;
}

template <>
__global__ void CheckNumerics(const size_t size, const half *input, int32_t *flag_device) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    if (std::isnan(__half2float(input[pos]))) {
       flag_device[0] = 1;
    } else if (std::isinf(__half2float(input[pos]))) {
       flag_device[1] = 1;
    }
  }
  return;
}

template <typename T>
void CalCheckNumerics(const size_t size, const T *input, int32_t *flag_device, const uint32_t &device_id,
                      cudaStream_t cuda_stream) {
  CheckNumerics<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input, flag_device);
  return;
}

template
CUDA_LIB_EXPORT void CalCheckNumerics<half>(const size_t size, const half *input, int32_t *flag_device,
                                            const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalCheckNumerics<float>(const size_t size, const float *input, int32_t *flag_device,
                                             const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalCheckNumerics<double>(const size_t size, const double *input, int32_t *flag_device,
                                              const uint32_t &device_id, cudaStream_t cuda_stream);
