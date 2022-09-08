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

#include "mvlgamma_impl.cuh"
#ifdef _WIN32
// for M_PI
#define _USE_MATH_DEFINES
#include <math.h>
#endif

template <typename T>
__global__ void Valid(bool *valid, const size_t size, const T *input, const int p) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    if (input[pos] <= (0.5 * (p - 1))) {
      *valid = false;
      return;
    }
  }
  return;
}

template <typename T>
__global__ void Mvlgamma(const size_t size, const T *input, const int p, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    T temp = 0;
    for (int i = 1; i <= p; i++) {
      temp += lgamma(input[pos] - static_cast<T>((i - 1) * 0.5));
    }
    output[pos] = temp + static_cast<T>(p * (p - 1) * 0.25 * log(M_PI));
  }
  return;
}

template <typename T>
void CalMvlgamma(bool *valid, const size_t size, const T *input, const int p, T *output, const uint32_t &device_id,
                 cudaStream_t cuda_stream) {
  Valid<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(valid, size, input, p);
  bool host_valid = true;
  cudaMemcpyFromSymbol(&host_valid, valid, sizeof(bool));
  if (!host_valid) {
    return;
  }
  Mvlgamma<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input, p, output);
  return;
}

template
CUDA_LIB_EXPORT void CalMvlgamma<float>(bool *valid, const size_t size, const float *input, const int p,
                                        float *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalMvlgamma<double>(bool *valid, const size_t size, const double *input, const int p,
                                         double *output, const uint32_t &device_id, cudaStream_t cuda_stream);
