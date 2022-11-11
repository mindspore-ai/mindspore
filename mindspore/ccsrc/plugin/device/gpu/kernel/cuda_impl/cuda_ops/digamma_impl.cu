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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/digamma_impl.cuh"
#include <limits>
#include "include/cuda_runtime.h"
#include "include/cuda_fp16.h"

inline __device__ float polevl(const float x, const float E[], size_t len) {
  float result = 0;
  for (size_t i = 0; i <= len; i++) {
    result = result * x + E[i];
  }
  return result;
}

inline __device__ float calc_digamma(float x) {
  float PSI_10 = 2.25175258906672110764f;
  float PI = 3.14159265358979323846f;
  if (x == 0) {
    int S_x = (*reinterpret_cast<int *>(&x));
    return (S_x >> 31) == 0 ? -INFINITY : INFINITY;
  }
  if (x < 0) {
    if (x == truncf(x)) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    float q, r;
    r = std::modf(x, &q);
    float tan_over_r = PI / tan(PI * r);
    return calc_digamma(1 - x) - tan_over_r;
  } else {
    const float E[] = {8.33333333333333333333E-2f,  -2.10927960927960927961E-2f, 7.57575757575757575758E-3f,
                       -4.16666666666666666667E-3f, 3.96825396825396825397E-3f,  -8.33333333333333333333E-3f,
                       8.33333333333333333333E-2f};
    float out = 0;
    while (x < 10) {
      out -= 1 / x;
      x += 1;
    }
    if (x == 10) {
      return out + PSI_10;
    }
    float y = 0;
    if (x < 1.0e17f) {
      float z = 1 / (x * x);
      y = z * polevl(z, E, 6);
    }
    return out + logf(x) - (0.5f / x) - y;
  }
}

__global__ void CalDigammaKernel(size_t size, const double *input, double *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = static_cast<double>(calc_digamma(static_cast<float>(input[pos])));
  }
  return;
}

__global__ void CalDigammaKernel(size_t size, const float *input, float *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = calc_digamma(input[pos]);
  }
  return;
}

__global__ void CalDigammaKernel(size_t size, const half *input, half *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = half(calc_digamma(static_cast<float>(input[pos])));
  }
  return;
}

template <typename T>
void CalDigamma(size_t size, const T *input, T *output, const uint32_t &device_id, cudaStream_t cuda_stream) {
  CalDigammaKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input, output);
}

template CUDA_LIB_EXPORT void CalDigamma<float>(size_t size, const float *input, float *output,
                                                const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalDigamma<double>(size_t size, const double *input, double *output,
                                                 const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalDigamma<half>(size_t size, const half *input, half *output, const uint32_t &device_id,
                                               cudaStream_t cuda_stream);
