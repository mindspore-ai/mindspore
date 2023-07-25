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
#include <math.h>
#include <limits>
#include "include/cuda_runtime.h"
#include "include/cuda_fp16.h"

template <typename T>
inline __device__ T polevl(const T x, const T E[], size_t len) {
  T result = 0;
  for (size_t i = 0; i <= len; i++) {
    result = result * x + E[i];
  }
  return result;
}

inline __device__ double calc_digamma(double x) {
  double PSI_10 = 2.25175258906672110764;
  double PI = 3.14159265358979323846;
  if (x == 0) {
    int S_x = (*reinterpret_cast<int *>(&x));
    return (S_x >> 31) == 0 ? -INFINITY : INFINITY;
  }
  if (x < 0) {
    if (x == truncf(x)) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    double q, r;
    r = modf(x, &q);
    double tan_over_r = PI / tan(PI * r);
    return calc_digamma(1 - x) - tan_over_r;
  } else {
    const double E[] = {8.33333333333333333333E-2,  -2.10927960927960927961E-2, 7.57575757575757575758E-3,
                        -4.16666666666666666667E-3, 3.96825396825396825397E-3,  -8.33333333333333333333E-3,
                        8.33333333333333333333E-2};
    double out = 0;
    while (x < 10) {
      out -= 1 / x;
      x += 1;
    }
    if (x == 10) {
      return out + PSI_10;
    }
    double y = 0;
    if (x < 1.0e17f) {
      double z = 1 / (x * x);
      y = z * polevl(z, E, 6);
    }
    return out + logf(x) - (0.5f / x) - y;
  }
}

__global__ void CalDigammaKernel(size_t size, const double *input, double *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = calc_digamma(input[pos]);
  }
  return;
}

__global__ void CalDigammaKernel(size_t size, const float *input, float *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = static_cast<float>(static_cast<double>(calc_digamma(input[pos])));
  }
  return;
}

__global__ void CalDigammaKernel(size_t size, const half *input, half *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = half(calc_digamma(static_cast<double>(input[pos])));
  }
  return;
}

template <typename T>
cudaError_t CalDigamma(size_t size, const T *input, T *output, const uint32_t &device_id, cudaStream_t cuda_stream) {
  CalDigammaKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input, output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalDigamma<float>(size_t size, const float *input, float *output,
                                                       const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalDigamma<double>(size_t size, const double *input, double *output,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalDigamma<half>(size_t size, const half *input, half *output,
                                                      const uint32_t &device_id, cudaStream_t cuda_stream);
