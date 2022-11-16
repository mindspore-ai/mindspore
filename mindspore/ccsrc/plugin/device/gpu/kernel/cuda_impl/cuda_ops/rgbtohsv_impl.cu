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

#include <iostream>
#include "rgbtohsv_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "utils/ms_utils.h"
#include "include/cuda_fp16.h"

template <typename T>
__device__ __forceinline__  T Max(T a, T b) {
     return a > b ? a : b;
}

template <typename T>
__device__ __forceinline__  T Min(T a, T b) {
     return a < b ? a : b;
}

template <typename T>
__device__ __forceinline__  T Mod(T a, T b) {
     return a - b * floor(a/b);
}

template <typename T>
__device__ __forceinline__  void rgb2hsv(const T r, const T g, const T b, T *h, T *s, T *v) {
    const T M = Max(r, Max(g, b));
    const T m = Min(r, Min(g, b));
    const T chroma = M - m;
    *h = 0.0f, *s = 0.0f;
    if (chroma > T(0.0f)) {
      if (M == r) {
        const T num = (g - b) / chroma;
        const T sign = copysignf(1.0f, num);
        *h = ((sign < 0.0f) * 6.0f + sign * Mod(sign * num, T(6.0f))) / 6.0f;
      } else if (M == g) {
        *h = ((b - r) / chroma + 2.0f) / 6.0f;
      } else {
        *h = ((r - g) / chroma + 4.0f) / 6.0f;
      }
    } else {
      *h = 0.0f;
    }
    if (M > 0.0) {
      *s = chroma / M;
    } else {
      *s = 0.0f;
    }
    *v = M;
    return;
}

template <typename T>
__global__ void Rgbtohsv(const size_t input_size, const T *input, T *output) {
  for (size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
       idx < input_size / 3; idx += blockDim.x * gridDim.x) {
        T h, s, v;
        rgb2hsv(input[idx * 3], input[idx * 3 + 1], input[idx * 3 + 2], &h, &s, &v);
        output[idx * 3] = h;
        output[idx * 3 + 1] = s;
        output[idx * 3 + 2] = v;
  }
  return;
}

template <>
__global__ void Rgbtohsv(const size_t input_size, const half *input, half *output) {
  for (size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
       idx < input_size / 3; idx += blockDim.x * gridDim.x) {
        float h, s, v;
        rgb2hsv(static_cast<float>(input[idx * 3]),
                static_cast<float>(input[idx * 3 + 1]),
                static_cast<float>(input[idx * 3 + 2]), &h, &s, &v);
        output[idx * 3] = static_cast<half>(h);
        output[idx * 3 + 1] = static_cast<half>(s);
        output[idx * 3 + 2] = static_cast<half>(v);
  }
  return;
}

template <typename T>
void CalRgbtohsv(const size_t input_size, const T *input,
                 T *output, const uint32_t &device_id, cudaStream_t cuda_stream) {
  Rgbtohsv<<<CUDA_BLOCKS(device_id, input_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(input_size, input, output);
  return;
}

template CUDA_LIB_EXPORT void CalRgbtohsv<half>(const size_t input_size, const half *input, half *output,
                                                      const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalRgbtohsv<float>(const size_t input_size, const float *input, float *output,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalRgbtohsv<double>(const size_t input_size, const double *input, double *output,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
