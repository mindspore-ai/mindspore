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
#include "hsvtorgb_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "utils/ms_utils.h"
#include "include/cuda_fp16.h"

template <typename T>
__device__ __forceinline__  void hsv2rgb(const T h, const T s, const T v, T *r = 0, T *g = 0, T *b = 0) {
    const T h60 = h * T(6.0);
    const T h60f = T(floor(static_cast<float>(h60)));
    const int hi = static_cast<int>(h60f) % 6;
    const T f = h60 - h60f;
    const T p = v * (T(1) - s);
    const T q = v * (T(1) - f * s);
    const T t = v * (T(1) - (T(1) - f) * s);
    switch (hi) {
      case 0:
        *r = v; *g = t; *b = p;
        break;
      case 1:
        *r = q; *g = v; *b = p;
        break;
      case 2:
        *r = p; *g = v; *b = t;
        break;
      case 3:
        *r = p; *g = q; *b = v;
        break;
      case 4:
        *r = t; *g = p; *b = v;
        break;
      case 5:
        *r = v; *g = p; *b = q;
        break;
      default:
        break;
    }
}

template <typename T>
__global__ void Hsvtorgb(const size_t input_size, const T *input, T *output) {
  for (size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
       idx < input_size / 3; idx += blockDim.x * gridDim.x) {
        T r, g, b;
        hsv2rgb(input[idx * 3], input[idx * 3 + 1], input[idx * 3 + 2], &r, &g, &b);
        output[idx * 3] = r;
        output[idx * 3 + 1] = g;
        output[idx * 3 + 2] = b;
  }
  return;
}

template <>
__global__ void Hsvtorgb(const size_t input_size, const half *input, half *output) {
  for (size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
       idx < input_size / 3; idx += blockDim.x * gridDim.x) {
        float r, g, b;
        hsv2rgb(static_cast<float>(input[idx * 3]),
                static_cast<float>(input[idx * 3 + 1]),
                static_cast<float>(input[idx * 3 + 2]), &r, &g, &b);
        output[idx * 3] = static_cast<half>(r);
        output[idx * 3 + 1] = static_cast<half>(g);
        output[idx * 3 + 2] = static_cast<half>(b);
  }
  return;
}

template <typename T>
void CalHsvtorgb(const size_t input_size, const T *input, T *output,
                  const uint32_t &device_id, cudaStream_t cuda_stream) {
  Hsvtorgb<<<CUDA_BLOCKS(device_id, input_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(input_size, input, output);
  return;
}

template CUDA_LIB_EXPORT void CalHsvtorgb<half>(const size_t input_size, const half *input, half *output,
                                                      const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalHsvtorgb<float>(const size_t input_size, const float *input, float *output,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalHsvtorgb<double>(const size_t input_size, const double *input, double *output,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
