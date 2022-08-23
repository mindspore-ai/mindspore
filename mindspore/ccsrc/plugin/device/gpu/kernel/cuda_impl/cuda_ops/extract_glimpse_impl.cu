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

#include <curand_kernel.h>
#include <string>
#include <random>
#include <iostream>
#include <algorithm>
#include <utility>
#include <functional>
#include <cmath>
#include <tuple>
#include <type_traits>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/extract_glimpse_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "include/cuda_runtime.h"
#include "include/cuda_fp16.h"

template <typename T>
__global__ void CalExtractGlimpseKernel(const size_t output_size, const size_t batch_cnt, const size_t channels,
                                        const size_t image_height, const size_t image_width,
                                        ExtractGlimpsenoiseMode noise, const bool centered, const bool normalized,
                                        const bool uniform_noise, const T *inputs, const int *size, const T *offsets,
                                        T *output) {
  curandState localState;
  int64_t g_height = size[0], g_width = size[1];
  int64_t size1 = image_width * image_height * channels;
  int64_t size2 = image_width * channels;
  int64_t g_size = g_width * g_height;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < output_size; pos += blockDim.x * gridDim.x) {
    int64_t i = pos / (g_size * channels);
    float x = offsets[i << 1];
    float y = offsets[(i << 1) + 1];
    if (normalized) {
      x *= image_height;
      y *= image_width;
    }
    if (centered) {
      x /= 2.0;
      y /= 2.0;
      x += image_height / 2.0;
      y += image_width / 2.0;
    }
    x -= g_height / 2.0;
    y -= g_width / 2.0;
    int64_t v = (pos - i * g_size * channels) / channels;
    int64_t j = v / g_width, k = v % g_width;
    int64_t x_i = static_cast<int64_t>(x);
    int64_t y_i = static_cast<int64_t>(y);
    int64_t a = x_i + j, b = y_i + k;
    int64_t u = pos % channels;
    if (a >= static_cast<int64_t>(image_height) || b >= static_cast<int64_t>(image_width) || a < 0 || b < 0) {
      if (uniform_noise) {
        output[pos] = curand_uniform(&localState);
      } else if (noise == ExtractGlimpsenoiseMode::ZERO) {
        output[pos] = 0.0f;
      } else if (noise == ExtractGlimpsenoiseMode::GAUSSIAN) {
        output[pos] = curand_normal(&localState);
      } else if (noise == ExtractGlimpsenoiseMode::UNIFORM) {
        output[pos] = curand_uniform(&localState);
      }
    } else {
      int64_t w = i * size1 + a * size2 + b * channels;
      output[pos] = inputs[w + u];
    }
  }
}

template <typename T>
void CalExtractGlimpse(const size_t output_size, const size_t batch_cnt, const size_t channels,
                       const size_t image_height, const size_t image_width, const ExtractGlimpsenoiseMode noise,
                       const bool centered, const bool normalized, const bool uniform_noise, const T *inputs,
                       const int *size, const T *offsets, T *output, cudaStream_t cuda_stream) {
  CalExtractGlimpseKernel<<<GET_BLOCKS(output_size), GET_THREADS, 0, cuda_stream>>>(
    output_size, batch_cnt, channels, image_height, image_width, noise, centered, normalized, uniform_noise, inputs,
    size, offsets, output);
  return;
}

template CUDA_LIB_EXPORT void CalExtractGlimpse<float>(const size_t output_size, const size_t batch_cnt,
                                                       const size_t channels, const size_t image_height,
                                                       const size_t image_width, const ExtractGlimpsenoiseMode noise,
                                                       const bool centered, const bool normalized,
                                                       const bool uniform_noise, const float *inputs, const int *size,
                                                       const float *offsets, float *output, cudaStream_t cuda_stream);
