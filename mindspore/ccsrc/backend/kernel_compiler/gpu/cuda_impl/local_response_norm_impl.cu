/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/gpu/cuda_impl/local_response_norm_impl.cuh"
#include "runtime/device/gpu/cuda_common.h"
#include "include/cuda_fp16.h"

template <typename T>
__global__ void ComputeScaleNHWC(const T *input, const int depth_radius, const float bias, const float alpha,
  const size_t channels, const size_t num_elements, float *scale) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < num_elements; pos += blockDim.x * gridDim.x) {
    const int posc = static_cast<int>(pos % channels);
    float sqr_sum = 0;
    for (int i = -depth_radius; i < depth_radius + 1; i++) {
      if (posc + i >= 0 && posc + i < static_cast<int>(channels)) {
        float a = static_cast<float>(input[pos + i]);
        sqr_sum += a * a;
      }
    }
    scale[pos] = bias + alpha * sqr_sum;
  }
  return;
}

template <typename T>
__global__ void LocalResponseNormNHWC(const T *input, const float *scale, const float beta, const size_t num_elements,
  T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < num_elements; pos += blockDim.x * gridDim.x) {
    float z = expf(logf(scale[pos]) * -beta);
    output[pos] = input[pos] * static_cast<T>(z);
  }
  return;
}

template <typename T>
__global__ void LocalResponseNormGradNHWC(const T *dy, const T *x, const T *y, const float *scale,
  const int depth_radius, const float alpha, const float beta, const float neg2_alpha_beta, const size_t channels,
  const size_t num_elements, T *dx) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < num_elements; pos += blockDim.x * gridDim.x) {
    const int posc = static_cast<int>(pos % channels);
    float ratio_sum = 0;
    for (int i = -depth_radius; i <= depth_radius; i++) {
      if (posc + i >= 0 && posc + i < static_cast<int>(channels)) {
        ratio_sum += static_cast<float>(dy[pos + i] * y[pos + i]) / scale[pos + i];
      }
    }
    float z = expf(logf(scale[pos]) * -beta);
    float ratio_2ab = ratio_sum * neg2_alpha_beta;
    dx[pos] = dy[pos] * static_cast<T>(z) + x[pos] * static_cast<T>(ratio_2ab);
  }
  return;
}

template <typename T>
void CalLocalResponseNormNHWC(const T *input, const int depth_radius, const float bias, const float alpha,
  const float beta, const size_t channels, const size_t num_elements, float *scale, T *output,
  cudaStream_t cuda_stream) {
  ComputeScaleNHWC<<<GET_BLOCKS(num_elements), GET_THREADS, 0, cuda_stream>>>(input, depth_radius, bias, alpha,
    channels, num_elements, scale);
  LocalResponseNormNHWC<<<GET_BLOCKS(num_elements), GET_THREADS, 0, cuda_stream>>>(input, scale, beta, num_elements,
    output);
  return;
}

template <typename T>
void CalLocalResponseNormGradNHWC(const T *dy, const T *x, const T *y, const int depth_radius, const float bias,
  const float alpha, const float beta, const size_t channels, const size_t num_elements, float *scale, T *dx,
  cudaStream_t cuda_stream) {
  float neg2_alpha_beta = -2.0f * alpha * beta;
  ComputeScaleNHWC<<<GET_BLOCKS(num_elements), GET_THREADS, 0, cuda_stream>>>(x, depth_radius, bias, alpha, channels,
    num_elements, scale);
  LocalResponseNormGradNHWC<<<GET_BLOCKS(num_elements), GET_THREADS, 0, cuda_stream>>>(dy, x, y, scale, depth_radius,
    alpha, beta, neg2_alpha_beta, channels, num_elements, dx);
  return;
}

template void CalLocalResponseNormNHWC<float>(const float *input, const int depth_radius, const float bias,
  const float alpha, const float beta, const size_t channels, const size_t num_elements, float *scale, float *output,
  cudaStream_t cuda_stream);

template void CalLocalResponseNormNHWC<half>(const half *input, const int depth_radius, const float bias,
  const float alpha, const float beta, const size_t channels, const size_t num_elements, float *scale, half *output,
  cudaStream_t cuda_stream);

template void CalLocalResponseNormGradNHWC<float>(const float *dy, const float *x, const float *y,
  const int depth_radius, const float bias, const float alpha, const float beta, const size_t channels,
  const size_t num_elements, float *scale, float *dx, cudaStream_t cuda_stream);

template void CalLocalResponseNormGradNHWC<half>(const half *dy, const half *x, const half *y,
  const int depth_radius, const float bias, const float alpha, const float beta, const size_t channels,
  const size_t num_elements, float *scale, half *dx, cudaStream_t cuda_stream);
