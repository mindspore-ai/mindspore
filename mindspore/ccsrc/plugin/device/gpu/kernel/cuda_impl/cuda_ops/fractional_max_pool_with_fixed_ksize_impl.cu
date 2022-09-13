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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/fractional_max_pool_with_fixed_ksize_impl.cuh"
#include <limits>

template <typename S>
__device__ inline int64_t get_intervals(S sample, int64_t index, int64_t inputSize, int64_t outputSize,
                                        int64_t poolSize) {
  S alpha = static_cast<S>(inputSize - poolSize) / static_cast<S>(outputSize - 1);
  if (index == outputSize - 1) {
    return inputSize - poolSize;
  } else {
    return static_cast<int64_t>((index + sample) * alpha) - static_cast<int64_t>(sample * alpha);
  }
}

// half
template <>
__device__ inline int64_t get_intervals(half sample, int64_t index, int64_t inputSize, int64_t outputSize,
                                        int64_t poolSize) {
  float alpha = static_cast<float>(inputSize - poolSize) / static_cast<float>(outputSize - 1);
  if (index == outputSize - 1) {
    return inputSize - poolSize;
  } else {
    return static_cast<int64_t>((index + __half2float(sample)) * alpha) -
           static_cast<int64_t>(__half2float(sample) * alpha);
  }
}

template <typename T, typename S, typename G>
__global__ void Fractionalmaxpoolwithfixedksize(const T *input, const S *random_samples, T *output, G *argmax,
                                                int64_t outputH, int64_t outputW, int64_t N,
                                                int64_t C, int64_t inputH, int64_t inputW,
                                                int64_t kernelsizeH, int64_t kernelsizeW,
                                                const int64_t outer_size) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < outer_size; pos += blockDim.x * gridDim.x) {
    const int posn = pos / (C * outputH * outputW);
    const int posc = pos / (outputH * outputW) % C;
    const int posh = pos / outputW % outputH;
    const int posw = pos % outputW;

    int64_t poolH = get_intervals<S>(random_samples[(posn * C + posc) * 2 + 1], posh, inputH, outputH, kernelsizeH);
    int64_t poolW = get_intervals<S>(random_samples[(posn * C + posc) * 2 + 0], posw, inputW, outputW, kernelsizeW);

    int64_t maxIndex = ((posn * C + posc) * inputH + poolH) * inputW + poolW;
    T maxVal = input[maxIndex];
    maxIndex = poolH * inputW + poolW;

    for (int64_t h = poolH; h < poolH + kernelsizeH; ++h) {
      for (int64_t w = poolW; w < poolW + kernelsizeW; ++w) {
        int64_t index = ((posn * C + posc) * inputH + h) * inputW + w;
        T val = input[index];
        if (val > maxVal) {
          maxVal = val;
          maxIndex = h * inputW + w;
        }
      }
    }
    argmax[pos] = static_cast<G>(maxIndex);
    output[pos] = maxVal;
  }
  return;
}

template <typename T, typename S, typename G>
void CalFractionalmaxpoolwithfixedksize(const T *input, const S *random_samples, T *output, G *argmax,
                                        int64_t outputH, int64_t outputW, int64_t inputN,
                                        int64_t inputC, int64_t inputH, int64_t inputW,
                                        int64_t kernelsizeH, int64_t kernelsizeW,
                                        const int64_t outer_size, const uint32_t &device_id,
                                        cudaStream_t cuda_stream) {
  Fractionalmaxpoolwithfixedksize<<<CUDA_BLOCKS(device_id, outer_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input, random_samples, output, argmax, outputH, outputW, inputN, inputC, inputH, inputW,
    kernelsizeH, kernelsizeW, outer_size);
  return;
}

template CUDA_LIB_EXPORT void CalFractionalmaxpoolwithfixedksize<half, float, int64_t>(
  const half *input, const float *random_samples, half *output, int64_t *argmax, int64_t outputH,
  int64_t outputW, int64_t inputN, int64_t inputC, int64_t inputH, int64_t inputW,
  int64_t kernelsizeH, int64_t kernelsizeW, const int64_t outer_size, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalmaxpoolwithfixedksize<half, double, int64_t>(
  const half *input, const double *random_samples, half *output, int64_t *argmax, int64_t outputH,
  int64_t outputW, int64_t inputN, int64_t inputC, int64_t inputH, int64_t inputW,
  int64_t kernelsizeH, int64_t kernelsizeW, const int64_t outer_size, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalmaxpoolwithfixedksize<half, half, int64_t>(
  const half *input, const half *random_samples, half *output, int64_t *argmax, int64_t outputH,
  int64_t outputW, int64_t inputN, int64_t inputC, int64_t inputH, int64_t inputW,
  int64_t kernelsizeH, int64_t kernelsizeW, const int64_t outer_size, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalmaxpoolwithfixedksize<float, float, int64_t>(
  const float *input, const float *random_samples, float *output, int64_t *argmax, int64_t outputH,
  int64_t outputW, int64_t inputN, int64_t inputC, int64_t inputH, int64_t inputW,
  int64_t kernelsizeH, int64_t kernelsizeW, const int64_t outer_size, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalmaxpoolwithfixedksize<float, double, int64_t>(
  const float *input, const double *random_samples, float *output, int64_t *argmax, int64_t outputH,
  int64_t outputW, int64_t inputN, int64_t inputC, int64_t inputH, int64_t inputW,
  int64_t kernelsizeH, int64_t kernelsizeW, const int64_t outer_size, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalmaxpoolwithfixedksize<float, half, int64_t>(
  const float *input, const half *random_samples, float *output, int64_t *argmax, int64_t outputH,
  int64_t outputW, int64_t inputN, int64_t inputC, int64_t inputH, int64_t inputW,
  int64_t kernelsizeH, int64_t kernelsizeW, const int64_t outer_size, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalmaxpoolwithfixedksize<double, float, int64_t>(
  const double *input, const float *random_samples, double *output, int64_t *argmax, int64_t outputH,
  int64_t outputW, int64_t inputN, int64_t inputC, int64_t inputH, int64_t inputW,
  int64_t kernelsizeH, int64_t kernelsizeW, const int64_t outer_size, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalmaxpoolwithfixedksize<double, double, int64_t>(
  const double *input, const double *random_samples, double *output, int64_t *argmax, int64_t outputH,
  int64_t outputW, int64_t inputN, int64_t inputC, int64_t inputH, int64_t inputW,
  int64_t kernelsizeH, int64_t kernelsizeW, const int64_t outer_size, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalmaxpoolwithfixedksize<double, half, int64_t>(
  const double *input, const half *random_samples, double *output, int64_t *argmax, int64_t outputH,
  int64_t outputW, int64_t inputN, int64_t inputC, int64_t inputH, int64_t inputW,
  int64_t kernelsizeH, int64_t kernelsizeW, const int64_t outer_size, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalmaxpoolwithfixedksize<int32_t, float, int64_t>(
  const int32_t *input, const float *random_samples, int32_t *output, int64_t *argmax, int64_t outputH,
  int64_t outputW, int64_t inputN, int64_t inputC, int64_t inputH, int64_t inputW,
  int64_t kernelsizeH, int64_t kernelsizeW, const int64_t outer_size, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalmaxpoolwithfixedksize<int32_t, double, int64_t>(
  const int32_t *input, const double *random_samples, int32_t *output, int64_t *argmax,
  int64_t outputH, int64_t outputW, int64_t inputN, int64_t inputC, int64_t inputH, int64_t inputW,
 int64_t kernelsizeH, int64_t kernelsizeW, const int64_t outer_size, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalmaxpoolwithfixedksize<int32_t, half, int64_t>(
  const int32_t *input, const half *random_samples, int32_t *output, int64_t *argmax, int64_t outputH,
  int64_t outputW, int64_t inputN, int64_t inputC, int64_t inputH, int64_t inputW,
  int64_t kernelsizeH, int64_t kernelsizeW, const int64_t outer_size, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalmaxpoolwithfixedksize<int64_t, float, int64_t>(
  const int64_t *input, const float *random_samples, int64_t *output, int64_t *argmax, int64_t outputH,
  int64_t outputW, int64_t inputN, int64_t inputC, int64_t inputH, int64_t inputW,
  int64_t kernelsizeH, int64_t kernelsizeW, const int64_t outer_size, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalmaxpoolwithfixedksize<int64_t, double, int64_t>(
  const int64_t *input, const double *random_samples, int64_t *output, int64_t *argmax,
  int64_t outputH, int64_t outputW, int64_t inputN, int64_t inputC, int64_t inputH, int64_t inputW,
 int64_t kernelsizeH, int64_t kernelsizeW, const int64_t outer_size, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalmaxpoolwithfixedksize<int64_t, half, int64_t>(
  const int64_t *input, const half *random_samples, int64_t *output, int64_t *argmax, int64_t outputH,
  int64_t outputW, int64_t inputN, int64_t inputC, int64_t inputH, int64_t inputW,
  int64_t kernelsizeH, int64_t kernelsizeW, const int64_t outer_size, const uint32_t &device_id,
  cudaStream_t cuda_stream);
