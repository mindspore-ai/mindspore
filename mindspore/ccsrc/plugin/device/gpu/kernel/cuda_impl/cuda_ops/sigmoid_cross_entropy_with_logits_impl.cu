/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sigmoid_cross_entropy_with_logits_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T, typename S>
__global__ void SigmoidCrossEntropyWithLogitsKernel(const size_t size, const T *logits, const S *labels, T *outputs) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    const T reverse_factor = static_cast<T>(logits[i] >= 0);
    outputs[i] =
      log1p(exp(logits[i] - static_cast<T>(2) * reverse_factor * logits[i])) - logits[i] * (labels[i] - reverse_factor);
  }
}

template <>
__global__ void SigmoidCrossEntropyWithLogitsKernel(const size_t size, const half *logits, const half *labels,
                                                    half *outputs) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    const half reverse_factor = static_cast<half>(logits[i] >= static_cast<half>(0.));
    const float exp_logit = exp(__half2float(logits[i] - static_cast<half>(2) * reverse_factor * logits[i]));
    outputs[i] = __float2half(log1p(exp_logit)) - logits[i] * (labels[i] - reverse_factor);
  }
}

template <typename T, typename S>
cudaError_t SigmoidCrossEntropyWithLogits(const size_t size, const T *logits, const S *labels, T *outputs,
                                          cudaStream_t cuda_stream) {
  SigmoidCrossEntropyWithLogitsKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, logits, labels, outputs);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t SigmoidCrossEntropyWithLogits<half, half>(const size_t size, const half *logits,
                                                                               const half *labels, half *outputs,
                                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t SigmoidCrossEntropyWithLogits<float, float>(const size_t size, const float *logits,
                                                                                 const float *labels, float *outputs,
                                                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t SigmoidCrossEntropyWithLogits<double, double>(
  const size_t size, const double *logits, const double *labels, double *outputs, cudaStream_t cuda_stream);
