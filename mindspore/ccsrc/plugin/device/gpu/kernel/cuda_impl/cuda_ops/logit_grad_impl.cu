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

#include "logit_grad_impl.cuh"
#include <limits>
#include "include/cuda_fp16.h"

template <typename T>
__global__ void LogitGradLessZero(const T *grad, const T *input, const float eps, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = (input[i] < T(0) || input[i] > T(1)) ? std::numeric_limits<T>::quiet_NaN()
                                                     : (grad[i] / input[i] / (T(1) - input[i]));
  }
  return;
}

template <>
__global__ void LogitGradLessZero(const half *grad, const half *input, const float eps, half *output,
                                  const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = (input[i] < half(0) || input[i] > half(1))
                  ? half(std::numeric_limits<float>::quiet_NaN())
                  : half(static_cast<float>(grad[i]) / static_cast<float>(input[i]) /
                         (static_cast<float>(1) - static_cast<float>(input[i])));
  }
  return;
}

template <typename T>
__global__ void LogitGradGreaterZero(const T *grad, const T *input, const float eps, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = (input[i] < static_cast<T>(eps) || input[i] > T(1) - static_cast<T>(eps))
                  ? T(0)
                  : (grad[i] / input[i] / (T(1) - input[i]));
  }
  return;
}

template <>
__global__ void LogitGradGreaterZero(const half *grad, const half *input, const float eps, half *output,
                                     const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = (input[i] < static_cast<half>(eps) || input[i] > half(1) - static_cast<half>(eps))
                  ? half(0)
                  : half(static_cast<float>(grad[i]) / static_cast<float>(input[i]) /
                         (static_cast<float>(1) - static_cast<float>(input[i])));
  }
  return;
}

template <typename T>
cudaError_t CalLogitGrad(const T *grad, const T *input, const float eps, T *output, const size_t count,
                         const uint32_t &device_id, cudaStream_t cuda_stream) {
  if (eps < 0) {
    LogitGradLessZero<<<CUDA_BLOCKS(device_id, count), CUDA_THREADS(device_id), 0, cuda_stream>>>(grad, input, eps,
                                                                                                  output, count);
  } else {
    LogitGradGreaterZero<<<CUDA_BLOCKS(device_id, count), CUDA_THREADS(device_id), 0, cuda_stream>>>(grad, input, eps,
                                                                                                     output, count);
  }
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalLogitGrad<half>(const half *grad, const half *input, const float eps,
                                                        half *output, const size_t count, const uint32_t &device_id,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalLogitGrad<float>(const float *grad, const float *input, const float eps,
                                                         float *output, const size_t count, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalLogitGrad<double>(const double *grad, const double *input, const float eps,
                                                          double *output, const size_t count, const uint32_t &device_id,
                                                          cudaStream_t cuda_stream);
