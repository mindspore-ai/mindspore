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

#include "logit_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__global__ void LogitGreaterZero(const T *input, const T up_bound, const T eps, T *output, const size_t count) {
  T one = T(1);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    T z;
    T x = input[i];
    z = x < eps ? eps : (x > up_bound ? up_bound : x);
    output[i] = log(z / (one - z));
  }
  return;
}
template <>
__global__ void LogitGreaterZero(const half *input, const half up_bound, const half eps, half *output,
                                 const size_t count) {
  half one = half(1);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    half z;
    half x = input[i];
    z = x < eps ? eps : (x > up_bound ? up_bound : x);
    output[i] = hlog(z / (one - z));
  }
  return;
}

template <typename T>
__global__ void LogitLessZero(const T *input, const T up_bound, const T eps, T *output, const size_t count) {
  T one = T(1);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    T x = input[i];
    output[i] = log(x / (one - x));
  }
  return;
}
template <>
__global__ void LogitLessZero(const half *input, const half up_bound, const half eps, half *output,
                              const size_t count) {
  half one = half(1);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    half x = input[i];
    output[i] = hlog(x / (one - x));
  }
  return;
}

template <typename T>
cudaError_t CalLogit(const T *input, const T up_bound, const float eps, T *output, const size_t count,
                     const uint32_t &device_id, cudaStream_t cuda_stream) {
  T eps_value;
  eps_value = T(eps);
  if (eps < 0) {
    LogitLessZero<<<CUDA_BLOCKS(device_id, count), CUDA_THREADS(device_id), 0, cuda_stream>>>(input, up_bound,
                                                                                              eps_value, output, count);
  } else {
    LogitGreaterZero<<<CUDA_BLOCKS(device_id, count), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      input, up_bound, eps_value, output, count);
  }
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalLogit<half>(const half *input, const half up_bound, const float eps,
                                                    half *output, const size_t count, const uint32_t &device_id,
                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalLogit<float>(const float *input, const float up_bound, const float eps,
                                                     float *output, const size_t count, const uint32_t &device_id,
                                                     cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalLogit<double>(const double *input, const double up_bound, const float eps,
                                                      double *output, const size_t count, const uint32_t &device_id,
                                                      cudaStream_t cuda_stream);
