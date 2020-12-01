/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/gpu/cuda_impl/adagrad_impl.cuh"

template <typename T>
__device__ __forceinline__ T SqrtFunc(T input) {
  return sqrt(input);
}

template <>
__device__ __forceinline__ half SqrtFunc(half input) {
  return hsqrt(input);
}

template <typename T>
__global__ void ApplyAdagradKernel(const size_t size,
                                   const bool update_slots,
                                   const T *learning_rate,
                                   const T *gradient,
                                   T *variable,
                                   T *accumulation) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    if (update_slots) {
      accumulation[i] += gradient[i] * gradient[i];
    }
    variable[i] -= learning_rate[0] * gradient[i] / SqrtFunc(accumulation[i]);
  }
}

template <typename T>
void ApplyAdagrad(const size_t size,
                  const bool update_slots,
                  const T *learning_rate,
                  const T *gradient,
                  T *variable,
                  T *accumulation,
                  cudaStream_t cuda_stream) {
  ApplyAdagradKernel<<< GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(
          size, update_slots, learning_rate, gradient, variable, accumulation);
}

template void ApplyAdagrad<float>(const size_t size,
                                  const bool update_slots,
                                  const float *learning_rate,
                                  const float *gradient,
                                  float *variable,
                                  float *accumulation,
                                  cudaStream_t cuda_stream);

template void ApplyAdagrad<half>(const size_t size,
                                 const bool update_slots,
                                 const half *learning_rate,
                                 const half *gradient,
                                 half *variable,
                                 half *accumulation,
                                 cudaStream_t cuda_stream);
