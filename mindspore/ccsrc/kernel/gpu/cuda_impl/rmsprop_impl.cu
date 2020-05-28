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

#include <iostream>
#include "kernel/gpu/cuda_impl/rmsprop_impl.cuh"
#include "device/gpu/cuda_common.h"

template <typename T>
__global__ void RmsPropKernel(const T* learning_rate, const T* decay, const T* momentum, const T* epsilon, T* variable,
                              T* mean_square, T*moment, T* gradients, const size_t size) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (size); i += blockDim.x * gridDim.x)  {
    mean_square[i] = decay[0] * mean_square[i] + (1.0 - decay[0]) * gradients[i] * gradients[i];
    moment[i] = momentum[0] * moment[i] + learning_rate[0] * rsqrt(mean_square[i] + epsilon[0]) * gradients[i];
    variable[i] -= moment[i];
  }
}

template <typename T>
void RmsProp(const T* learning_rate, const T* decay, const T* momentum, const T* epsilon,
             T* variable, T* mean_square, T* moment, T* gradients, const size_t size, cudaStream_t cuda_stream) {
  RmsPropKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(learning_rate, decay, momentum, epsilon,
                                                                   variable, mean_square, moment, gradients, size);
}

template <typename T>
__global__ void RmsPropCenterKernel(const T* learning_rate, const T* decay, const T* momentum, const T* epsilon,
                                    T* variable, T* mean_gradients, T* mean_square, T*moment, T* gradients,
                                    const size_t size) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (size); i += blockDim.x * gridDim.x) {
    mean_gradients[i] = decay[0] * mean_gradients[i] + (1.0 - decay[0]) * gradients[i];
    mean_square[i] = decay[0] * mean_square[i] + (1.0 - decay[0]) * gradients[i] * gradients[i];
    moment[i] = momentum[0] * moment[i] + learning_rate[0] *
                rsqrt(mean_square[i] - mean_gradients[i] * mean_gradients[i] + epsilon[0]) * gradients[i];
    variable[i] -= moment[i];
  }
}

template <typename T>
void RmsPropCenter(const T* learning_rate, const T* decay, const T* momentum, const T* epsilon, T* variable,
                   T* mean_gradients, T* mean_square,  T*moment, T* gradients, const size_t size,
                   cudaStream_t cuda_stream) {
  RmsPropCenterKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(learning_rate, decay, momentum, epsilon,
                                                                         variable, mean_gradients, mean_square,
                                                                         moment, gradients, size);
}

template
void RmsProp(const float* learning_rate, const float* decay, const float* momentum, const float* epsilon,
            float* variable, float* mean_square, float* moment, float* gradients, const size_t size,
            cudaStream_t cuda_stream);

template
void RmsPropCenter(const float* learning_rate, const float* decay, const float* momentum, const float* epsilon,
                   float* variable, float* mean_gradients, float* mean_square, float*moment, float* gradients,
                   const size_t size, cudaStream_t cuda_stream);
