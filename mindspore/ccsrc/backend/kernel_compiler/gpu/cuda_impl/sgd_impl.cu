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
#include "backend/kernel_compiler/gpu/cuda_impl/sgd_impl.cuh"

template <typename T>
__global__ void SGDKernel(const int size, const T dampening, const T weight_decay, const bool nesterov, const T *grad,
                          const T *momentum, const T *lr, T *param, T *accum, T *stat) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (size); i += blockDim.x * gridDim.x) {
    T grad_new = grad[i];
    if (weight_decay > static_cast<T>(0)) {
      grad_new += param[i] * weight_decay;
    }

    if (momentum[0] > static_cast<T>(0)) {
      if (stat[i] > static_cast<T>(0)) {
        accum[i] = grad_new;
        stat[i] = 0;
      } else {
        accum[i] = accum[i] * momentum[0] + (1.0 - dampening) * grad_new;
      }

      if (nesterov) {
        grad_new += accum[i] * momentum[0];
      } else {
        grad_new = accum[i];
      }
    }

    param[i] -= lr[0] * grad_new;
  }
}

template <typename T>
void SGD(const int size, const T dampening, const T weight_decay, const bool nesterov, const T *lr, const T *momentum,
         const T *grad, T *param, T *accum, T *stat, cudaStream_t cuda_stream) {
  SGDKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, dampening, weight_decay, nesterov, grad, momentum,
                                                               lr, param, accum, stat);
}

template void SGD(const int size, const float dampening, const float weight_decay, const bool nesterov, const float *lr,
                  const float *momentum, const float *grad, float *param, float *accum, float *stat,
                  cudaStream_t cuda_stream);
