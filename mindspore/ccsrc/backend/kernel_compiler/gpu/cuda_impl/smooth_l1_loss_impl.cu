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

#include "smooth_l1_loss_impl.cuh"
#include "runtime/device/gpu/cuda_common.h"

template <typename T>
__global__ void SmoothL1LossKernel(const int input_size, const float beta, const T *prediction, const T *target,
                                   T *loss) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
    T value = fabsf(prediction[i] - target[i]);
    if (value < beta) {
      loss[i] = 0.5 * value * value / beta;
    } else {
      loss[i] = value - (0.5 * beta);
    }
  }
}

template <typename T>
void SmoothL1Loss(const int &input_size, const float &beta, const T *prediction, const T *target, T *loss,
                  cudaStream_t stream) {
  SmoothL1LossKernel<<<GET_BLOCKS(input_size), GET_THREADS, 0, stream>>>(input_size, beta, prediction, target, loss);
}

template <typename T>
__global__ void SmoothL1LossGradKernel(const int input_size, const float beta, const T *prediction, const T *target,
                                       const T *dloss, T *dx) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
    T value = prediction[i] - target[i];
    if (value > beta) {
      dx[i] = dloss[i];
    } else if (value < -beta) {
      dx[i] = -dloss[i];
    } else {
      dx[i] = (value / beta) * dloss[i];
    }
  }
}

template <typename T>
void SmoothL1LossGrad(const int &input_size, const float &beta, const T *prediction, const T *target, const T *dloss,
                      T *dx, cudaStream_t stream) {
  SmoothL1LossGradKernel<<<GET_BLOCKS(input_size), GET_THREADS, 0, stream>>>(input_size, beta, prediction, target,
                                                                             dloss, dx);
}

template void SmoothL1Loss<float>(const int &input_size, const float &beta, const float *prediction,
                                  const float *target, float *loss, cudaStream_t stream);
template void SmoothL1LossGrad<float>(const int &input_size, const float &beta, const float *prediction,
                                      const float *target, const float *dloss, float *dx, cudaStream_t stream);
