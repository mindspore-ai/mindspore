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

#include <algorithm>
#include "loss_with_reduction_impl.cuh"
#include "runtime/device/gpu/cuda_common.h"

inline __device__ float logT(float x) { return logf(x); }
inline __device__ half logT(half x) { return hlog(x); }
inline __device__ float castT(float ref, int x) { return __int2float_rd(x); }
inline __device__ half castT(half ref, int x) { return __int2half_rd(x); }
inline __device__ float maxT(float a, float b) { return fmaxf(a, b); }
inline __device__ half maxT(half a, half b) { return a > b ? a : b; }

template <typename T>
__global__ void Copy(T *loss, T *tmp_loss, int reduction, int input_size) {
  loss[0] += tmp_loss[0];
  if (reduction == 1) {
    loss[0] /= castT(loss[0], input_size);
  }
}

template <typename T>
__global__ void AddTile(T *tmp_loss, int index) {
  tmp_loss[0] += tmp_loss[index];
}
template <typename T>
__global__ void PartialSum(T *tmp_loss, int stride) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < stride; i += blockDim.x * gridDim.x) {
    tmp_loss[i] += tmp_loss[i + stride];
  }
}

template <typename T>
__global__ void LossInitKernel(T *loss) {
  loss[0] = static_cast<T>(0.);
}

template <typename T>
__global__ void KLDivLossKernel(const int input_size, const int reduction, const T *input_x, const T *input_y, T *loss,
                                T *tmp_loss) {
  T epsilon = 1e-6;
  if (reduction == 0) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
      T denominator = maxT(input_y[i], epsilon);
      T value = input_y[i] * (logT(denominator) - input_x[i]);
      loss[i] = value;
    }
  } else {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
      T denominator = maxT(input_y[i], epsilon);
      T value = input_y[i] * (logT(denominator) - input_x[i]);
      tmp_loss[i] = value;
    }
  }
}

template <typename T>
void KLDivLoss(const int &input_size, const int &reduction, const T *input_x, const T *input_y, T *loss, T *tmp_loss,
               cudaStream_t stream) {
  LossInitKernel<<<1, 1, 0, stream>>>(loss);
  KLDivLossKernel<<<GET_BLOCKS(input_size), GET_THREADS, 0, stream>>>(input_size, reduction, input_x, input_y, loss,
                                                                      tmp_loss);
  if (reduction != 0) {
    if (input_size % 2 == 1) {
      AddTile<<<1, 1, 0, stream>>>(tmp_loss, input_size - 1);
    }
    for (int stride = input_size / 2; stride > 0; stride >>= 1) {
      PartialSum<<<GET_BLOCKS(stride), GET_THREADS, 0, stream>>>(tmp_loss, stride);
      if (stride > 2 && stride % 2 == 1) {
        AddTile<<<1, 1, 0, stream>>>(tmp_loss, stride - 1);
      }
    }
    Copy<<<1, 1, 0, stream>>>(loss, tmp_loss, reduction, input_size);
  }
}

template <typename T>
__global__ void KLDivLossGradKernel(const int input_size, const int reduction, const T *input_x, const T *input_y,
                                    const T *dloss, T *dx, T *dy) {
  T epsilon = 1e-6;
  T one = static_cast<T>(1);
  if (reduction == 0) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
      T denominator = maxT(input_y[i], epsilon);
      dx[i] = -input_y[i] * dloss[i];
      dy[i] = (logT(denominator) + one - input_x[i]) * dloss[i];
    }
  } else {
    T dloss1 = dloss[0];
    if (reduction == 1) {
      dloss1 = dloss[0] / castT(dloss[0], input_size);
    }
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
      T denominator = maxT(input_y[i], epsilon);
      dx[i] = -input_y[i] * dloss1;
      dy[i] = (logT(denominator) + one - input_x[i]) * dloss1;
    }
  }
}

template <typename T>
void KLDivLossGrad(const int &input_size, const int &reduction, const T *input_x, const T *input_y, const T *dloss,
                   T *dx, T *dy, cudaStream_t stream) {
  KLDivLossGradKernel<<<GET_BLOCKS(input_size), GET_THREADS, 0, stream>>>(input_size, reduction, input_x, input_y,
                                                                          dloss, dx, dy);
}

template <typename T>
__global__ void BinaryCrossEntropyLossKernel(const int input_size, const int reduction, const T *input_x,
                                             const T *input_y, const T *weight, T *loss, T *tmp_loss) {
  T epsilon = 1e-12;
  T one = static_cast<T>(1);
  if (reduction == 0) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
      T value =
        -weight[i] * (input_y[i] * logT(input_x[i] + epsilon) + (one - input_y[i]) * logT(one - input_x[i] + epsilon));
      loss[i] = value;
    }
  } else {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
      T value =
        -weight[i] * (input_y[i] * logT(input_x[i] + epsilon) + (one - input_y[i]) * logT(one - input_x[i] + epsilon));
      tmp_loss[i] = value;
    }
  }
}

template <typename T>
void BinaryCrossEntropyLoss(const int &input_size, const int &reduction, const T *input_x, const T *input_y,
                            const T *weight, T *loss, T *tmp_loss, cudaStream_t stream) {
  LossInitKernel<<<1, 1, 0, stream>>>(loss);
  BinaryCrossEntropyLossKernel<<<GET_BLOCKS(input_size), GET_THREADS, 0, stream>>>(input_size, reduction, input_x,
                                                                                   input_y, weight, loss, tmp_loss);
  if (reduction != 0) {
    if (input_size % 2 == 1) {
      AddTile<<<1, 1, 0, stream>>>(tmp_loss, input_size - 1);
    }
    for (int stride = input_size / 2; stride > 0; stride >>= 1) {
      PartialSum<<<GET_BLOCKS(stride), GET_THREADS, 0, stream>>>(tmp_loss, stride);
      if (stride > 2 && stride % 2 == 1) {
        AddTile<<<1, 1, 0, stream>>>(tmp_loss, stride - 1);
      }
    }
    Copy<<<1, 1, 0, stream>>>(loss, tmp_loss, reduction, input_size);
  }
}

template <typename T>
__global__ void BinaryCrossEntropyLossGradKernel(const int input_size, const int reduction, const T *input_x,
                                                 const T *input_y, const T *weight, const T *dloss, T *dx) {
  T epsilon = 1e-12;
  T one = static_cast<T>(1);
  if (reduction == 0) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
      T denominator = maxT(input_x[i] * (one - input_x[i]), epsilon);
      T value = weight[i] * (input_x[i] - input_y[i]) / denominator;
      dx[i] = value * dloss[i];
    }
  } else {
    T dloss1 = dloss[0];
    if (reduction == 1) {
      dloss1 = dloss[0] / castT(dloss[0], input_size);
    }
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
      T denominator = maxT(input_x[i] * (one - input_x[i]), epsilon);
      T value = weight[i] * (input_x[i] - input_y[i]) / denominator;
      dx[i] = value * dloss1;
    }
  }
}

template <typename T>
void BinaryCrossEntropyLossGrad(const int &input_size, const int &reduction, const T *input_x, const T *input_y,
                                const T *weight, const T *dloss, T *dx, cudaStream_t stream) {
  BinaryCrossEntropyLossGradKernel<<<GET_BLOCKS(input_size), GET_THREADS, 0, stream>>>(input_size, reduction, input_x,
                                                                                       input_y, weight, dloss, dx);
}

template void KLDivLoss<float>(const int &input_size, const int &reduction, const float *input_x, const float *input_y,
                               float *loss, float *tmp_loss, cudaStream_t stream);

template void KLDivLossGrad<float>(const int &input_size, const int &reduction, const float *input_x,
                                   const float *input_y, const float *dloss, float *dx, float *dy, cudaStream_t stream);

template void BinaryCrossEntropyLoss<float>(const int &input_size, const int &reduction, const float *input_x,
                                            const float *input_y, const float *weight, float *loss, float *tmp_loss,
                                            cudaStream_t stream);

template void BinaryCrossEntropyLossGrad<float>(const int &input_size, const int &reduction, const float *input_x,
                                                const float *input_y, const float *weight, const float *dloss,
                                                float *dx, cudaStream_t stream);

template void KLDivLoss<half>(const int &input_size, const int &reduction, const half *input_x, const half *input_y,
                              half *loss, half *tmp_loss, cudaStream_t stream);

template void KLDivLossGrad<half>(const int &input_size, const int &reduction, const half *input_x, const half *input_y,
                                  const half *dloss, half *dx, half *dy, cudaStream_t stream);

template void BinaryCrossEntropyLoss<half>(const int &input_size, const int &reduction, const half *input_x,
                                           const half *input_y, const half *weight, half *loss, half *tmp_loss,
                                           cudaStream_t stream);

template void BinaryCrossEntropyLossGrad<half>(const int &input_size, const int &reduction, const half *input_x,
                                               const half *input_y, const half *weight, const half *dloss, half *dx,
                                               cudaStream_t stream);
