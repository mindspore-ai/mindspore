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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/smooth_l1_loss_impl.cuh"
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T>
__global__ void SmoothL1LossNoReduce(const int64_t input_size, const float beta, const T *prediction, const T *target,
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

template <>
__global__ void SmoothL1LossNoReduce(const int64_t input_size, const float beta, const half *prediction,
                                     const half *target, half *loss) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
    half value = fabsf(prediction[i] - target[i]);
    half h_beta = __float2half(beta);
    if (value < h_beta) {
      loss[i] = __float2half(0.5) * value * value / h_beta;
    } else {
      loss[i] = value - (__float2half(0.5) * h_beta);
    }
  }
}

template <typename T>
__global__ void SmoothL1LossSum(const int64_t input_size, const float beta, const T *prediction, const T *target,
                                T *loss, double *tmp_loss) {
  double tmp = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
    double value = static_cast<double>(fabsf(prediction[i] - target[i]));
    if (value < beta) {
      tmp = 0.5 * value * value / beta;
    } else {
      tmp = value - (0.5 * beta);
    }
    MsAtomicAdd(tmp_loss, tmp);
  }
}

template <>
__global__ void SmoothL1LossSum(const int64_t input_size, const float beta, const half *prediction, const half *target,
                                half *loss, double *tmp_loss) {
  double tmp = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
    float value = __half2float(fabsf(prediction[i] - target[i]));
    if (value < beta) {
      tmp = 0.5 * value * value / beta;
    } else {
      tmp = value - 0.5 * beta;
    }
    MsAtomicAdd(tmp_loss, tmp);
  }
}

template <typename T>
__global__ void SmoothL1LossMean(const int64_t input_size, const float beta, const T *prediction, const T *target,
                                 T *loss, double *tmp_loss) {
  double tmp = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
    double value = static_cast<double>(fabsf(prediction[i] - target[i]));
    if (value < beta) {
      tmp = 0.5 * value * value / beta;
    } else {
      tmp = value - (0.5 * beta);
    }
    tmp /= static_cast<double>(input_size);
    MsAtomicAdd(tmp_loss, tmp);
  }
}

template <>
__global__ void SmoothL1LossMean(const int64_t input_size, const float beta, const half *prediction, const half *target,
                                 half *loss, double *tmp_loss) {
  double tmp = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
    float value = __half2float(fabsf(prediction[i] - target[i]));
    if (value < beta) {
      tmp = 0.5 * value * value / beta;
    } else {
      tmp = value - 0.5 * beta;
    }
    tmp /= static_cast<double>(input_size);
    MsAtomicAdd(tmp_loss, tmp);
  }
}

template <typename T>
__global__ void CastLoss(const double *tmp_loss, T *loss) {
  loss[0] = static_cast<T>(tmp_loss[0]);
}

template <>
__global__ void CastLoss(const double *tmp_loss, half *loss) {
  float f_loss = static_cast<float>(tmp_loss[0]);
  loss[0] = __half2float(f_loss);
}

template <typename T>
void SmoothL1Loss(const SmoothL1LossReductionMode mode, const int64_t input_size, const float beta, const T *prediction,
                  const T *target, T *loss, double *tmp_loss, const uint32_t device_id, cudaStream_t stream) {
  switch (mode) {
    case NONE: {
      return SmoothL1LossNoReduce<<<CUDA_BLOCKS(device_id, input_size), CUDA_THREADS(device_id), 0, stream>>>(
        input_size, beta, prediction, target, loss);
    }
    case SUM: {
      SmoothL1LossSum<<<CUDA_BLOCKS(device_id, input_size), CUDA_THREADS(device_id), 0, stream>>>(
        input_size, beta, prediction, target, loss, tmp_loss);
      return CastLoss<<<CUDA_BLOCKS(device_id, 1), CUDA_THREADS(device_id), 0, stream>>>(tmp_loss, loss);
    }

    case MEAN: {
      SmoothL1LossMean<<<CUDA_BLOCKS(device_id, input_size), CUDA_THREADS(device_id), 0, stream>>>(
        input_size, beta, prediction, target, loss, tmp_loss);
      return CastLoss<<<CUDA_BLOCKS(device_id, 1), CUDA_THREADS(device_id), 0, stream>>>(tmp_loss, loss);
    }
    default:
      break;
  }
}

template <typename T>
__global__ void SmoothL1LossGradNoReduce(const int64_t input_size, const float beta, const T *prediction,
                                         const T *target, const T *dloss, T *dx) {
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

template <>
__global__ void SmoothL1LossGradNoReduce(const int64_t input_size, const float beta, const half *prediction,
                                         const half *target, const half *dloss, half *dx) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
    half value = prediction[i] - target[i];
    half h_beta = __float2half(beta);
    if (value > h_beta) {
      dx[i] = dloss[i];
    } else if (value < -h_beta) {
      dx[i] = -dloss[i];
    } else {
      dx[i] = (value / h_beta) * dloss[i];
    }
  }
}

template <typename T>
__global__ void SmoothL1LossGradSum(const int64_t input_size, const float beta, const T *prediction, const T *target,
                                    const T *dloss, T *dx) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
    T value = prediction[i] - target[i];
    if (value > beta) {
      dx[i] = dloss[0];
    } else if (value < -beta) {
      dx[i] = -dloss[0];
    } else {
      dx[i] = (value / beta) * dloss[0];
    }
  }
}

template <>
__global__ void SmoothL1LossGradSum(const int64_t input_size, const float beta, const half *prediction,
                                    const half *target, const half *dloss, half *dx) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
    half value = prediction[i] - target[i];
    half h_beta = __float2half(beta);
    if (value > h_beta) {
      dx[i] = dloss[0];
    } else if (value < -h_beta) {
      dx[i] = -dloss[0];
    } else {
      dx[i] = (value / h_beta) * dloss[0];
    }
  }
}

template <typename T>
__global__ void SmoothL1LossGradMean(const int64_t input_size, const float beta, const T *prediction, const T *target,
                                     const T *dloss, T *dx) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
    T value = prediction[i] - target[i];
    T val = (T)(1) / (T)(input_size);
    if (value > beta) {
      dx[i] = dloss[0] * val;
    } else if (value < -beta) {
      dx[i] = -dloss[0] * val;
    } else {
      dx[i] = (value / beta) * dloss[0] * val;
    }
  }
}

template <>
__global__ void SmoothL1LossGradMean(const int64_t input_size, const float beta, const half *prediction,
                                     const half *target, const half *dloss, half *dx) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_size; i += blockDim.x * gridDim.x) {
    half value = prediction[i] - target[i];
    half h_beta = __float2half(beta);
    half val = (half)(1) / __float2half(static_cast<float>(input_size));
    if (value > h_beta) {
      dx[i] = dloss[0] * val;
    } else if (value < -h_beta) {
      dx[i] = -dloss[0] * val;
    } else {
      dx[i] = (value / h_beta) * dloss[0] * val;
    }
  }
}

template <typename T>
void SmoothL1LossGrad(const SmoothL1LossReductionMode mode, const int64_t input_size, const float beta,
                      const T *prediction, const T *target, const T *dloss, T *dx, const uint32_t device_id,
                      cudaStream_t stream) {
  switch (mode) {
    case NONE:
      return SmoothL1LossGradNoReduce<<<CUDA_BLOCKS(device_id, input_size), CUDA_THREADS(device_id), 0, stream>>>(
        input_size, beta, prediction, target, dloss, dx);
    case SUM:
      return SmoothL1LossGradSum<<<CUDA_BLOCKS(device_id, input_size), CUDA_THREADS(device_id), 0, stream>>>(
        input_size, beta, prediction, target, dloss, dx);
    case MEAN:
      return SmoothL1LossGradMean<<<CUDA_BLOCKS(device_id, input_size), CUDA_THREADS(device_id), 0, stream>>>(
        input_size, beta, prediction, target, dloss, dx);
    default:
      break;
  }
}

template CUDA_LIB_EXPORT void SmoothL1Loss<double>(const SmoothL1LossReductionMode mode, const int64_t input_size,
                                                   const float beta, const double *prediction, const double *target,
                                                   double *loss, double *tmp_loss, const uint32_t device_id,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT void SmoothL1LossGrad<double>(const SmoothL1LossReductionMode mode, const int64_t input_size,
                                                       const float beta, const double *prediction, const double *target,
                                                       const double *dloss, double *dx, const uint32_t device_id,
                                                       cudaStream_t stream);

template CUDA_LIB_EXPORT void SmoothL1Loss<float>(const SmoothL1LossReductionMode mode, const int64_t input_size,
                                                  const float beta, const float *prediction, const float *target,
                                                  float *loss, double *tmp_loss, const uint32_t device_id,
                                                  cudaStream_t stream);
template CUDA_LIB_EXPORT void SmoothL1LossGrad<float>(const SmoothL1LossReductionMode mode, const int64_t input_size,
                                                      const float beta, const float *prediction, const float *target,
                                                      const float *dloss, float *dx, const uint32_t device_id,
                                                      cudaStream_t stream);

template CUDA_LIB_EXPORT void SmoothL1Loss<half>(const SmoothL1LossReductionMode mode, const int64_t input_size,
                                                 const float beta, const half *prediction, const half *target,
                                                 half *loss, double *tmp_loss, const uint32_t device_id,
                                                 cudaStream_t stream);
template CUDA_LIB_EXPORT void SmoothL1LossGrad<half>(const SmoothL1LossReductionMode mode, const int64_t input_size,
                                                     const float beta, const half *prediction, const half *target,
                                                     const half *dloss, half *dx, const uint32_t device_id,
                                                     cudaStream_t stream);
