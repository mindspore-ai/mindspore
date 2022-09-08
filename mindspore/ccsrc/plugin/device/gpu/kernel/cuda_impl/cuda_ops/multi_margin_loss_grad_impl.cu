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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/multi_margin_loss_grad_impl.cuh"
#include "include/cuda_runtime.h"
#include "include/cuda_fp16.h"

constexpr int MULTIMARGIN_THREADS = 128;

template <int P, typename scalar_t>
__global__ void MultiMarginLoss_backward_kernel(scalar_t *gradInput, const scalar_t *gradOutput, const scalar_t *input,
                                                const int64_t *target, const scalar_t *weights, int nframe, int dim,
                                                bool sizeAverage, scalar_t margin, bool reduce) {
  __shared__ scalar_t buffer[MULTIMARGIN_THREADS];
  int k = blockIdx.x;
  const scalar_t *input_k = input + k * dim;
  scalar_t *gradInput_k = gradInput + k * dim;
  CUDA_KERNEL_ASSERT(target[k] >= 0);
  int target_k = static_cast<int>(target[k]);
  scalar_t input_target_k = input_k[target_k];

  const scalar_t *gradOutput_k = gradOutput;
  if (!reduce) {
    gradOutput_k += k;
  }

  const int denom = sizeAverage && reduce ? nframe * dim : dim;
  const scalar_t g = scalar_t(1) / static_cast<scalar_t>(denom);

  int i_start = threadIdx.x;
  int i_end = dim;
  int i_step = blockDim.x;

  buffer[threadIdx.x] = 0;
  for (int i = i_start; i < i_end; i += i_step) {
    scalar_t z =
      static_cast<scalar_t>(margin) - static_cast<scalar_t>(input_target_k) + static_cast<scalar_t>(input_k[i]);
    if (i == target_k) {
      continue;
    }

    if (z > static_cast<scalar_t>(0)) {
      scalar_t h = (P == 1) ? g : static_cast<scalar_t>(2) * static_cast<scalar_t>(g) * static_cast<scalar_t>(z);
      if (weights) {
        h *= static_cast<scalar_t>(weights[target_k]);
      }
      buffer[threadIdx.x] -= static_cast<scalar_t>(h);
      gradInput_k[i] = static_cast<scalar_t>(static_cast<scalar_t>(h) * static_cast<scalar_t>(*gradOutput_k));
    } else {
      gradInput_k[i] = static_cast<scalar_t>(0);
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    scalar_t gradInput_target_k = 0;
    for (int i = 0; i < blockDim.x; i++) {
      gradInput_target_k += static_cast<scalar_t>(buffer[i]);
    }
    gradInput_k[target_k] =
      static_cast<scalar_t>(static_cast<scalar_t>(gradInput_target_k) * static_cast<scalar_t>(*gradOutput_k));
  }
}

template <int P>
__global__ void MultiMarginLoss_backward_kernel_half(half *gradInput, const half *gradOutput, const half *input,
                                                     const int64_t *target, const half *weights, int nframe, int dim,
                                                     bool sizeAverage, half margin, bool reduce) {
  __shared__ double buffer[MULTIMARGIN_THREADS];
  int k = blockIdx.x;
  const half *input_k = input + k * dim;
  half *gradInput_k = gradInput + k * dim;
  CUDA_KERNEL_ASSERT(target[k] >= 0);
  int target_k = static_cast<int>(target[k]);
  half input_target_k = input_k[target_k];

  const half *gradOutput_k = gradOutput;
  if (!reduce) {
    gradOutput_k += k;
  }

  const int denom = sizeAverage && reduce ? nframe * dim : dim;
  const double g = static_cast<double>(1) / static_cast<double>(denom);

  int i_start = threadIdx.x;
  int i_end = dim;
  int i_step = blockDim.x;

  buffer[threadIdx.x] = 0;
  for (int i = i_start; i < i_end; i += i_step) {
    double z = static_cast<double>(margin) - static_cast<double>(input_target_k) + static_cast<double>(input_k[i]);
    if (i == target_k) {
      continue;
    }

    if (z > static_cast<double>(0)) {
      double h = (P == 1) ? g : static_cast<double>(2) * static_cast<double>(g) * static_cast<double>(z);
      if (weights) {
        h *= static_cast<double>(weights[target_k]);
      }
      buffer[threadIdx.x] -= static_cast<double>(h);
      gradInput_k[i] = static_cast<half>(static_cast<double>(h) * static_cast<double>(*gradOutput_k));
    } else {
      gradInput_k[i] = static_cast<half>(0);
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    double gradInput_target_k = 0;
    for (int i = 0; i < blockDim.x; i++) {
      gradInput_target_k += static_cast<double>(buffer[i]);
    }
    gradInput_k[target_k] =
      static_cast<half>(static_cast<double>(gradInput_target_k) * static_cast<double>(*gradOutput_k));
  }
}

// namespace str
template <typename T>
void MultiMarginLossGrad(int64_t p, float margin, int64_t reduction, int nframe, int dim,
                         const T *output_grad, const T *input, const int64_t *target, const T *weight,
                         T *output, const uint32_t &device_id, cudaStream_t cuda_stream) {
  dim3 blocks1(nframe);
  dim3 threads1(MULTIMARGIN_THREADS);
  bool reduce = false;
  bool sizeAverage = false;
  if (reduction == 1) {
    reduce = true;
    sizeAverage = true;
  } else if (reduction == 0) {
    reduce = true;
    sizeAverage = false;
  }
  if (p == 1)
    MultiMarginLoss_backward_kernel<1><<<blocks1, threads1, 0, cuda_stream>>>(
      output, output_grad, input, target, weight, nframe, dim, sizeAverage, (T)margin, reduce);
  else
    MultiMarginLoss_backward_kernel<2><<<blocks1, threads1, 0, cuda_stream>>>(
      output, output_grad, input, target, weight, nframe, dim, sizeAverage, (T)margin, reduce);
  return;
}

// namespace str
template <>
void MultiMarginLossGrad(int64_t p, float margin, int64_t reduction, int nframe, int dim,
                         const half *output_grad, const half *input, const int64_t *target,
                         const half *weight, half *output, const uint32_t &device_id,
                         cudaStream_t cuda_stream) {
  dim3 blocks1(nframe);
  dim3 threads1(MULTIMARGIN_THREADS);
  bool reduce = false;
  bool sizeAverage = false;
  if (reduction == 1) {
    reduce = true;
    sizeAverage = true;
  } else if (reduction == 0) {
    reduce = true;
    sizeAverage = false;
  }
  if (p == 1)
    MultiMarginLoss_backward_kernel_half<1><<<blocks1, threads1, 0, cuda_stream>>>(
      output, output_grad, input, target, weight, nframe, dim, sizeAverage, (half)margin, reduce);
  else
    MultiMarginLoss_backward_kernel_half<2><<<blocks1, threads1, 0, cuda_stream>>>(
      output, output_grad, input, target, weight, nframe, dim, sizeAverage, (half)margin, reduce);
  return;
}

template CUDA_LIB_EXPORT void MultiMarginLossGrad<float>(int64_t p, float margin, int64_t reduction, int nframe,
                                                         int dim, const float *output_grad, const float *input,
                                                         const int64_t *target, const float *weight, float *output,
                                                         const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MultiMarginLossGrad<double>(int64_t p, float margin, int64_t reduction, int nframe,
                                                          int dim, const double *output_grad, const double *input,
                                                          const int64_t *target, const double *weight, double *output,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MultiMarginLossGrad<half>(int64_t p, float margin, int64_t reduction, int nframe,
                                                        int dim, const half *output_grad, const half *input,
                                                        const int64_t *target, const half *weight, half *output,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);
