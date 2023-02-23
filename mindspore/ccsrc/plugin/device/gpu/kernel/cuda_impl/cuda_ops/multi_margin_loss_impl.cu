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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/multi_margin_loss_impl.cuh"
#include "include/cuda_runtime.h"
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

constexpr int MULTIMARGIN_THREADS = 128;
constexpr int MULTIMARGIN_REDUCE_THREADS = 1024;
constexpr int MULTIMARGIN_REDUCE_THREADS1 = 512;
constexpr int MULTIMARGIN_REDUCE_THREADS2 = 256;
constexpr int MULTIMARGIN_REDUCE_THREADS3 = 128;
constexpr int MULTIMARGIN_REDUCE_THREADS4 = 64;
constexpr int MULTIMARGIN_REDUCE_THREADS5 = 32;
constexpr int MULTIMARGIN_REDUCE_THREADS6 = 16;
constexpr int MULTIMARGIN_REDUCE_THREADS7 = 8;
constexpr int MULTIMARGIN_REDUCE_THREADS8 = 4;
constexpr int MULTIMARGIN_REDUCE_THREADS9 = 2;

template <int P>
__global__ void MultiMarginLoss_forward_kernel_half(half *output, const half *input, const int64_t *target,
                                                    const half *weights, int nframe, int dim, bool sizeAverage,
                                                    half margin) {
  __shared__ double buffer[MULTIMARGIN_THREADS];
  int k = blockIdx.x;
  const half *input_k = input + k * dim;
  half *output_k = output + k;
  CUDA_KERNEL_ASSERT(target[k] >= 0 && target[k] < dim);
  int target_k = static_cast<int>(target[k]);
  half input_target_k = input_k[target_k];

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
      double h = (P == 1) ? static_cast<double>(z) : static_cast<double>(z) * static_cast<double>(z);
      if (weights) {
        h *= static_cast<double>(weights[target_k]);
      }
      buffer[threadIdx.x] += static_cast<double>(h);
    }
  }
  __syncthreads();

  // reduce
  if (threadIdx.x == 0) {
    double sum = 0;
    for (int i = 0; i < blockDim.x; i++) sum += static_cast<double>(buffer[i]);

    const int denom = sizeAverage ? nframe * dim : dim;
    *output_k = static_cast<double>(static_cast<double>(sum) / static_cast<double>(denom));
  }
}

__global__ void MultiMarginLossReduceKernel_half(int dim, half *output) {
  __shared__ double buffer[MULTIMARGIN_REDUCE_THREADS];
  __shared__ double buffer1[MULTIMARGIN_REDUCE_THREADS1];
  __shared__ double buffer2[MULTIMARGIN_REDUCE_THREADS2];
  __shared__ double buffer3[MULTIMARGIN_REDUCE_THREADS3];
  __shared__ double buffer4[MULTIMARGIN_REDUCE_THREADS4];
  __shared__ double buffer5[MULTIMARGIN_REDUCE_THREADS5];
  __shared__ double buffer6[MULTIMARGIN_REDUCE_THREADS6];
  __shared__ double buffer7[MULTIMARGIN_REDUCE_THREADS7];
  __shared__ double buffer8[MULTIMARGIN_REDUCE_THREADS8];
  __shared__ double buffer9[MULTIMARGIN_REDUCE_THREADS9];
  int i_start = threadIdx.x;
  int i_end = dim;
  int i_step = blockDim.x;

  buffer[threadIdx.x] = 0;
  for (int i = i_start; i < i_end; i += i_step) {
    buffer[threadIdx.x] += static_cast<double>(output[i]);
  }

  __syncthreads();

  // reduce
  if (threadIdx.x < MULTIMARGIN_REDUCE_THREADS1) {
    buffer1[threadIdx.x] = buffer[threadIdx.x * 2] + buffer[threadIdx.x * 2 + 1];
  }

  __syncthreads();

  if (threadIdx.x < MULTIMARGIN_REDUCE_THREADS2) {
    buffer2[threadIdx.x] = buffer1[threadIdx.x * 2] + buffer1[threadIdx.x * 2 + 1];
  }

  __syncthreads();

  if (threadIdx.x < MULTIMARGIN_REDUCE_THREADS3) {
    buffer3[threadIdx.x] = buffer2[threadIdx.x * 2] + buffer2[threadIdx.x * 2 + 1];
  }

  __syncthreads();

  if (threadIdx.x < MULTIMARGIN_REDUCE_THREADS4) {
    buffer4[threadIdx.x] = buffer3[threadIdx.x * 2] + buffer3[threadIdx.x * 2 + 1];
  }

  __syncthreads();

  if (threadIdx.x < MULTIMARGIN_REDUCE_THREADS5) {
    buffer5[threadIdx.x] = buffer4[threadIdx.x * 2] + buffer4[threadIdx.x * 2 + 1];
  }

  __syncthreads();

  if (threadIdx.x < MULTIMARGIN_REDUCE_THREADS6) {
    buffer6[threadIdx.x] = buffer5[threadIdx.x * 2] + buffer5[threadIdx.x * 2 + 1];
  }

  __syncthreads();

  if (threadIdx.x < MULTIMARGIN_REDUCE_THREADS7) {
    buffer7[threadIdx.x] = buffer6[threadIdx.x * 2] + buffer6[threadIdx.x * 2 + 1];
  }

  __syncthreads();

  if (threadIdx.x < MULTIMARGIN_REDUCE_THREADS8) {
    buffer8[threadIdx.x] = buffer7[threadIdx.x * 2] + buffer7[threadIdx.x * 2 + 1];
  }

  __syncthreads();

  if (threadIdx.x < MULTIMARGIN_REDUCE_THREADS9) {
    buffer9[threadIdx.x] = buffer8[threadIdx.x * 2] + buffer8[threadIdx.x * 2 + 1];
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    *output = buffer9[0] + buffer9[1];
  }
}

template <int P, typename scalar_t>
__global__ void MultiMarginLoss_forward_kernel(scalar_t *output, const scalar_t *input, const int64_t *target,
                                               const scalar_t *weights, int nframe, int dim, bool sizeAverage,
                                               scalar_t margin) {
  __shared__ scalar_t buffer[MULTIMARGIN_THREADS];
  int k = blockIdx.x;
  const scalar_t *input_k = input + k * dim;
  scalar_t *output_k = output + k;
  CUDA_KERNEL_ASSERT(target[k] >= 0 && target[k] < dim);
  int target_k = static_cast<int>(target[k]);
  scalar_t input_target_k = input_k[target_k];

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
      scalar_t h = (P == 1) ? static_cast<scalar_t>(z) : static_cast<scalar_t>(z) * static_cast<scalar_t>(z);
      if (weights) {
        h *= static_cast<scalar_t>(weights[target_k]);
      }
      buffer[threadIdx.x] += static_cast<scalar_t>(h);
    }
  }

  __syncthreads();

  // reduce
  if (threadIdx.x == 0) {
    scalar_t sum = 0;
    for (int i = 0; i < blockDim.x; i++) sum += static_cast<scalar_t>(buffer[i]);

    const int denom = sizeAverage ? nframe * dim : dim;
    *output_k = static_cast<scalar_t>(static_cast<scalar_t>(sum) / static_cast<scalar_t>(denom));
  }
}

template <typename T>
__global__ void MultiMarginLossReduceKernel(int dim, T *output) {
  __shared__ T buffer[MULTIMARGIN_REDUCE_THREADS];
  __shared__ T buffer1[MULTIMARGIN_REDUCE_THREADS1];
  __shared__ T buffer2[MULTIMARGIN_REDUCE_THREADS2];
  __shared__ T buffer3[MULTIMARGIN_REDUCE_THREADS3];
  __shared__ T buffer4[MULTIMARGIN_REDUCE_THREADS4];
  __shared__ T buffer5[MULTIMARGIN_REDUCE_THREADS5];
  __shared__ T buffer6[MULTIMARGIN_REDUCE_THREADS6];
  __shared__ T buffer7[MULTIMARGIN_REDUCE_THREADS7];
  __shared__ T buffer8[MULTIMARGIN_REDUCE_THREADS8];
  __shared__ T buffer9[MULTIMARGIN_REDUCE_THREADS9];
  int i_start = threadIdx.x;
  int i_end = dim;
  int i_step = blockDim.x;

  buffer[threadIdx.x] = 0;
  for (int i = i_start; i < i_end; i += i_step) {
    buffer[threadIdx.x] += static_cast<T>(output[i]);
  }

  __syncthreads();

  // reduce
  if (threadIdx.x < MULTIMARGIN_REDUCE_THREADS1) {
    buffer1[threadIdx.x] = buffer[threadIdx.x * 2] + buffer[threadIdx.x * 2 + 1];
  }

  __syncthreads();

  if (threadIdx.x < MULTIMARGIN_REDUCE_THREADS2) {
    buffer2[threadIdx.x] = buffer1[threadIdx.x * 2] + buffer1[threadIdx.x * 2 + 1];
  }

  __syncthreads();

  if (threadIdx.x < MULTIMARGIN_REDUCE_THREADS3) {
    buffer3[threadIdx.x] = buffer2[threadIdx.x * 2] + buffer2[threadIdx.x * 2 + 1];
  }

  __syncthreads();

  if (threadIdx.x < MULTIMARGIN_REDUCE_THREADS4) {
    buffer4[threadIdx.x] = buffer3[threadIdx.x * 2] + buffer3[threadIdx.x * 2 + 1];
  }

  __syncthreads();

  if (threadIdx.x < MULTIMARGIN_REDUCE_THREADS5) {
    buffer5[threadIdx.x] = buffer4[threadIdx.x * 2] + buffer4[threadIdx.x * 2 + 1];
  }

  __syncthreads();

  if (threadIdx.x < MULTIMARGIN_REDUCE_THREADS6) {
    buffer6[threadIdx.x] = buffer5[threadIdx.x * 2] + buffer5[threadIdx.x * 2 + 1];
  }

  __syncthreads();

  if (threadIdx.x < MULTIMARGIN_REDUCE_THREADS7) {
    buffer7[threadIdx.x] = buffer6[threadIdx.x * 2] + buffer6[threadIdx.x * 2 + 1];
  }

  __syncthreads();

  if (threadIdx.x < MULTIMARGIN_REDUCE_THREADS8) {
    buffer8[threadIdx.x] = buffer7[threadIdx.x * 2] + buffer7[threadIdx.x * 2 + 1];
  }

  __syncthreads();

  if (threadIdx.x < MULTIMARGIN_REDUCE_THREADS9) {
    buffer9[threadIdx.x] = buffer8[threadIdx.x * 2] + buffer8[threadIdx.x * 2 + 1];
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    *output = buffer9[0] + buffer9[1];
  }
}

// namespace str
template <typename T>
void MultiMarginLoss(int64_t p, float margin, int64_t reduction, int nframe, int dim, const T *input,
                     const int64_t *target, const T *weight, T *output, const uint32_t &device_id,
                     cudaStream_t cuda_stream) {
  dim3 blocks(nframe);
  dim3 threads(MULTIMARGIN_THREADS);
  bool sizeAverage = false;
  if (reduction == 1) {
    sizeAverage = true;
  }
  if (p == 1)
    MultiMarginLoss_forward_kernel<1>
      <<<blocks, threads, 0, cuda_stream>>>(output, input, target, weight, nframe, dim, sizeAverage, (T)margin);
  else
    MultiMarginLoss_forward_kernel<2>
      <<<blocks, threads, 0, cuda_stream>>>(output, input, target, weight, nframe, dim, sizeAverage, (T)margin);
  if (reduction != 2) {
    dim3 reduce_blocks(1);
    dim3 reduce_threads(MULTIMARGIN_REDUCE_THREADS);
    int reduce_dim = nframe;
    MultiMarginLossReduceKernel<<<reduce_blocks, reduce_threads, 0, cuda_stream>>>(reduce_dim, output);
  }
}

// namespace str
template <>
void MultiMarginLoss(int64_t p, float margin, int64_t reduction, int nframe, int dim, const half *input,
                     const int64_t *target, const half *weight, half *output, const uint32_t &device_id,
                     cudaStream_t cuda_stream) {
  dim3 blocks(nframe);
  dim3 threads(128);
  bool sizeAverage = false;
  if (reduction == 1) {
    sizeAverage = true;
  }
  if (p == 1)
    MultiMarginLoss_forward_kernel_half<1>
      <<<blocks, threads, 0, cuda_stream>>>(output, input, target, weight, nframe, dim, sizeAverage, (half)margin);
  else
    MultiMarginLoss_forward_kernel_half<2>
      <<<blocks, threads, 0, cuda_stream>>>(output, input, target, weight, nframe, dim, sizeAverage, (half)margin);
  if (reduction != 2) {
    dim3 reduce_blocks(1);
    dim3 reduce_threads(MULTIMARGIN_REDUCE_THREADS);
    int reduce_dim = nframe;
    MultiMarginLossReduceKernel_half<<<reduce_blocks, reduce_threads, 0, cuda_stream>>>(reduce_dim, output);
  }
}

template CUDA_LIB_EXPORT void MultiMarginLoss<float>(int64_t p, float margin, int64_t reduction, int nframe, int dim,
                                                     const float *input, const int64_t *target, const float *weight,
                                                     float *output, const uint32_t &device_id,
                                                     cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MultiMarginLoss<double>(int64_t p, float margin, int64_t reduction, int nframe, int dim,
                                                      const double *input, const int64_t *target, const double *weight,
                                                      double *output, const uint32_t &device_id,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MultiMarginLoss<half>(int64_t p, float margin, int64_t reduction, int nframe, int dim,
                                                    const half *input, const int64_t *target, const half *weight,
                                                    half *output, const uint32_t &device_id,
                                                    cudaStream_t cuda_stream);
