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

#include <stdint.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/system/cuda/execution_policy.h>
#include "batchnorm_grad_impl.cuh"
#include "include/cuda_runtime.h"
#include "include/cuda_fp16.h"

const int kWarpSize = 32;
const int kBlockSize = 1024;
const int kNumWarps = 32;

template <typename T>
__global__ void BatchNormGradKernel(T *x_input, T *dy, float *scale, float *save_mean, float *save_variance, T *dx,
                                    float *bn_scale, float *bn_bias, double epsilon, int N, int C, int H, int W) {
  __shared__ T shared_dy[kNumWarps];
  __shared__ T shared_p[kNumWarps];
  int warpId = threadIdx.x / kWarpSize;
  int laneId = threadIdx.x % kWarpSize;

  int plane = blockIdx.x;
  int plane_size = N * H * W;

  T invstd = static_cast<T>(1) / static_cast<T>(sqrt(save_variance[plane] + epsilon));
  T scale_val = scale != nullptr ? static_cast<T>(scale[plane]) : static_cast<T>(1);
  T grad_scale = invstd * scale_val;

  T mean = static_cast<T>(save_mean[plane]);
  T dy_sum = static_cast<T>(0);
  T dot_p = static_cast<T>(0);

  if (threadIdx.x < kNumWarps) {
    shared_dy[threadIdx.x] = static_cast<T>(0);
    shared_p[threadIdx.x] = static_cast<T>(0);
  }
  __syncthreads();

  // Compute three values across (Batch, Height, Width) in one pass:
  // 1. dx
  // 2. Sum(dy)
  // 3. DotProduct(x - mean, dy)
  for (int x = threadIdx.x; x < plane_size; x += blockDim.x) {
    int index = (x / (H * W) * C * H * W) + (plane * H * W) + (x % (H * W));
    dx[index] = static_cast<T>(dy[index] * grad_scale);
    dy_sum += dy[index];
    dot_p += (x_input[index] - mean) * dy[index];
  }
  __syncthreads();

  // Warp reduction
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    T other_dy = __shfl_down_sync(0xffffffff, dy_sum, offset);
    T other_p = __shfl_down_sync(0xffffffff, dot_p, offset);
    dy_sum += other_dy;
    dot_p += other_p;
  }
  __syncwarp();

  // Move warp-reduction result to shared memory
  if (laneId == 0) {
    shared_dy[warpId] = dy_sum;
    shared_p[warpId] = dot_p;
  }
  __syncthreads();

  // Shared memory reduction
  // There are exactly 32 items in shared memory, can be reduced within one warp.
  if (warpId == 0) {
    dy_sum = shared_dy[laneId];
    dot_p = shared_p[laneId];
    __syncwarp();
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
      T other_dy = __shfl_down_sync(0xffffffff, dy_sum, offset);
      T other_p = __shfl_down_sync(0xffffffff, dot_p, offset);
      dy_sum += other_dy;
      dot_p += other_p;
    }
    __syncwarp();
  }

  // Compute bn_scale & bn_bias
  if (threadIdx.x == 0) {
    bn_scale[plane] = static_cast<T>(dot_p * invstd);
  }

  if (threadIdx.x == 0) {
    bn_bias[plane] = static_cast<T>(dy_sum);
  }
}

template <typename T>
cudaError_t CalBatchNormGrad(T *x, T *dy, float *scale, float *save_mean, float *save_variance, T *dx, float *bn_scale,
                             float *bn_bias, double epsilon, int N, int C, int H, int W, cudaStream_t cuda_stream) {
  BatchNormGradKernel<<<C, kBlockSize, 0, cuda_stream>>>(x, dy, scale, save_mean, save_variance, dx, bn_scale, bn_bias,
                                                         epsilon, N, C, H, W);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalBatchNormGrad<float>(float *x, float *dy, float *scale, float *save_mean,
                                                             float *save_variance, float *dx, float *bn_scale,
                                                             float *bn_bias, double epsilon, int N, int C, int H, int W,
                                                             cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalBatchNormGrad<half>(half *x, half *dy, float *scale, float *save_mean,
                                                            float *save_variance, half *dx, float *bn_scale,
                                                            float *bn_bias, double epsilon, int N, int C, int H, int W,
                                                            cudaStream_t cuda_stream);
