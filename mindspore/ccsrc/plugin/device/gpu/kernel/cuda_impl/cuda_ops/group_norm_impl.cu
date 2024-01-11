/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/group_norm_impl.cuh"

constexpr int NUM_PER_THREAD_REDUCE = 4;
constexpr int WARP_SIZE = 32;

template <typename T>
inline __device__ T general_sqrt(T val) {
  return (T)sqrt(static_cast<float>(val));
}

template <>
inline __device__ half general_sqrt(half val) {
  return hsqrt(val);
}

template <typename T>
inline __device__ void MeanAndVarAccumulation(T *mean, T *var, T *count, const T &val) {
  // Welford Algorithm:
  // \mu_k = \mu_{k-1} + (x_k - \mu_{k-1})/k
  // \sigma_k^2 = \sigma_{k-1}^2 + (x_k - \mu_{k-1}) * (x_k - \mu_k)
  count[0]++;
  T mean_new = mean[0] + (val - mean[0]) / count[0];
  var[0] = var[0] + (val - mean[0]) * (val - mean_new);
  mean[0] = mean_new;
}

template <typename T>
inline __device__ void MeanAndVarMerge(T *mean1, T *var1, T *count1, const T &mean2, const T &var2, const T &count2) {
  T zero = 0;
  if (count2 == zero) {
    return;
  }

  T count = count1[0] + count2;
  var1[0] = var1[0] + var2 + (mean1[0] - mean2) * (mean1[0] - mean2) * count1[0] * count2 / count;
  mean1[0] = (count1[0] * mean1[0] + count2 * mean2) / count;
  count1[0] = count;
}

template <typename T>
inline __device__ void ThreadReduce(const int col_dim, const T *block_addr, float *mean, float *var, float *count) {
  int loop_num = (col_dim + NUM_PER_THREAD_REDUCE - 1) / NUM_PER_THREAD_REDUCE;
  for (int i = threadIdx.x; i < loop_num; i += blockDim.x) {
    for (int j = 0; j < NUM_PER_THREAD_REDUCE; j++) {
      int pos = NUM_PER_THREAD_REDUCE * i + j;
      if (pos >= col_dim) {
        return;
      }
      MeanAndVarAccumulation(mean, var, count, static_cast<float>(block_addr[pos]));
    }
  }
}

template <typename T>
inline __device__ void WarpReduce(T *mean, T *var, T *count) {
  for (int delta = (WARP_SIZE >> 1); delta > 0; delta >>= 1) {
    T mean_other = __shfl_down_sync(0xffffffff, mean[0], delta);
    T var_other = __shfl_down_sync(0xffffffff, var[0], delta);
    T count_other = __shfl_down_sync(0xffffffff, count[0], delta);
    MeanAndVarMerge(mean, var, count, mean_other, var_other, count_other);
  }
}

template <typename T>
inline __device__ void BlockReduce(const int col_dim, float *mean, float *var, float *count, T *mean_addr,
                                   T *rstd_addr, float *share_mem, const float epsilon) {
  // load data to share memory
  // thread(0, 32, 64, 96, ...) keep the data
  if (threadIdx.x % WARP_SIZE == 0) {
    int offset = threadIdx.x / WARP_SIZE * 3;
    share_mem[offset] = mean[0];
    share_mem[offset + 1] = var[0];
    share_mem[offset + 2] = count[0];
  }
  __syncthreads();

  for (int stride = blockDim.x / WARP_SIZE / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      int offset = (threadIdx.x + stride) * 3;
      MeanAndVarMerge(&share_mem[threadIdx.x * 3], &share_mem[threadIdx.x * 3 + 1], &share_mem[threadIdx.x * 3 + 2],
                      share_mem[offset], share_mem[offset + 1], share_mem[offset + 2]);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    mean_addr[blockIdx.x] = static_cast<T>(share_mem[0]);
    share_mem[1] = 1.0 / general_sqrt((share_mem[1] / col_dim + epsilon));
    rstd_addr[blockIdx.x] = static_cast<T>(share_mem[1]);
  }
}

template <typename T>
inline __device__ void GroupNorm(const int row, const int col_dim, const int num_channel, const int HxW, const T *x,
                                 const float *share_mem, const T *gamma, const T *beta, T *y) {
  for (int col = threadIdx.x; col < col_dim; col += blockDim.x) {
    int pos = row * col_dim + col;
    int i = (pos / HxW) % num_channel;
    float tmp_y = (static_cast<float>(x[pos]) - share_mem[0]) * share_mem[1] *
                   static_cast<float>(gamma[i]) + static_cast<float>(beta[i]);
    y[pos] = (T)(tmp_y);
  }
}

template <typename T>
__global__ void GroupNormKernel(const int row_dim, const int col_dim, const int num_channel, const int HxW,
                                const float epsilon, const T *x, const T *gamma, const T *beta, T *y,
                                T *mean_addr, T *rstd_addr) {
  for (auto row = blockIdx.x; row < row_dim; row += gridDim.x) {
    float mean = 0;
    float var = 0;
    float count = 0;
    const T *block_addr = x + row * col_dim;
    DynamicSharedMem<float> share_mem;

    ThreadReduce(col_dim, block_addr, &mean, &var, &count);
    WarpReduce(&mean, &var, &count);
    BlockReduce<T>(col_dim, &mean, &var, &count, mean_addr, rstd_addr, share_mem.addr(), epsilon);

    __syncthreads();
    GroupNorm(row, col_dim, num_channel, HxW, x, share_mem.addr(), gamma, beta, y);
  }
}

template <typename T>
cudaError_t GroupNorm(const int row_dim, const int col_dim, const int num_channel, const int HxW, const float epsilon,
                      const T *x, const T *gamma, const T *beta, T *y, T *mean, T *rstd, cudaStream_t stream) {
  const int thread_per_block = 256;
  // keep the mean/var/count after warp reduce
  int share_mem_size = thread_per_block / WARP_SIZE * 3 * sizeof(float);
  GroupNormKernel<<<row_dim, thread_per_block, share_mem_size, stream>>>(row_dim, col_dim, num_channel, HxW, epsilon, x,
                                                                        gamma, beta, y, mean, rstd);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t GroupNorm(const int row_dim, const int col_dim, const int num_channel,
                                               const int HxW, const float epsilon, const float *x, const float *gamma,
                                               const float *beta, float *y, float *mean, float *rstd,
                                               cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GroupNorm(const int row_dim, const int col_dim, const int num_channel,
                                               const int HxW, const float epsilon, const half *x, const half *gamma,
                                               const half *beta, half *y, half *mean, half *rstd, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GroupNorm(const int row_dim, const int col_dim, const int num_channel,
                                               const int HxW, const float epsilon, const double *x, const double *gamma,
                                               const double *beta, double *y, double *mean, double *rstd,
                                               cudaStream_t stream);
