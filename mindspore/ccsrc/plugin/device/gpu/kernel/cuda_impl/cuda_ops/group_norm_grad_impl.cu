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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/group_norm_grad_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/group_norm_impl.cuh"
#include "include/cuda_fp16.h"

constexpr int NUM_PER_THREAD_REDUCE = 4;
constexpr int WARP_SIZE = 32;

template <typename T>
inline __device__ T my_pow(T a, double b) {
  return pow(a, static_cast<float>(b));
}

template <>
inline __device__ half my_pow(half a, double b) {
  return __float2half(pow(__half2float(a), static_cast<float>(b)));
}

template <typename T>
inline __device__ void WarpReduce(T *x1, T *x2) {
  for (int delta = (WARP_SIZE >> 1); delta > 0; delta >>= 1) {
    x1[0] += __shfl_down_sync(0xffffffff, x1[0], delta);
    x2[0] += __shfl_down_sync(0xffffffff, x2[0], delta);
  }
}

template <typename T>
inline __device__ void BlockReduce(const int col, float *x1, float *x2, T *x1_addr, T *x2_addr) {
  // load data to share memory
  // thread(0, 32, 64, 96, ...) keep the data
  DynamicSharedMem<float> share_mem;
  if (threadIdx.x % WARP_SIZE == 0) {
    int offset = threadIdx.x / WARP_SIZE * 2;
    share_mem.addr()[offset] = x1[0];
    share_mem.addr()[offset + 1] = x2[0];
  }
  __syncthreads();

  for (int stride = blockDim.x / WARP_SIZE / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      int offset = (threadIdx.x + stride) * 2;
      share_mem.addr()[threadIdx.x * 2] += share_mem.addr()[offset];
      share_mem.addr()[threadIdx.x * 2 + 1] += share_mem.addr()[offset + 1];
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    x1_addr[col] = (T)(share_mem.addr()[0]);
    x2_addr[col] = (T)(share_mem.addr()[1]);
  }
  __syncthreads();
}

template <typename T>
inline __device__ void DsAndDbThreadReduce(const int col, const int row_dim, const int col_dim,
                                           const T *dy, const T *x, float *dscale, float *dbias) {
  int loop_num = (row_dim + NUM_PER_THREAD_REDUCE - 1) / NUM_PER_THREAD_REDUCE;
  for (int i = threadIdx.x; i < loop_num; i += blockDim.x) {
    for (int j = 0; j < NUM_PER_THREAD_REDUCE; j++) {
      int row = NUM_PER_THREAD_REDUCE * i + j;
      if (row >= row_dim) {
        return;
      }

      int pos = col * row_dim + row;
      dscale[0] += static_cast<float>(dy[pos]) * static_cast<float>(x[pos]);
      dbias[0] += static_cast<float>(dy[pos]);
    }
  }
}

template <typename T>
__global__ void CalDsAndDbKernel(const int row_dim, const int col_dim, const T *dy, const T *x,
                                 float *dscale_addr, float *dbias_addr) {
  for (int col = blockIdx.x; col < col_dim; col += gridDim.x) {
    float dscale = 0;
    float dbias = 0;
    DsAndDbThreadReduce(col, row_dim, col_dim, dy, x, &dscale, &dbias);
    WarpReduce(&dscale, &dbias);
    BlockReduce<float>(col, &dscale, &dbias, dscale_addr, dbias_addr);
  }
}

template <typename T>
inline __device__ void GammaAndBetaThreadReduce(const int col, const int batch, const int num_channel,
                                                const int num_groups, const float *dscale, const float *dbias,
                                                const T *mean, const T *rstd, float *dg, float *db) {
  int loop_num = (batch + NUM_PER_THREAD_REDUCE - 1) / NUM_PER_THREAD_REDUCE;
  for (int i = threadIdx.x; i < loop_num; i += blockDim.x) {
    for (int j = 0; j < NUM_PER_THREAD_REDUCE; j++) {
      int row = NUM_PER_THREAD_REDUCE * i + j;
      if (row >= batch) {
        return;
      }

      int idx1 = row * num_channel + col;
      int idx2 = idx1 * num_groups / num_channel;
      dg[0] += (dscale[idx1] - dbias[idx1] * static_cast<float>(mean[idx2])) * static_cast<float>(rstd[idx2]);
      db[0] += dbias[idx1];
    }
  }
}

template <typename T>
__global__ void GammaAndBetaPropKernel(const int batch, const int num_channel, const int num_groups,
                                       const float *dscale, const float *dbias, const T *mean_addr,
                                       const T *rstd_addr, T *dg_addr, T *db_addr) {
  for (int col = blockIdx.x; col < num_channel; col += gridDim.x) {
    float dg = 0;
    float db = 0;
    GammaAndBetaThreadReduce(col, batch, num_channel, num_groups, dscale, dbias, mean_addr, rstd_addr, &dg, &db);
    WarpReduce(&dg, &db);
    BlockReduce(col, &dg, &db, dg_addr, db_addr);
  }
}

template <typename T>
inline __device__ void InputThreadReduce(const int row, const int col_dim, const int num_channel, const int HxW,
                                         float *sum1, float *sum2, float *sum3, const T *dy, const T *x,
                                         const T *mean, const T *rstd, const T *gamma) {
  int loop_num = (col_dim + NUM_PER_THREAD_REDUCE - 1) / NUM_PER_THREAD_REDUCE;
  for (int i = threadIdx.x; i < loop_num; i += blockDim.x) {
    for (int j = 0; j < NUM_PER_THREAD_REDUCE; j++) {
      int col = NUM_PER_THREAD_REDUCE * i + j;
      if (col >= col_dim) {
        sum1[0] = -0.5 * sum1[0] * my_pow(static_cast<float>(rstd[row]), 3.0);
        sum3[0] = -2.0 * sum3[0];
        return;
      }

      int pos = row * col_dim + col;
      int gamma_offset = (pos / HxW) % num_channel;
      float v1 = static_cast<float>(dy[pos] * gamma[gamma_offset]);
      float v2 = static_cast<float>(x[pos]) - static_cast<float>(mean[row]);

      sum1[0] += v1 * v2;
      sum2[0] += v1;
      sum3[0] += v2;
    }
  }
  sum1[0] = -0.5 * sum1[0] * my_pow(static_cast<float>(rstd[row]), 3.0);
  sum3[0] = -2.0 * sum3[0];
}

template <typename T>
inline __device__ void InputWarpReduce(T *sum1, T *sum2, T *sum3) {
  for (int delta = (WARP_SIZE >> 1); delta > 0; delta >>= 1) {
    sum1[0] += __shfl_down_sync(0xffffffff, sum1[0], delta);
    sum2[0] += __shfl_down_sync(0xffffffff, sum2[0], delta);
    sum3[0] += __shfl_down_sync(0xffffffff, sum3[0], delta);
  }
}

template <typename T>
inline __device__ void InputBlockReduce(const int col_dim, T *sum1, T *sum2, T *sum3, T *share_mem) {
  // load data to share memory
  // thread(0, 32, 64, 96, ...) keep the data
  if (threadIdx.x % WARP_SIZE == 0) {
    int offset = threadIdx.x / WARP_SIZE * 3;
    share_mem[offset] = sum1[0];
    share_mem[offset + 1] = sum2[0];
    share_mem[offset + 2] = sum3[0];
  }
  __syncthreads();

  for (int stride = blockDim.x / WARP_SIZE / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      int offset = (threadIdx.x + stride) * 3;
      share_mem[threadIdx.x * 3] += share_mem[offset];
      share_mem[threadIdx.x * 3 + 1] += share_mem[offset + 1];
      share_mem[threadIdx.x * 3 + 2] += share_mem[offset + 2];
    }
  }
  __syncthreads();
}

template <typename T>
inline __device__ void InputProp(const int row, const int col_dim, const int num_channel, const int HxW, const T *dy,
                                 const T *x, const T *mean, const T *rstd, const T *gamma, T *dx,
                                 const float *share_mem) {
  float v3 = static_cast<float>(rstd[row]);
  float v4 = share_mem[0] * (2.0 / col_dim);
  float v5 = (-1.0 * v3 * share_mem[1] + (1.0 / col_dim) * share_mem[0] * share_mem[2]) * (1.0 / col_dim);
  for (int col = threadIdx.x; col < col_dim; col += blockDim.x) {
    int pos = (row * col_dim + col);
    int gamma_offset = (pos / HxW) % num_channel;
    float v1 = static_cast<float>(dy[pos] * gamma[gamma_offset]);
    float v2 = static_cast<float>(x[pos]) - static_cast<float>(mean[row]);
    dx[pos] = (T)(v1 * v3 + v4 * v2 + v5);
  }
}

template <typename T>
__global__ void InputPropKernel(const int row_dim, const int col_dim, const int num_channel, const int HxW, const T *dy,
                                const T *x, const T *mean, const T *rstd, const T *gamma, T *dx) {
  for (int row = blockIdx.x; row < row_dim; row += gridDim.x) {
    float sum1 = 0;
    float sum2 = 0;
    float sum3 = 0;
    DynamicSharedMem<float> share_mem;
    InputThreadReduce(row, col_dim, num_channel, HxW, &sum1, &sum2, &sum3, dy, x, mean, rstd, gamma);
    InputWarpReduce(&sum1, &sum2, &sum3);
    InputBlockReduce(col_dim, &sum1, &sum2, &sum3, share_mem.addr());
    InputProp(row, col_dim, num_channel, HxW, dy, x, mean, rstd, gamma, dx, share_mem.addr());
  }
}

template <typename T>
cudaError_t GroupNormGrad(const int batch, const int num_channel, const int HxW, const int num_groups, const T *dy,
                              const T *x, const T *mean, const T *rstd, const T *gamma, T *dx, T *dg, T *db,
                              float *dscale, float *dbias, cudaStream_t stream) {
  const int thread_per_block = 256;
  int share_mem_size = thread_per_block / WARP_SIZE * 3 * sizeof(float);
  int row_dim = batch * num_groups;
  int col_dim = num_channel * HxW / num_groups;
  int dsdb_dim = batch * num_channel;

  InputPropKernel<<<row_dim, thread_per_block, share_mem_size, stream>>>(row_dim, col_dim, num_channel, HxW, dy, x,
                                                                         mean, rstd, gamma, dx);
  share_mem_size = thread_per_block / WARP_SIZE * 2 * sizeof(float);
  CalDsAndDbKernel<<<dsdb_dim, thread_per_block, share_mem_size, stream>>>(HxW, dsdb_dim, dy, x, dscale, dbias);
  GammaAndBetaPropKernel<<<num_channel, thread_per_block, share_mem_size, stream>>>(
                           batch, num_channel, num_groups, dscale, dbias, mean, rstd, dg, db);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t GroupNormGrad(const int batch, const int num_channel, const int HxW,
                                                   const int num_groups, const float *dy, const float *x,
                                                   const float *mean, const float *rstd, const float *gamma,
                                                   float *dx, float *dg, float *db, float *dscale, float *dbias,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GroupNormGrad(const int batch, const int num_channel, const int HxW,
                                                   const int num_groups, const half *dy, const half *x,
                                                   const half *mean, const half *rstd, const half *gamma,
                                                   half *dx, half *dg, half *db, float *dscale, float *dbias,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GroupNormGrad(const int batch, const int num_channel, const int HxW,
                                                   const int num_groups, const double *dy, const double *x,
                                                   const double *mean, const double *rstd, const double *gamma,
                                                   double *dx, double *dg, double *db, float *dscale, float *dbias,
                                                   cudaStream_t stream);
