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

#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/layer_norm_grad_grad_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/layer_norm_impl.cuh"
#include "include/cuda_fp16.h"

constexpr int THREAD_PER_BLOCK = 256;
constexpr int NUM_PER_THREAD_REDUCE = 4;
constexpr int WARP_SIZE = 32;
constexpr int NUM_SHARED_SUM_INPUT = 7;
constexpr int NUM_SHARED_SUM_GAMMA = 3;

template <typename T>
inline __device__ T my_pow(T a, double b) {
  return pow(a, static_cast<float>(b));
}

template <>
inline __device__ half my_pow(half a, double b) {
  return __float2half(pow(__half2float(a), static_cast<float>(b)));
}

template <typename T>
inline __device__ void GammaAndBetaThreadReduce(const int &col, const int &row_dim, const int &col_dim,
                                                const int &mean_dim, const T &epsilon, const T *dy, const T *x,
                                                const T *mean, const T *var, const T *grad_dx, T *part1, T *part2,
                                                T *part3, const T *global_sum1, const T *global_sum2) {
  int loop_num = (row_dim + NUM_PER_THREAD_REDUCE - 1) / NUM_PER_THREAD_REDUCE;
  for (int i = threadIdx.x; i < loop_num; i += blockDim.x) {
    for (int j = 0; j < NUM_PER_THREAD_REDUCE; j++) {
      int row = NUM_PER_THREAD_REDUCE * i + j;
      if (row >= row_dim) {
        return;
      }

      int pos = row * col_dim + col;
      int mean_offset = pos / mean_dim;

      T v1 = x[pos] - mean[mean_offset];
      T v2 = my_pow(var[mean_offset] + epsilon, -0.5);

      part1[0] += dy[pos] * v1 * v2 * global_sum2[pos];
      part2[0] += dy[pos] * global_sum1[pos];
      part3[0] += dy[pos] * v2 * grad_dx[pos];
    }
  }
}

template <typename T>
inline __device__ void GammaAndBetaWarpReduce(T *part1, T *part2, T *part3) {
  for (int delta = (WARP_SIZE >> 1); delta > 0; delta >>= 1) {
    part1[0] += __shfl_down_sync(0xffffffff, part1[0], delta);
    part2[0] += __shfl_down_sync(0xffffffff, part2[0], delta);
    part3[0] += __shfl_down_sync(0xffffffff, part3[0], delta);
  }
}

template <typename T>
inline __device__ void GammaAndBetaBlockReduce(const int &col, const int &row_dim, T *part1, T *part2, T *part3,
                                               T *d_gamma) {
  // load data to share memory
  // thread(0, 32, 64, 96, ...) keep the data
  DynamicSharedMem<T> share_mem;
  if (threadIdx.x % WARP_SIZE == 0) {
    int offset = threadIdx.x / WARP_SIZE * NUM_SHARED_SUM_GAMMA;
    share_mem.addr()[offset] = part1[0];
    share_mem.addr()[offset + 1] = part2[0];
    share_mem.addr()[offset + 2] = part3[0];
  }
  __syncthreads();

  for (int stride = blockDim.x / WARP_SIZE / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      int offset = (threadIdx.x + stride) * NUM_SHARED_SUM_GAMMA;
      share_mem.addr()[threadIdx.x * NUM_SHARED_SUM_GAMMA] += share_mem.addr()[offset];
      share_mem.addr()[threadIdx.x * NUM_SHARED_SUM_GAMMA + 1] += share_mem.addr()[offset + 1];
      share_mem.addr()[threadIdx.x * NUM_SHARED_SUM_GAMMA + 2] += share_mem.addr()[offset + 2];
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    d_gamma[col] = share_mem.addr()[0] + share_mem.addr()[1] + share_mem.addr()[2];
  }
}

template <typename T>
__global__ void GammaAndBetaPropKernel(const int row_dim, const int col_dim, const int mean_dim, const T epsilon,
                                       const T *dy, const T *x, const T *mean, const T *var, const T *grad_dx,
                                       T *d_gamma, T *global_sum1, T *global_sum2) {
  for (int col = blockIdx.x; col < col_dim; col += gridDim.x) {
    T part1 = 0;
    T part2 = 0;
    T part3 = 0;
    GammaAndBetaThreadReduce(col, row_dim, col_dim, mean_dim, epsilon, dy, x, mean, var, grad_dx, &part1, &part2,
                             &part3, global_sum1, global_sum2);
    GammaAndBetaWarpReduce(&part1, &part2, &part3);
    GammaAndBetaBlockReduce(col, row_dim, &part1, &part2, &part3, d_gamma);
  }
}

template <typename T>
inline __device__ void InputThreadReduceInnerMean(const int &row, const int &col_dim, const int &param_dim,
                                                  const T &epsilon, T *sum1, T *sum2, T *sum3, T *sum4, const T *dy,
                                                  const T *x, const T *mean, const T *var, const T *gamma,
                                                  const T *grad_dx) {
  int loop_num = (col_dim + NUM_PER_THREAD_REDUCE - 1) / NUM_PER_THREAD_REDUCE;
  for (int i = threadIdx.x; i < loop_num; i += blockDim.x) {
    for (int j = 0; j < NUM_PER_THREAD_REDUCE; j++) {
      int col = NUM_PER_THREAD_REDUCE * i + j;
      if (col >= col_dim) {
        return;
      }
      int pos = row * col_dim + col;
      int gamma_offset = pos % param_dim;

      T v1 = x[pos] - mean[row];
      T v2 = my_pow(var[row] + epsilon, -0.5);
      T v3 = v1 * v2;
      T v4 = dy[pos] * gamma[gamma_offset];

      sum1[0] -= v2 * grad_dx[pos];
      sum2[0] -= v3 * v2 * grad_dx[pos];
      sum3[0] += v4;
      sum4[0] += v4 * v3;
    }
  }
}

template <typename T>
inline __device__ void InputWarpReduceInnerMean(T *sum1, T *sum2, T *sum3, T *sum4) {
  for (int delta = (WARP_SIZE >> 1); delta > 0; delta >>= 1) {
    sum1[0] += __shfl_down_sync(0xffffffff, sum1[0], delta);
    sum2[0] += __shfl_down_sync(0xffffffff, sum2[0], delta);
    sum3[0] += __shfl_down_sync(0xffffffff, sum3[0], delta);
    sum4[0] += __shfl_down_sync(0xffffffff, sum4[0], delta);
  }
}

template <typename T>
inline __device__ void InputBlockReduceInnerMean(const int &col_dim, T *sum1, T *sum2, T *sum3, T *sum4, T *share_mem) {
  // load data to share memory
  // thread(0, 32, 64, 96, ...) keep the data
  if (threadIdx.x % WARP_SIZE == 0) {
    int offset = threadIdx.x / WARP_SIZE * NUM_SHARED_SUM_INPUT;
    share_mem[offset] = sum1[0];
    share_mem[offset + 1] = sum2[0];
    share_mem[offset + 2] = sum3[0];
    share_mem[offset + 3] = sum4[0];
  }
  __syncthreads();

  for (int stride = blockDim.x / WARP_SIZE / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      int offset = (threadIdx.x + stride) * NUM_SHARED_SUM_INPUT;

      share_mem[threadIdx.x * NUM_SHARED_SUM_INPUT] += share_mem[offset];
      share_mem[threadIdx.x * NUM_SHARED_SUM_INPUT + 1] += share_mem[offset + 1];
      share_mem[threadIdx.x * NUM_SHARED_SUM_INPUT + 2] += share_mem[offset + 2];
      share_mem[threadIdx.x * NUM_SHARED_SUM_INPUT + 3] += share_mem[offset + 3];
    }
  }
  __syncthreads();
}

template <typename T>
inline __device__ void InputThreadReduceOuterMean(const int &row, const int &col_dim, const int &param_dim,
                                                  const T &epsilon, T *sum5, T *sum6, T *sum7, T *share_mem,
                                                  const T *dy, const T *x, const T *mean, const T *var, const T *gamma,
                                                  const T *grad_dx, const T *grad_dg, T *d_x) {
  int loop_num = (col_dim + NUM_PER_THREAD_REDUCE - 1) / NUM_PER_THREAD_REDUCE;
  for (int i = threadIdx.x; i < loop_num; i += blockDim.x) {
    for (int j = 0; j < NUM_PER_THREAD_REDUCE; j++) {
      int col = NUM_PER_THREAD_REDUCE * i + j;
      if (col >= col_dim) {
        return;
      }
      int pos = row * col_dim + col;
      int gamma_offset = pos % param_dim;

      T v1 = x[pos] - mean[row];
      T v2 = my_pow(var[row] + epsilon, -0.5);
      T v3 = dy[pos] * gamma[gamma_offset];

      T v4 = v3 - share_mem[2] * (1.0 / col_dim) - v1 * v2 * share_mem[3] * (1.0 / col_dim);
      T v5 = v3 * share_mem[1] * (1.0 / col_dim);
      T v6 = grad_dx[pos] * v2 * share_mem[3] * (-1.0 / col_dim);
      T v7 = dy[pos] * grad_dg[gamma_offset];
      T v8 = v5 + v6 + v7;

      T part1 = v4 * grad_dx[pos];
      T part2 = v1 * v8;
      T part3 = v2 * v8;
      d_x[pos] = part3;

      sum5[0] += part1;
      sum6[0] += part2;
      sum7[0] -= part3;
    }
  }
}

template <>
inline __device__ void InputThreadReduceOuterMean(const int &row, const int &col_dim, const int &param_dim,
                                                  const half &epsilon, half *sum5, half *sum6, half *sum7,
                                                  half *share_mem, const half *dy, const half *x, const half *mean,
                                                  const half *var, const half *gamma, const half *grad_dx,
                                                  const half *grad_dg, half *d_x) {
  int loop_num = (col_dim + NUM_PER_THREAD_REDUCE - 1) / NUM_PER_THREAD_REDUCE;
  for (int i = threadIdx.x; i < loop_num; i += blockDim.x) {
    for (int j = 0; j < NUM_PER_THREAD_REDUCE; j++) {
      int col = NUM_PER_THREAD_REDUCE * i + j;
      if (col >= col_dim) {
        return;
      }
      int pos = row * col_dim + col;
      int gamma_offset = pos % param_dim;

      half v1 = x[pos] - mean[row];
      half v2 = my_pow(var[row] + epsilon, -0.5);
      half v3 = dy[pos] * gamma[gamma_offset];
      half v4 = v3 - share_mem[2] * __float2half(1.0 / col_dim) - v1 * v2 * share_mem[3] * __float2half(1.0 / col_dim);
      half v5 = v3 * share_mem[1] * __float2half(1.0 / col_dim);
      half v6 = grad_dx[pos] * v2 * share_mem[3] * __float2half(-1.0 / col_dim);
      half v7 = dy[pos] * grad_dg[gamma_offset];
      half v8 = v5 + v6 + v7;

      half part1 = v4 * grad_dx[pos];
      half part2 = v1 * v8;
      half part3 = v2 * v8;
      d_x[pos] = part3;

      sum5[0] += part1;
      sum6[0] += part2;
      sum7[0] -= part3;
    }
  }
}

template <typename T>
inline __device__ void InputWarpReduceOuterMean(T *sum5, T *sum6, T *sum7) {
  for (int delta = (WARP_SIZE >> 1); delta > 0; delta >>= 1) {
    sum5[0] += __shfl_down_sync(0xffffffff, sum5[0], delta);
    sum6[0] += __shfl_down_sync(0xffffffff, sum6[0], delta);
    sum7[0] += __shfl_down_sync(0xffffffff, sum7[0], delta);
  }
}

template <typename T>
inline __device__ void InputBlockReduceOuterMean(const int &col_dim, T *sum5, T *sum6, T *sum7, T *share_mem) {
  // load data to share memory
  // thread(0, 32, 64, 96, ...) keep the data
  if (threadIdx.x % WARP_SIZE == 0) {
    int offset = threadIdx.x / WARP_SIZE * NUM_SHARED_SUM_INPUT;

    share_mem[offset + 4] = sum5[0];
    share_mem[offset + 5] = sum6[0];
    share_mem[offset + 6] = sum7[0];
  }
  __syncthreads();

  for (int stride = blockDim.x / WARP_SIZE / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      int offset = (threadIdx.x + stride) * NUM_SHARED_SUM_INPUT;

      share_mem[threadIdx.x * NUM_SHARED_SUM_INPUT + 4] += share_mem[offset + 4];
      share_mem[threadIdx.x * NUM_SHARED_SUM_INPUT + 5] += share_mem[offset + 5];
      share_mem[threadIdx.x * NUM_SHARED_SUM_INPUT + 6] += share_mem[offset + 6];
    }
  }
  __syncthreads();
}

template <typename T>
inline __device__ void InputProp(const int &row, const int &col_dim, const int &param_dim, const T &epsilon,
                                 const T *dy, const T *x, const T *mean, const T *var, const T *gamma, const T *grad_dx,
                                 const T *grad_dg, const T *grad_db, T *d_dy, T *d_x, const T *share_mem,
                                 T *global_sum1, T *global_sum2) {
  for (int col = threadIdx.x; col < col_dim; col += blockDim.x) {
    int pos = (row * col_dim + col);
    int gamma_offset = pos % param_dim;

    T v1 = x[pos] - mean[row];
    T v2 = my_pow(var[row] + epsilon, -0.5);
    T v3 = v1 * v2;

    T part1 = gamma[gamma_offset] * grad_dx[pos] * v2;
    T part2 = gamma[gamma_offset] * share_mem[0] * (1.0 / col_dim);
    T part3 = gamma[gamma_offset] * v3 * share_mem[1] * (1.0 / col_dim);
    T part4 = v3 * grad_dg[gamma_offset];
    d_dy[pos] = part1 + part2 + part3 + part4 + grad_db[gamma_offset];

    T part5 = v1 * (my_pow(var[row] + epsilon, -1.5) * ((share_mem[4] + share_mem[5]) * (-1.0 / col_dim)));
    d_x[pos] += part5 + share_mem[6] * (1.0 / col_dim);

    global_sum1[pos] = share_mem[0] * (1.0 / col_dim);
    global_sum2[pos] = share_mem[1] * (1.0 / col_dim);
  }
}

template <>
inline __device__ void InputProp(const int &row, const int &col_dim, const int &param_dim, const half &epsilon,
                                 const half *dy, const half *x, const half *mean, const half *var, const half *gamma,
                                 const half *grad_dx, const half *grad_dg, const half *grad_db, half *d_dy, half *d_x,
                                 const half *share_mem, half *global_sum1, half *global_sum2) {
  for (int col = threadIdx.x; col < col_dim; col += blockDim.x) {
    int pos = (row * col_dim + col);
    int gamma_offset = pos % param_dim;

    half v1 = x[pos] - mean[row];
    half v2 = my_pow(var[row] + epsilon, -0.5);
    half v3 = v1 * v2;

    half part1 = gamma[gamma_offset] * grad_dx[pos] * v2;
    half part2 = gamma[gamma_offset] * share_mem[0] * __float2half(1.0 / col_dim);
    half part3 = gamma[gamma_offset] * v3 * share_mem[1] * __float2half(1.0 / col_dim);
    half part4 = v3 * grad_dg[gamma_offset];
    d_dy[pos] = part1 + part2 + part3 + part4 + grad_db[gamma_offset];

    half part5 =
      v1 * (my_pow(var[row] + epsilon, -1.5) * ((share_mem[4] + share_mem[5]) * __float2half(-1.0 / col_dim)));
    d_x[pos] += part5 + share_mem[6] * __float2half(1.0 / col_dim);

    global_sum1[pos] = share_mem[0] * __float2half(1.0 / col_dim);
    global_sum2[pos] = share_mem[1] * __float2half(1.0 / col_dim);
  }
}

template <typename T>
__global__ void InputPropKernel(const int row_dim, const int col_dim, const int param_dim, const T epsilon, const T *dy,
                                const T *x, const T *mean, const T *var, const T *gamma, const T *grad_dx,
                                const T *grad_dg, const T *grad_db, T *d_dy, T *d_x, T *global_sum1, T *global_sum2) {
  for (int row = blockIdx.x; row < row_dim; row += gridDim.x) {
    T sum1 = 0;
    T sum2 = 0;
    T sum3 = 0;
    T sum4 = 0;
    T sum5 = 0;
    T sum6 = 0;
    T sum7 = 0;
    DynamicSharedMem<T> share_mem;

    InputThreadReduceInnerMean(row, col_dim, param_dim, epsilon, &sum1, &sum2, &sum3, &sum4, dy, x, mean, var, gamma,
                               grad_dx);
    InputWarpReduceInnerMean(&sum1, &sum2, &sum3, &sum4);
    InputBlockReduceInnerMean(col_dim, &sum1, &sum2, &sum3, &sum4, share_mem.addr());

    InputThreadReduceOuterMean(row, col_dim, param_dim, epsilon, &sum5, &sum6, &sum7, share_mem.addr(), dy, x, mean,
                               var, gamma, grad_dx, grad_dg, d_x);
    InputWarpReduceOuterMean(&sum5, &sum6, &sum7);
    InputBlockReduceOuterMean(col_dim, &sum5, &sum6, &sum7, share_mem.addr());
    InputProp(row, col_dim, param_dim, epsilon, dy, x, mean, var, gamma, grad_dx, grad_dg, grad_db, d_dy, d_x,
              share_mem.addr(), global_sum1, global_sum2);
  }
}

template <typename T>
void CalLayerNormGradGrad(const int &row_dim, const int &col_dim, const int &param_dim, T *global_sum1, T *global_sum2,
                          const T &epsilon, const T *dy, const T *x, const T *mean, const T *var, const T *gamma,
                          const T *grad_dx, const T *grad_dg, const T *grad_db, T *d_dy, T *d_x, T *d_gamma,
                          cudaStream_t stream) {
  int share_mem_size = THREAD_PER_BLOCK / WARP_SIZE * NUM_SHARED_SUM_INPUT * sizeof(T);
  InputPropKernel<<<row_dim, THREAD_PER_BLOCK, share_mem_size, stream>>>(row_dim, col_dim, param_dim, epsilon, dy, x,
                                                                         mean, var, gamma, grad_dx, grad_dg, grad_db,
                                                                         d_dy, d_x, global_sum1, global_sum2);
  share_mem_size = THREAD_PER_BLOCK / WARP_SIZE * NUM_SHARED_SUM_GAMMA * sizeof(T);
  int param_reduce_dim = row_dim * col_dim / param_dim;
  GammaAndBetaPropKernel<<<param_dim, THREAD_PER_BLOCK, share_mem_size, stream>>>(
    param_reduce_dim, param_dim, col_dim, epsilon, dy, x, mean, var, grad_dx, d_gamma, global_sum1, global_sum2);
}

template CUDA_LIB_EXPORT void CalLayerNormGradGrad(const int &row_dim, const int &col_dim, const int &param_dim,
                                                   float *global_sum1, float *global_sum2, const float &epsilon,
                                                   const float *dy, const float *x, const float *mean, const float *var,
                                                   const float *gamma, const float *grad_dx, const float *grad_dg,
                                                   const float *grad_db, float *d_dy, float *d_x, float *d_gamma,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT void CalLayerNormGradGrad(const int &row_dim, const int &col_dim, const int &param_dim,
                                                   half *global_sum1, half *global_sum2, const half &epsilon,
                                                   const half *dy, const half *x, const half *mean, const half *var,
                                                   const half *gamma, const half *grad_dx, const half *grad_dg,
                                                   const half *grad_db, half *d_dy, half *d_x, half *d_gamma,
                                                   cudaStream_t stream);
