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
#include "kernel/gpu/cuda_impl/layer_norm_impl.cuh"

constexpr int NUM_PER_THREAD_REDUCE = 4;
constexpr int WARP_SIZE = 32;

template <typename T>
inline __device__ void MeanAndVarAccumulation(T *mean, T *var, T *num, const T &val) {
  // Welford Algorithm:
  // \mu_k = \mu_{k-1} + (x_k - \mu_{k-1})/k
  // \sigma_k^2 = \sigma_{k-1}^2 + (x_k - \mu_{k-1}) * (x_k - \mu_k)
  num[0]++;
  T mean_new = mean[0] + (val - mean[0]) / num[0];
  var[0] = var[0] + (val - mean[0]) * (val - mean_new);
  mean[0] = mean_new;
}

template <typename T>
inline __device__ void MeanAndVarMerge(T *m1, T *v1, T *n1, const T &m2, const T &v2, const T &n2) {
  if (n2 == 0) {
    return;
  }

  T count = n1[0] + n2;
  v1[0] = v1[0] + v2 + (m1[0] - m2) * (m1[0] - m2) * n1[0] * n2 / count;
  m1[0] = (n1[0] * m1[0] + n2 * m2) / count;
  n1[0] = count;
}

template <typename T>
inline __device__ void ThreadReduce(const int &col_dim, const T *block_addr, T *mean, T *var, T *num) {
  int loop_num = (col_dim + NUM_PER_THREAD_REDUCE - 1) / NUM_PER_THREAD_REDUCE;
  for (int i = threadIdx.x; i < loop_num; i += blockDim.x) {
    for (int j = 0; j < NUM_PER_THREAD_REDUCE; j++) {
      int pos = NUM_PER_THREAD_REDUCE * i + j;
      if (pos >= col_dim) {
        return;
      }
      MeanAndVarAccumulation(mean, var, num, block_addr[pos]);
    }
  }
}

template <typename T>
inline __device__ void WarpReduce(T *mean, T *var, T *num) {
  for (int delta = (WARP_SIZE >> 1); delta > 0; delta >>= 1) {
    T mean_other = __shfl_down_sync(0xffffffff, mean[0], delta);
    T var_other = __shfl_down_sync(0xffffffff, var[0], delta);
    T num_other = __shfl_down_sync(0xffffffff, num[0], delta);
    MeanAndVarMerge(mean, var, num, mean_other, var_other, num_other);
  }
}

template <typename T>
inline __device__ void BlockReduce(const int &col_dim, T *mean, T *var, T *num, T *mean_addr, T *var_addr,
                                   T *share_mem) {
  if (threadIdx.x >= col_dim) {
    return;
  }

  // load data to share memory
  // thread(0, 32, 64, 96, ...) keep the data
  if (threadIdx.x % WARP_SIZE == 0) {
    int offset = threadIdx.x / WARP_SIZE * 3;
    share_mem[offset] = mean[0];
    share_mem[offset + 1] = var[0];
    share_mem[offset + 2] = num[0];
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
    mean_addr[blockIdx.x] = share_mem[0];
    share_mem[1] /= col_dim;
    var_addr[blockIdx.x] = share_mem[1];
  }
}

template <typename T>
inline __device__ void LayerNorm(const int &row, const int &col_dim, const int &param_dim, const T *x,
                                 const T *share_mem, const T *gamma, const T *beta, const T epsilon, T *y) {
  for (int col = threadIdx.x; col < col_dim; col += blockDim.x) {
    int pos = row * col_dim + col;
    int i = pos % param_dim;
    y[pos] = (x[pos] - share_mem[0]) / sqrt(share_mem[1] + epsilon) * gamma[i] + beta[i];
  }
}

template <typename T>
__global__ void LayerNormKernel(const int row_dim, const int col_dim, const int param_dim, const T epsilon, const T *x,
                                const T *gamma, const T *beta, T *y, T *mean_addr, T *var_addr) {
  for (auto row = blockIdx.x; row < row_dim; row += gridDim.x) {
    T mean = 0;
    T var = 0;
    T num = 0;
    const T *block_addr = x + row * col_dim;
    extern __shared__ T share_mem[];

    ThreadReduce(col_dim, block_addr, &mean, &var, &num);
    WarpReduce(&mean, &var, &num);
    BlockReduce(col_dim, &mean, &var, &num, mean_addr, var_addr, share_mem);

    __syncthreads();
    LayerNorm(row, col_dim, param_dim, x, share_mem, gamma, beta, epsilon, y);
  }
}

template <typename T>
void LayerNorm(const int &row_dim, const int &col_dim, const int &param_dim, const T &epsilon, const T *x,
               const T *gamma, const T *beta, T *y, T *mean, T *var, cudaStream_t stream) {
  const dim3 block(row_dim);
  const dim3 thread(256);
  // keep the mean/var/num after warp reduce
  int share_mem =
    ((col_dim + NUM_PER_THREAD_REDUCE - 1) / NUM_PER_THREAD_REDUCE + WARP_SIZE - 1) / WARP_SIZE * 3 * sizeof(T);
  LayerNormKernel<<<block, thread, share_mem, stream>>>(row_dim, col_dim, param_dim, epsilon, x, gamma, beta, y, mean,
                                                        var);
}

template void LayerNorm(const int &row_dim, const int &col_dim, const int &param_dim, const float &epsilon,
                        const float *x, const float *gamma, const float *beta, float *y, float *mean, float *var,
                        cudaStream_t stream);
