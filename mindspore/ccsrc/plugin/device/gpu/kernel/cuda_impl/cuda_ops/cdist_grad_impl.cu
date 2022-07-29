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
 * limitations under the License
 */

#include <stdlib.h>
#include "cdist_grad_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T>
__device__ __forceinline__ T sign(T val) {
  return ((0.0) < static_cast<float>(val)) - (static_cast<float>(val) < (0.0));
}

template <typename T>
__global__ void InitOutput(T *grad, const size_t size) {
  T zero = 0.0;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    grad[pos] = zero;
  }
  return;
}

// ONE
template <typename T>
__global__ void CdistGradOne(T *grad, T *dist, T *t1, T *t2, T *res, double p, int64_t r1, int64_t r2, int64_t col,
                             int64_t count, size_t r_size, int64_t x1_size, int64_t x2_size) {
  const int current = (blockIdx.y * gridDim.z + blockIdx.z) * blockDim.y + threadIdx.y;
  const int current_i = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  if (current >= count || current_i >= col) {
    return;
  }
  const T grad_k = grad[current];
  const T dist_k = dist[current];

  const int current_l = current / r_size;
  const int current_k = current % r_size;
  int64_t m = current_k / r2;
  int64_t n = current_k % r2;

  const T * const start = t1 + current_l * x1_size + m * col;
  const T * const end = start + col;
  const T * self_m = start + current_i;
  const T * self_n = t2 + current_l * x2_size + n * col + current_i;
  T * res_m = res + current_l * x1_size + m * col + current_i;

  for (; self_m < end; self_m += stride, self_n += stride, res_m += stride) {
    T res = grad_k * sign(*self_m - *self_n);
    MsAtomicAdd(res_m, res);
  }
}

// less than 2
template <typename T>
__global__ void CdistGradLessthanTwo(T *grad, T *dist, T *t1, T *t2, T *res, double p, int64_t r1,
                                     int64_t r2, int64_t col, int64_t count, size_t r_size, int64_t x1_size,
                                     int64_t x2_size) {
  const int current = (blockIdx.y * gridDim.z + blockIdx.z) * blockDim.y + threadIdx.y;
  const int current_i = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  if (current >= count || current_i >= col) {
    return;
  }
  const T grad_k = grad[current];
  const T dist_k = dist[current];

  if (dist_k != 0.0 && p >= 1) {
    const int current_l = current / r_size;
    const int current_k = current % r_size;
    int64_t m = current_k / r2;
    int64_t n = current_k % r2;

    const T * const start = t1 + current_l * x1_size + m * col;
    const T * const end = start + col;
    const T * self_m = start + current_i;
    const T * self_n = t2 + current_l * x2_size + n * col + current_i;
    T * res_m = res + current_l * x1_size + m * col + current_i;
    for (; self_m < end; self_m += stride, self_n += stride, res_m += stride) {
      const T diff = *self_m - *self_n;
      T res = (sign(diff) * std::pow(std::abs(diff), p - 1) * (grad_k) / std::pow(dist_k, p - 1));
      MsAtomicAdd(res_m, res);
    }
  }
}

// 2
template <typename T>
__global__ void CdistGradTwo(T *grad, T *dist, T *t1, T *t2, T *res, double p, int64_t r1, int64_t r2, int64_t col,
                             int64_t count, size_t r_size, int64_t x1_size, int64_t x2_size) {
  const int current = (blockIdx.y * gridDim.z + blockIdx.z) * blockDim.y + threadIdx.y;
  const int current_i = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  if (current >= count || current_i >= col) {
    return;
  }
  const T grad_k = grad[current];
  const T dist_k = dist[current];

  if (dist_k != 0.0) {
    const int current_l = current / r_size;
    const int current_k = current % r_size;
    int64_t m = current_k / r2;
    int64_t n = current_k % r2;

    const T * const start = t1 + current_l * x1_size + m * col;
    const T * const end = start + col;
    const T * self_m = start + current_i;
    const T * self_n = t2 + current_l * x2_size + n * col + current_i;
    T * res_m = res + current_l * x1_size + m * col + current_i;

    for (; self_m < end; self_m += stride, self_n += stride, res_m += stride) {
      T res = grad_k * (*self_m - *self_n) / dist_k;
      MsAtomicAdd(res_m, res);
    }
  }
}

// P
template <typename T>
__global__ void CdistGradP(T *grad, T *dist, T *t1, T *t2, T *res, double p, int64_t r1, int64_t r2, int64_t col,
                             int64_t count, size_t r_size, int64_t x1_size, int64_t x2_size) {
  const int current = (blockIdx.y * gridDim.z + blockIdx.z) * blockDim.y + threadIdx.y;
  const int current_i = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  if (current >= count || current_i >= col) {
    return;
  }
  const T grad_k = grad[current];
  const T dist_k = dist[current];

  if (dist_k != 0.0) {
    const int current_l = current / r_size;
    const int current_k = current % r_size;
    int64_t m = current_k / r2;
    int64_t n = current_k % r2;

    const T * const start = t1 + current_l * x1_size + m * col;
    const T * const end = start + col;
    const T * self_m = start + current_i;
    const T * self_n = t2 + current_l * x2_size + n * col + current_i;
    T * res_m = res + current_l * x1_size + m * col + current_i;
    T dist_k_pow = std::pow(dist_k, p - 1);
    for (; self_m < end; self_m += stride, self_n += stride, res_m += stride) {
      const T diff = *self_m - *self_n;
      T res_num = diff * std::pow(std::abs(diff), p - 2) * grad_k / std::pow(dist_k, p - 1);
      MsAtomicAdd(res_m, res_num);
    }
  }
}

// INF
template <typename T>
__global__ void CdistGradInf(T *grad, T *dist, T *t1, T *t2, T *res, double p, int64_t r1, int64_t r2, int64_t col,
                             int64_t count, size_t r_size, int64_t x1_size, int64_t x2_size) {
  const int current = (blockIdx.y * gridDim.z + blockIdx.z) * blockDim.y + threadIdx.y;
  const int current_i = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  if (current >= count || current_i >= col) {
    return;
  }
  const T grad_k = grad[current];
  const T dist_k = dist[current];

  const int current_l = current / r_size;
  const int current_k = current % r_size;
  int64_t m = current_k / r2;
  int64_t n = current_k % r2;

  const T * const start = t1 + current_l * x1_size + m * col;
  const T * const end = start + col;
  const T * self_m = start + current_i;
  const T * self_n = t2 + current_l * x2_size + n * col + current_i;
  T * res_m = res + current_l * x1_size + m * col + current_i;

  for (; self_m < end; self_m += stride, self_n += stride, res_m += stride) {
    T diff = *self_m - *self_n;
    T res = grad_k * sign(diff) * (std::abs(diff) == (dist_k));
    MsAtomicAdd(res_m, res);
  }
}


// CAL
template <typename T>
void CalCdistGrad(size_t out_size, int64_t l1_size, int64_t l2_size, T *grad_start, T *dist_start, T *t1_start,
                  T *t2_start, T *res_start, int64_t m, double p, int64_t r1, int64_t r2, int64_t batch,
                  const uint32_t &device_id, cudaStream_t cuda_stream) {
  InitOutput<<<CUDA_BLOCKS(device_id, out_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(res_start, out_size);

  if (p == 0.0 || out_size == 0 || l1_size == 0 || l2_size == 0) {
    return;
  }

  const int block_x = 64;
  const int block_y = 16;
  const int grid_x = (m + block_x * 8 - 1) / (block_x * 8);

  const int64_t count = batch * r2 * r1;
  const int64_t grid_temp = (count + block_y - 1) / block_y;

  const int grid_y = (grid_temp - 1) / 65535 + 1;
  const int grid_z = (grid_temp - 1) / grid_y + 1;

  const dim3 grid(grid_x, grid_y, grid_z);
  const dim3 block(block_x, block_y);

  const int64_t r_size = r1 * r2;

  if (std::isinf(p)) {
    CdistGradInf<T><<<grid, block, 0, cuda_stream>>>(grad_start, dist_start, t1_start, t2_start, res_start,
                                                     p, r1, r2, m, count, r_size, l1_size, l2_size);
  } else if (p == 1.0) {
    CdistGradOne<T><<<grid, block, 0, cuda_stream>>>(grad_start, dist_start, t1_start, t2_start, res_start,
                                                     p, r1, r2, m, count, r_size, l1_size, l2_size);
  } else if (p < 2.0) {
    CdistGradLessthanTwo<T><<<grid, block, 0, cuda_stream>>>(grad_start, dist_start, t1_start, t2_start,
                                                             res_start, p, r1, r2, m, count, r_size,
                                                             l1_size, l2_size);
  } else if (p == 2.0) {
    CdistGradTwo<T><<<grid, block, 0, cuda_stream>>>(grad_start, dist_start, t1_start, t2_start, res_start,
                                                     p, r1, r2, m, count, r_size, l1_size, l2_size);
  } else {
    CdistGradP<T><<<grid, block, 0, cuda_stream>>>(grad_start, dist_start, t1_start, t2_start, res_start,
                                                   p, r1, r2, m, count, r_size, l1_size, l2_size);
  }

  return;
}



template
CUDA_LIB_EXPORT void CalCdistGrad<float>(size_t out_size, int64_t l1_size, int64_t l2_size, float *grad_start,
                                         float *dist_start, float *t1_start, float *t2_start, float *res_start,
                                         int64_t m, double p, int64_t r1, int64_t r2, int64_t batch,
                                         const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalCdistGrad<double>(size_t out_size, int64_t l1_size, int64_t l2_size, double *grad_start,
                                          double *dist_start, double *t1_start, double *t2_start, double *res_start,
                                          int64_t m, double p, int64_t r1, int64_t r2, int64_t batch,
                                          const uint32_t &device_id, cudaStream_t cuda_stream);
