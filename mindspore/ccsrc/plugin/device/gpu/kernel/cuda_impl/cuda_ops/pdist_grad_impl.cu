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

#include "pdist_grad_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T>
__device__ __forceinline__ T sign(T val) {
  return (0.0 < val) - (val < 0.0);
}

template <typename T>
__global__ void InitOutput(T *x_grad, const size_t x_size) {
  T zero = 0.0;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < x_size; pos += blockDim.x * gridDim.x) {
    x_grad[pos] = zero;
  }
  return;
}

template <typename T>
__global__ void PDist_Grad_One(const size_t y_size, const T *y_grad, const T *x, const T *y, T *buffer, const int64_t n,
                               const int64_t m, const float p, const float n1, const float n2) {
  const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  const int init = blockIdx.y * blockDim.y + threadIdx.y;
  const int s = blockDim.y * gridDim.y;

  if (k >= y_size) {
    return;
  }

  int64_t i = static_cast<int64_t>((n1 - sqrt(n2 - 2 * k)));
  int64_t j = k - n * i + i * (i + 1) / 2 + i + 1;
  int64_t ib = j - i - 1;
  int64_t jb = n - 2 - i;

  const T grad_k = y_grad[k];

  const T *const begin = x + i * m;
  const T *const end = begin + m;
  const T *x_i = begin + init;
  const T *x_j = x + j * m + init;
  T *buff1 = buffer + (ib * n + i) * m + init;
  T *buff2 = buffer + (jb * n + j) * m + init;
  for (; x_i < end; x_i += s, x_j += s, buff1 += s, buff2 += s) {
    T res = grad_k * sign(*x_i - *x_j);
    *buff1 = res;
    *buff2 = -res;
  }
}

template <typename T>
__global__ void PDist_Grad_Lt_Two(const size_t y_size, const T *y_grad, const T *x, const T *y, T *buffer,
                                  const int64_t n, const int64_t m, const float p, const float n1, const float n2) {
  const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  const int init = blockIdx.y * blockDim.y + threadIdx.y;
  const int s = blockDim.y * gridDim.y;

  if (k >= y_size) {
    return;
  }

  int64_t i = static_cast<int64_t>((n1 - sqrt(n2 - 2 * k)));
  int64_t j = k - n * i + i * (i + 1) / 2 + i + 1;
  int64_t ib = j - i - 1;
  int64_t jb = n - 2 - i;

  const T grad_k = y_grad[k];
  const T dist_k = y[k];

  const T *const begin = x + i * m;
  const T *const end = begin + m;
  const T *x_i = begin + init;
  const T *x_j = x + j * m + init;
  T *buff1 = buffer + (ib * n + i) * m + init;
  T *buff2 = buffer + (jb * n + j) * m + init;
  for (; x_i < end; x_i += s, x_j += s, buff1 += s, buff2 += s) {
    T res;
    const T diff = *x_i - *x_j;
    if (dist_k == 0.0 || (diff == 0.0 && p < 1)) {
      res = 0;
    } else {
      res = (sign(diff) * std::pow(std::abs(diff), p - 1) * (grad_k) / std::pow(dist_k, p - 1));
    }
    *buff1 = res;
    *buff2 = -res;
  }
}

template <typename T>
__global__ void PDist_Grad_Two(const size_t y_size, const T *y_grad, const T *x, const T *y, T *buffer, const int64_t n,
                               const int64_t m, const float p, const float n1, const float n2) {
  const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  const int init = blockIdx.y * blockDim.y + threadIdx.y;
  const int s = blockDim.y * gridDim.y;

  if (k >= y_size) {
    return;
  }

  int64_t i = static_cast<int64_t>((n1 - sqrt(n2 - 2 * k)));
  int64_t j = k - n * i + i * (i + 1) / 2 + i + 1;
  int64_t ib = j - i - 1;
  int64_t jb = n - 2 - i;

  const T grad_k = y_grad[k];
  const T dist_k = y[k];

  const T *const begin = x + i * m;
  const T *const end = begin + m;
  const T *x_i = begin + init;
  const T *x_j = x + j * m + init;
  T *buff1 = buffer + (ib * n + i) * m + init;
  T *buff2 = buffer + (jb * n + j) * m + init;
  for (; x_i < end; x_i += s, x_j += s, buff1 += s, buff2 += s) {
    T res = 0;
    if (dist_k != 0.0) {
      res = grad_k * (*x_i - *x_j) / dist_k;
    }
    *buff1 = res;
    *buff2 = -res;
  }
}

template <typename T>
__global__ void PDist_Grad_P(const size_t y_size, const T *y_grad, const T *x, const T *y, T *buffer, const int64_t n,
                             const int64_t m, const float p, const float n1, const float n2) {
  const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  const int init = blockIdx.y * blockDim.y + threadIdx.y;
  const int s = blockDim.y * gridDim.y;

  if (k >= y_size) {
    return;
  }

  int64_t i = static_cast<int64_t>((n1 - sqrt(n2 - 2 * k)));
  int64_t j = k - n * i + i * (i + 1) / 2 + i + 1;
  int64_t ib = j - i - 1;
  int64_t jb = n - 2 - i;

  const T grad_k = y_grad[k];
  const T dist_k = y[k];

  const T *const begin = x + i * m;
  const T *const end = begin + m;
  const T *x_i = begin + init;
  const T *x_j = x + j * m + init;
  T *buff1 = buffer + (ib * n + i) * m + init;
  T *buff2 = buffer + (jb * n + j) * m + init;
  for (; x_i < end; x_i += s, x_j += s, buff1 += s, buff2 += s) {
    T res = 0;
    const T diff = (*x_i - *x_j);
    if (dist_k != (0.0)) {
      res = diff * std::pow(std::abs(diff), p - 2) * grad_k / std::pow(dist_k, p - 1);
    }
    *buff1 = res;
    *buff2 = -res;
  }
}

template <typename T>
__global__ void PDist_Grad_Inf(const size_t y_size, const T *y_grad, const T *x, const T *y, T *buffer, const int64_t n,
                               const int64_t m, const float p, const float n1, const float n2) {
  const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
  const int init = blockIdx.y * blockDim.y + threadIdx.y;
  const int s = blockDim.y * gridDim.y;

  if (k >= y_size) {
    return;
  }

  int64_t i = static_cast<int64_t>((n1 - sqrt(n2 - 2 * k)));
  int64_t j = k - n * i + i * (i + 1) / 2 + i + 1;
  int64_t ib = j - i - 1;
  int64_t jb = n - 2 - i;

  const T grad_k = y_grad[k];
  const T dist_k = y[k];

  const T *const begin = x + i * m;
  const T *const end = begin + m;
  const T *x_i = begin + init;
  const T *x_j = x + j * m + init;
  T *buff1 = buffer + (ib * n + i) * m + init;
  T *buff2 = buffer + (jb * n + j) * m + init;
  for (; x_i < end; x_i += s, x_j += s, buff1 += s, buff2 += s) {
    T diff = *x_i - *x_j;
    T res = grad_k * sign(diff) * (std::abs(diff) == (dist_k));
    *buff1 = res;
    *buff2 = -res;
  }
}

template <typename T>
__global__ void AddBuffer(T *x_grad, T *buffer, const int64_t n, const size_t size) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    T res = 0.0;
    T *buff = buffer + pos;
    for (int64_t i = 0; i < n - 1; ++i, buff += size) {
      res += *(buff);
    }
    x_grad[pos] = res;
  }
  return;
}

template <typename T>
void CalPDistGrad(const size_t x_size, const size_t y_size, const size_t grad_size, const T *y_grad, const T *x,
                  const T *y, const int64_t n, const int64_t m, const float p, T *x_grad, T *buffer,
                  const uint32_t &device_id, cudaStream_t cuda_stream) {
  if (p == 0.0 || grad_size == 0 || x_size == 0) {
    InitOutput<<<CUDA_BLOCKS(device_id, x_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(x_grad, x_size);
    return;
  }

  const int block_x = 8;
  const int block_y = 128;
  const int grid_x = (y_size + block_x - 1) / block_x;
  const int grid_y = (m + block_y * 8 - 1) / (block_y * 8);
  const dim3 grid(grid_x, grid_y);
  const dim3 block(block_x, block_y);
  const float n1 = n - .5;
  const float n2 = n1 * n1 - 1;
  if (p == 1.0) {
    PDist_Grad_One<T><<<grid, block, 0, cuda_stream>>>(y_size, y_grad, x, y, buffer, n, m, p, n1, n2);
  } else if (p < 2.0) {
    PDist_Grad_Lt_Two<T><<<grid, block, 0, cuda_stream>>>(y_size, y_grad, x, y, buffer, n, m, p, n1, n2);
  } else if (p == 2.0) {
    PDist_Grad_Two<T><<<grid, block, 0, cuda_stream>>>(y_size, y_grad, x, y, buffer, n, m, p, n1, n2);
  } else if (std::isinf(p)) {
    PDist_Grad_Inf<T><<<grid, block, 0, cuda_stream>>>(y_size, y_grad, x, y, buffer, n, m, p, n1, n2);
  } else {
    PDist_Grad_P<T><<<grid, block, 0, cuda_stream>>>(y_size, y_grad, x, y, buffer, n, m, p, n1, n2);
  }
  AddBuffer<<<CUDA_BLOCKS(device_id, x_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(x_grad, buffer, n, x_size);
}

template CUDA_LIB_EXPORT void CalPDistGrad<float>(const size_t x_size, const size_t y_size, const size_t grad_size,
                                                  const float *y_grad, const float *x, const float *y, const int64_t n,
                                                  const int64_t m, const float p, float *x_grad, float *buffer,
                                                  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalPDistGrad<double>(const size_t x_size, const size_t y_size, const size_t grad_size,
                                                   const double *y_grad, const double *x, const double *y,
                                                   const int64_t n, const int64_t m, const float p,
                                                   double *x_grad, double *buffer,
                                                   const uint32_t &device_id, cudaStream_t cuda_stream);
