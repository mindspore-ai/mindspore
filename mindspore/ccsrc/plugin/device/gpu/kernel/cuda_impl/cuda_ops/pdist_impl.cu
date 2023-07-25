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

#include "pdist_impl.cuh"
#include <math.h>

static const int threads = 256;

template <typename T>
__device__ __forceinline__ T WARP_SHFL_DOWN(T value, unsigned int delta, int width = warpSize,
                                            unsigned int mask = 0xffffffff) {
#if !defined(USE_ROCM)
  return __shfl_down_sync(mask, value, delta, width);
#else
  return __shfl_down(value, delta, width);
#endif
}

template <typename T>
__global__ void PDist_Zero(const T *x, T *y, const float p, const int64_t n, const int64_t m, const float n1,
                           const float n2) {
  const int64_t pos = blockIdx.x;
  const int s = blockDim.x;

  int64_t i = static_cast<int64_t>((n1 - sqrt(n2 - 2 * pos)));
  int64_t j = pos - n * i + i * (i + 1) / 2 + i + 1;

  const T *const begin = x + i * m;
  const T *const end = begin + m;
  const T *x_i = begin + threadIdx.x;
  const T *x_j = x + j * m + threadIdx.x;
  T res = 0.0;
  for (; x_i < end; x_i += s, x_j += s) {
    res += (*x_i == *x_j) ? 0 : 1;
  }

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    res += WARP_SHFL_DOWN(res, offset);
  }

  __shared__ T shared[threads];
  int lane = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;
  if (lane == 0) {
    shared[warp_id] = res;
  }

  __syncthreads();
  res = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0;
  if (warp_id == 0) {
    for (int offset = blockDim.x / warpSize / 2; offset > 0; offset /= 2) {
      res += WARP_SHFL_DOWN(res, offset);
    }
  }

  if (threadIdx.x == 0) {
    y[pos] = res;
  }
}

template <typename T>
__global__ void PDist_One(const T *x, T *y, const float p, const int64_t n, const int64_t m, const float n1,
                          const float n2) {
  const int64_t pos = blockIdx.x;
  const int s = blockDim.x;

  int64_t i = static_cast<int64_t>((n1 - sqrt(n2 - 2 * pos)));
  int64_t j = pos - n * i + i * (i + 1) / 2 + i + 1;

  const T *const begin = x + i * m;
  const T *const end = begin + m;
  const T *x_i = begin + threadIdx.x;
  const T *x_j = x + j * m + threadIdx.x;
  T res = 0.0;
  for (; x_i < end; x_i += s, x_j += s) {
    res += abs(*x_i - *x_j);
  }

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    res += WARP_SHFL_DOWN(res, offset);
  }

  __shared__ T shared[threads];
  int lane = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;
  if (lane == 0) {
    shared[warp_id] = res;
  }

  __syncthreads();
  res = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0;
  if (warp_id == 0) {
    for (int offset = blockDim.x / warpSize / 2; offset > 0; offset /= 2) {
      res += WARP_SHFL_DOWN(res, offset);
    }
  }

  if (threadIdx.x == 0) {
    y[pos] = res;
  }
}

template <typename T>
__global__ void PDist_Inf(const T *x, T *y, const float p, const int64_t n, const int64_t m, const float n1,
                          const float n2) {
  const int64_t pos = blockIdx.x;
  const int s = blockDim.x;

  // The -1 accounts for floating point truncation issues
  int64_t i = static_cast<int64_t>((n1 - sqrt(n2 - 2 * pos)));
  int64_t j = pos - n * i + i * (i + 1) / 2 + i + 1;

  const T *const begin = x + i * m;
  const T *const end = begin + m;
  const T *x_i = begin + threadIdx.x;
  const T *x_j = x + j * m + threadIdx.x;
  T res = 0.0;
  for (; x_i < end; x_i += s, x_j += s) {
    res = abs(*x_i - *x_j) > res ? abs(*x_i - *x_j) : res;
  }

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    const T other = WARP_SHFL_DOWN(res, offset);
    if (other > res) {
      res = other;
    }
  }

  __shared__ T shared[threads];
  int lane = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;
  if (lane == 0) {
    shared[warp_id] = res;
  }

  __syncthreads();
  res = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0;
  if (warp_id == 0) {
    for (int offset = blockDim.x / warpSize / 2; offset > 0; offset /= 2) {
      const T other = WARP_SHFL_DOWN(res, offset);
      if (other > res) {
        res = other;
      }
    }
  }

  if (threadIdx.x == 0) {
    y[pos] = res;
  }
}

template <typename T>
__global__ void PDist_Other(const T *x, T *y, const float p, const int64_t n, const int64_t m, const float n1,
                            const float n2) {
  const int64_t pos = blockIdx.x;
  const int s = blockDim.x;

  // The -1 accounts for floating point truncation issues
  int64_t i = static_cast<int64_t>((n1 - sqrt(n2 - 2 * pos)));
  int64_t j = pos - n * i + i * (i + 1) / 2 + i + 1;

  const T *const begin = x + i * m;
  const T *const end = begin + m;
  const T *x_i = begin + threadIdx.x;
  const T *x_j = x + j * m + threadIdx.x;
  T res = 0.0;
  for (; x_i < end; x_i += s, x_j += s) {
    res += pow(abs(*x_i - *x_j), static_cast<T>(p));
  }

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    res += WARP_SHFL_DOWN(res, offset);
  }

  __shared__ T shared[threads];
  int lane = threadIdx.x % warpSize;
  int warp_id = threadIdx.x / warpSize;
  if (lane == 0) {
    shared[warp_id] = res;
  }

  __syncthreads();
  res = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0;
  if (warp_id == 0) {
    for (int offset = blockDim.x / warpSize / 2; offset > 0; offset /= 2) {
      res += WARP_SHFL_DOWN(res, offset);
    }
  }

  if (threadIdx.x == 0) {
    y[pos] = pow(res, static_cast<T>(1.0 / p));
  }
}

template <typename T>
cudaError_t CalPDist(const size_t x_size, const size_t y_size, const T *x, T *y, const float p, const int64_t n,
                     const int64_t m, const uint32_t &device_id, cudaStream_t cuda_stream) {
  const dim3 grid(y_size);
  const dim3 block(threads);
  const float n1 = n - .5;
  const float n2 = n1 * n1 - 1;
  if (p == 0.0) {
    PDist_Zero<T><<<grid, block, 0, cuda_stream>>>(x, y, p, n, m, n1, n2);
  } else if (p == 1.0) {
    PDist_One<T><<<grid, block, 0, cuda_stream>>>(x, y, p, n, m, n1, n2);
  } else if (std::isinf(p)) {
    PDist_Inf<T><<<grid, block, 0, cuda_stream>>>(x, y, p, n, m, n1, n2);
  } else {
    PDist_Other<T><<<grid, block, 0, cuda_stream>>>(x, y, p, n, m, n1, n2);
  }
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalPDist<float>(const size_t x_size, const size_t y_size, const float *x, float *y,
                                                     const float p, const int64_t n, const int64_t m,
                                                     const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPDist<double>(const size_t x_size, const size_t y_size, const double *x,
                                                      double *y, const float p, const int64_t n, const int64_t m,
                                                      const uint32_t &device_id, cudaStream_t cuda_stream);
