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

#include "cdist_impl.cuh"
#include <float.h>

static const int forward_threads = 256;

template <typename T>
__device__ __forceinline__ T WARP_SHFL_DOWN(T value, unsigned int delta, int width = warpSize,
                                            unsigned int mask = 0xffffffff) {
#if !defined(USE_ROCM)
  return __shfl_down_sync(mask, value, delta, width);
#else
  return __shfl_down(value, delta, width);
#endif
}

// ZERO
template <typename T>
__global__ void CdistZero(T *x1, T *x2, T *result, double p, const int64_t r2, const int64_t m, const int64_t r_size,
                          const int64_t l1_size, const int64_t l2_size) {
  const int64_t l = blockIdx.x / r_size;
  const int64_t k = blockIdx.x % r_size;
  const int64_t i = k / r2;
  const int64_t j = k % r2;
  const int stride = blockDim.x;

  const T *const start = x1 + l * l1_size + i * m;
  const T *const end = start + m;
  const T *a = start + threadIdx.x;
  const T *b = x2 + l * l2_size + j * m + threadIdx.x;
  T res = 0.0;

  for (; a < end; a += stride, b += stride) {
    res += (*a == *b) ? 0 : 1;
  }

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    res += WARP_SHFL_DOWN(res, offset);
  }

  __shared__ T shared[forward_threads];
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
    result[blockIdx.x] = res;
  }
  return;
}

// One
template <typename T>
__global__ void CdistOne(T *x1, T *x2, T *result, double p, const int64_t r2, const int64_t m, const int64_t r_size,
                         const int64_t l1_size, const int64_t l2_size) {
  const int64_t l = blockIdx.x / r_size;
  const int64_t k = blockIdx.x % r_size;
  const int64_t i = k / r2;
  const int64_t j = k % r2;
  const int stride = blockDim.x;

  const T *const start = x1 + l * l1_size + i * m;
  const T *const end = start + m;
  const T *a = start + threadIdx.x;
  const T *b = x2 + l * l2_size + j * m + threadIdx.x;
  T res = 0.0;
  for (; a < end; a += stride, b += stride) {
    res += abs(*a - *b);
  }
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    res += WARP_SHFL_DOWN(res, offset);
  }

  __shared__ T shared[forward_threads];
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
    result[blockIdx.x] = res;
  }
  return;
}

// P
template <typename T>
__global__ void CdistP(T *x1, T *x2, T *result, double p, const int64_t r2, const int64_t m, const int64_t r_size,
                       const int64_t l1_size, const int64_t l2_size) {
  const int64_t l = blockIdx.x / r_size;
  const int64_t k = blockIdx.x % r_size;
  const int64_t i = k / r2;
  const int64_t j = k % r2;
  const int stride = blockDim.x;

  const T *const start = x1 + l * l1_size + i * m;
  const T *const end = start + m;
  const T *a = start + threadIdx.x;
  const T *b = x2 + l * l2_size + j * m + threadIdx.x;
  T res = 0.0;
  for (; a < end; a += stride, b += stride) {
    res += static_cast<T>(pow(static_cast<double>(abs(*a - *b)), p));
  }

  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    res += WARP_SHFL_DOWN(res, offset);
  }

  __shared__ T shared[forward_threads];
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
    result[blockIdx.x] = static_cast<T>(pow(static_cast<double>(res), 1.0 / p));
  }
  return;
}

// Inf
template <typename T>
__global__ void CdistInf(T *x1, T *x2, T *result, double p, const int64_t r2, const int64_t m, const int64_t r_size,
                         const int64_t l1_size, const int64_t l2_size) {
  const int64_t l = blockIdx.x / r_size;
  const int64_t k = blockIdx.x % r_size;
  const int64_t i = k / r2;
  const int64_t j = k % r2;
  const int stride = blockDim.x;

  const T *const start = x1 + l * l1_size + i * m;
  const T *const end = start + m;
  const T *a = start + threadIdx.x;
  const T *b = x2 + l * l2_size + j * m + threadIdx.x;
  T res = 0.0;
  for (; a < end; a += stride, b += stride) {
    res = abs(*a - *b) > res ? abs(*a - *b) : res;
  }
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    const T other = WARP_SHFL_DOWN(res, offset);
    if (other > res) {
      res = other;
    }
  }

  __shared__ T shared[forward_threads];
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
    result[blockIdx.x] = res;
  }
  return;
}

bool checkinf(const double p) { return (p >= INT_MAX || p <= -INT_MAX); }

// CAL
template <typename T>
cudaError_t CalCdist(size_t out_size, T *input_x, T *input_y, T *output, int64_t x_row, int64_t y_row, int64_t col,
                     double p, int64_t batch, const uint32_t &device_id, cudaStream_t cuda_stream) {
  const int64_t r_size = x_row * y_row;
  const int64_t l1_size = x_row * col;
  const int64_t l2_size = y_row * col;
  const dim3 grid(out_size);
  const dim3 block(forward_threads);
  if (p == 0.0) {
    CdistZero<T><<<grid, block, 0, cuda_stream>>>(input_x, input_y, output, p, y_row, col, r_size, l1_size, l2_size);
  } else if (p == 1.0) {
    CdistOne<T><<<grid, block, 0, cuda_stream>>>(input_x, input_y, output, p, y_row, col, r_size, l1_size, l2_size);
  } else if (checkinf(p)) {
    CdistInf<T><<<grid, block, 0, cuda_stream>>>(input_x, input_y, output, p, y_row, col, r_size, l1_size, l2_size);
  } else {
    CdistP<T><<<grid, block, 0, cuda_stream>>>(input_x, input_y, output, p, y_row, col, r_size, l1_size, l2_size);
  }
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalCdist<float>(size_t out_size, float *input_x, float *input_y, float *output,
                                                     int64_t x_row, int64_t y_row, int64_t col, double p, int64_t batch,
                                                     const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalCdist<double>(size_t out_size, double *input_x, double *input_y, double *output,
                                                      int64_t x_row, int64_t y_row, int64_t col, double p,
                                                      int64_t batch, const uint32_t &device_id,
                                                      cudaStream_t cuda_stream);
