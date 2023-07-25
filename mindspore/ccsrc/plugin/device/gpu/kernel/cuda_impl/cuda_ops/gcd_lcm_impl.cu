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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/gcd_lcm_impl.cuh"
#include <math.h>

template <typename T>
struct GcdFunc {
  __device__ __host__ __forceinline__ T operator()(const T &x1, const T &x2) {
    T a = abs(x1);
    T b = abs(x2);
    while (a != 0) {
      T c = a;
      a = b % a;
      b = c;
    }
    return b;
  }
};

template <typename T, typename Func>
__global__ void CalGcdKernel(const int size, const T *x1, const T *x2, T *y) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    y[pos] = Func()(x1[pos], x2[pos]);
  }
}

__device__ __forceinline__ size_t Index(const size_t &index, const size_t &dim) { return dim == 1 ? 0 : index; }

template <typename T, typename Func>
__global__ void BroadcastGcdKernel(const size_t l0, const size_t l1, const size_t l2, const size_t l3, const size_t l4,
                                   const size_t l5, const size_t l6, const size_t r0, const size_t r1, const size_t r2,
                                   const size_t r3, const size_t r4, const size_t r5, const size_t r6, const size_t d0,
                                   const size_t d1, const size_t d2, const size_t d3, const size_t d4, const size_t d5,
                                   const size_t d6, const T *x1, const T *x2, T *y) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < d0 * d1 * d2 * d3 * d4 * d5 * d6;
       pos += blockDim.x * gridDim.x) {
    size_t i = pos / (d1 * d2 * d3 * d4 * d5 * d6) % d0;
    size_t j = pos / (d2 * d3 * d4 * d5 * d6) % d1;
    size_t k = pos / (d3 * d4 * d5 * d6) % d2;
    size_t l = pos / (d4 * d5 * d6) % d3;
    size_t m = pos / (d5 * d6) % d4;
    size_t n = pos / d6 % d5;
    size_t o = pos % d6;

    size_t l_index = Index(i, l0) * l1 * l2 * l3 * l4 * l5 * l6;
    l_index += Index(j, l1) * l2 * l3 * l4 * l5 * l6;
    l_index += Index(k, l2) * l3 * l4 * l5 * l6;
    l_index += Index(l, l3) * l4 * l5 * l6;
    l_index += Index(m, l4) * l5 * l6;
    l_index += Index(n, l5) * l6;
    l_index += Index(o, l6);
    size_t r_index = Index(i, r0) * r1 * r2 * r3 * r4 * r5 * r6;
    r_index += Index(j, r1) * r2 * r3 * r4 * r5 * r6;
    r_index += Index(k, r2) * r3 * r4 * r5 * r6;
    r_index += Index(l, r3) * r4 * r5 * r6;
    r_index += Index(m, r4) * r5 * r6;
    r_index += Index(n, r5) * r6;
    r_index += Index(o, r6);

    y[pos] = Func()(x1[l_index], x2[r_index]);
  }
}

template <typename T, typename Func>
__global__ void CalLcmKernel(const int size, const T *x1, const T *x2, T *y) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    T g = Func()(x1[pos], x2[pos]);
    y[pos] = (g == 0) ? g : abs(x1[pos] / g * x2[pos]);
  }
}

template <typename T, typename Func>
__global__ void BroadcastLcmKernel(const size_t l0, const size_t l1, const size_t l2, const size_t l3, const size_t l4,
                                   const size_t l5, const size_t l6, const size_t r0, const size_t r1, const size_t r2,
                                   const size_t r3, const size_t r4, const size_t r5, const size_t r6, const size_t d0,
                                   const size_t d1, const size_t d2, const size_t d3, const size_t d4, const size_t d5,
                                   const size_t d6, const T *x1, const T *x2, T *y) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < d0 * d1 * d2 * d3 * d4 * d5 * d6;
       pos += blockDim.x * gridDim.x) {
    size_t i = pos / (d1 * d2 * d3 * d4 * d5 * d6) % d0;
    size_t j = pos / (d2 * d3 * d4 * d5 * d6) % d1;
    size_t k = pos / (d3 * d4 * d5 * d6) % d2;
    size_t l = pos / (d4 * d5 * d6) % d3;
    size_t m = pos / (d5 * d6) % d4;
    size_t n = pos / d6 % d5;
    size_t o = pos % d6;

    size_t l_index = Index(i, l0) * l1 * l2 * l3 * l4 * l5 * l6;
    l_index += Index(j, l1) * l2 * l3 * l4 * l5 * l6;
    l_index += Index(k, l2) * l3 * l4 * l5 * l6;
    l_index += Index(l, l3) * l4 * l5 * l6;
    l_index += Index(m, l4) * l5 * l6;
    l_index += Index(n, l5) * l6;
    l_index += Index(o, l6);
    size_t r_index = Index(i, r0) * r1 * r2 * r3 * r4 * r5 * r6;
    r_index += Index(j, r1) * r2 * r3 * r4 * r5 * r6;
    r_index += Index(k, r2) * r3 * r4 * r5 * r6;
    r_index += Index(l, r3) * r4 * r5 * r6;
    r_index += Index(m, r4) * r5 * r6;
    r_index += Index(n, r5) * r6;
    r_index += Index(o, r6);

    T g = Func()(x1[l_index], x2[r_index]);
    y[pos] = (g == 0) ? g : abs(x1[l_index] / g * x2[r_index]);
  }
}

template <typename T>
cudaError_t CalGcd(size_t size, const T *x1, const T *x2, T *y, const uint32_t &device_id, cudaStream_t cuda_stream) {
  CalGcdKernel<T, GcdFunc<T>>
    <<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, x1, x2, y);
  return GetCudaStatus();
}

template <typename T>
cudaError_t BroadcastGcd(const std::vector<size_t> &x1_shape, const std::vector<size_t> &x2_shape,
                         const std::vector<size_t> &y_shape, const T *x1, const T *x2, T *y, const uint32_t &device_id,
                         cudaStream_t cuda_stream) {
  size_t size = 1;
  for (auto d : y_shape) {
    size *= d;
  }
  BroadcastGcdKernel<T, GcdFunc<T>><<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    x1_shape[0], x1_shape[1], x1_shape[2], x1_shape[3], x1_shape[4], x1_shape[5], x1_shape[6], x2_shape[0], x2_shape[1],
    x2_shape[2], x2_shape[3], x2_shape[4], x2_shape[5], x2_shape[6], y_shape[0], y_shape[1], y_shape[2], y_shape[3],
    y_shape[4], y_shape[5], y_shape[6], x1, x2, y);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalLcm(size_t size, const T *x1, const T *x2, T *y, const uint32_t &device_id, cudaStream_t cuda_stream) {
  CalLcmKernel<T, GcdFunc<T>>
    <<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, x1, x2, y);
  return GetCudaStatus();
}

template <typename T>
cudaError_t BroadcastLcm(const std::vector<size_t> &x1_shape, const std::vector<size_t> &x2_shape,
                         const std::vector<size_t> &y_shape, const T *x1, const T *x2, T *y, const uint32_t &device_id,
                         cudaStream_t cuda_stream) {
  size_t size = 1;
  for (auto d : y_shape) {
    size *= d;
  }
  BroadcastLcmKernel<T, GcdFunc<T>><<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    x1_shape[0], x1_shape[1], x1_shape[2], x1_shape[3], x1_shape[4], x1_shape[5], x1_shape[6], x2_shape[0], x2_shape[1],
    x2_shape[2], x2_shape[3], x2_shape[4], x2_shape[5], x2_shape[6], y_shape[0], y_shape[1], y_shape[2], y_shape[3],
    y_shape[4], y_shape[5], y_shape[6], x1, x2, y);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalGcd<int32_t>(size_t, const int32_t *, const int32_t *, int32_t *,
                                                     const uint32_t &, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalGcd<int64_t>(size_t, const int64_t *, const int64_t *, int64_t *,
                                                     const uint32_t &, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t BroadcastGcd<int32_t>(const std::vector<size_t> &, const std::vector<size_t> &,
                                                           const std::vector<size_t> &, const int32_t *,
                                                           const int32_t *, int32_t *, const uint32_t &,
                                                           cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t BroadcastGcd<int64_t>(const std::vector<size_t> &, const std::vector<size_t> &,
                                                           const std::vector<size_t> &, const int64_t *,
                                                           const int64_t *, int64_t *, const uint32_t &,
                                                           cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalLcm<int32_t>(size_t, const int32_t *, const int32_t *, int32_t *,
                                                     const uint32_t &, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalLcm<int64_t>(size_t, const int64_t *, const int64_t *, int64_t *,
                                                     const uint32_t &, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t BroadcastLcm<int32_t>(const std::vector<size_t> &, const std::vector<size_t> &,
                                                           const std::vector<size_t> &, const int32_t *,
                                                           const int32_t *, int32_t *, const uint32_t &,
                                                           cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t BroadcastLcm<int64_t>(const std::vector<size_t> &, const std::vector<size_t> &,
                                                           const std::vector<size_t> &, const int64_t *,
                                                           const int64_t *, int64_t *, const uint32_t &,
                                                           cudaStream_t cuda_stream);
