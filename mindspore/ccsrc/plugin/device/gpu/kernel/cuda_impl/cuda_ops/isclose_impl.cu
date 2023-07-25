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
#include <cuda_runtime.h>
#include <limits>
#include <algorithm>
#include <vector>
#include <numeric>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/isclose_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__device__ __forceinline__ T Abs(const T &input) {
  return abs(input);
}
template <>
__device__ __forceinline__ half Abs(const half &input) {
  return abs(__half2float(input));
}
template <typename T>
__device__ __forceinline__ bool IsNan(const T &input) {
  return input != input;
}
template <typename T>
__device__ __forceinline__ bool IsFinite(const T &input) {
  return isfinite(input);
}
template <>
__device__ __forceinline__ bool IsFinite(const half &input) {
  return isfinite(__half2float(input));
}
template <>
__device__ __forceinline__ bool IsFinite(const int8_t &input) {
  return true;
}
template <>
__device__ __forceinline__ bool IsFinite(const int16_t &input) {
  return true;
}
template <>
__device__ __forceinline__ bool IsFinite(const int32_t &input) {
  return true;
}
template <>
__device__ __forceinline__ bool IsFinite(const int64_t &input) {
  return true;
}
template <>
__device__ __forceinline__ bool IsFinite(const uint8_t &input) {
  return true;
}
template <typename T>
__device__ __forceinline__ bool IsCloseFunc(const T &inputx, const T &inputy, const float rtol, const float atol) {
  bool close = inputx == inputy;
  if (atol == 0 && rtol == 0) return close;
  auto diff = Abs(inputx - inputy);
  auto limit = static_cast<T>(atol) + (static_cast<T>(rtol) * Abs(inputy));
  close |= ((IsFinite(diff)) && (diff <= limit));
  return close;
}
template <typename T>
__device__ __forceinline__ bool IsCloseEqualNanFunc(const T &inputx, const T &inputy, const float rtol,
                                                    const float atol) {
  bool close = inputx == inputy;
  close |= (IsNan(inputx) && IsNan(inputy));
  if (atol == 0 && rtol == 0) return close;
  auto diff = Abs(inputx - inputy);
  auto limit = static_cast<T>(atol) + (static_cast<T>(rtol) * Abs(inputy));
  close |= ((IsFinite(diff)) && (diff <= limit));
  return close;
}

template <typename T>
__global__ void IsCloseEqualNanKernel(size_t size, const T *inputx, const T *inputy, const float rtol, const float atol,
                                      const bool equal_nan, bool *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = IsCloseEqualNanFunc(inputx[pos], inputy[pos], rtol, atol);
  }
}

template <typename T>
__global__ void IsCloseKernel(size_t size, const T *inputx, const T *inputy, const float rtol, const float atol,
                              const bool equal_nan, bool *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = IsCloseFunc(inputx[pos], inputy[pos], rtol, atol);
  }
}

__device__ __forceinline__ size_t Index(const size_t &index, const size_t &dim) { return dim == 1 ? 0 : index; }

template <typename T>
__global__ void BroadcastIsCloseEqualNanKernel(const size_t l0, const size_t l1, const size_t l2, const size_t l3,
                                               const size_t l4, const size_t l5, const size_t l6, const size_t r0,
                                               const size_t r1, const size_t r2, const size_t r3, const size_t r4,
                                               const size_t r5, const size_t r6, const size_t d0, const size_t d1,
                                               const size_t d2, const size_t d3, const size_t d4, const size_t d5,
                                               const size_t d6, const T *inputx, const T *inputy, const float rtol,
                                               const float atol, const bool equal_nan, bool *output) {
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
    output[pos] = IsCloseEqualNanFunc(inputx[l_index], inputy[r_index], rtol, atol);
  }
}

template <typename T>
__global__ void BroadcastIsCloseKernel(const size_t l0, const size_t l1, const size_t l2, const size_t l3,
                                       const size_t l4, const size_t l5, const size_t l6, const size_t r0,
                                       const size_t r1, const size_t r2, const size_t r3, const size_t r4,
                                       const size_t r5, const size_t r6, const size_t d0, const size_t d1,
                                       const size_t d2, const size_t d3, const size_t d4, const size_t d5,
                                       const size_t d6, const T *inputx, const T *inputy, const float rtol,
                                       const float atol, const bool equal_nan, bool *output) {
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
    output[pos] = IsCloseFunc(inputx[l_index], inputy[r_index], rtol, atol);
  }
}

template <typename T>
cudaError_t IsClose(size_t size, const T *inputx, const T *inputy, const float atol, const float rtol,
                    const bool equal_nan, bool *output, const uint32_t &device_id, cudaStream_t cuda_stream) {
  if (equal_nan) {
    IsCloseEqualNanKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      size, inputx, inputy, atol, rtol, equal_nan, output);
  } else {
    IsCloseKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, inputx, inputy, atol,
                                                                                             rtol, equal_nan, output);
  }
  return GetCudaStatus();
}

template <typename T>
cudaError_t BroadcastIsClose(const std::vector<size_t> &inputx_shape, const std::vector<size_t> &inputy_shape,
                             const std::vector<size_t> &output_shape, const T *inputx, const T *inputy,
                             const float atol, const float rtol, const bool equal_nan, bool *output,
                             const uint32_t &device_id, cudaStream_t cuda_stream) {
  size_t size = 1;
  for (auto d : output_shape) {
    size *= d;
  }
  if (equal_nan) {
    BroadcastIsCloseEqualNanKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      inputx_shape[0], inputx_shape[1], inputx_shape[2], inputx_shape[3], inputx_shape[4], inputx_shape[5],
      inputx_shape[6], inputy_shape[0], inputy_shape[1], inputy_shape[2], inputy_shape[3], inputy_shape[4],
      inputy_shape[5], inputy_shape[6], output_shape[0], output_shape[1], output_shape[2], output_shape[3],
      output_shape[4], output_shape[5], output_shape[6], inputx, inputy, atol, rtol, equal_nan, output);
  } else {
    BroadcastIsCloseKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      inputx_shape[0], inputx_shape[1], inputx_shape[2], inputx_shape[3], inputx_shape[4], inputx_shape[5],
      inputx_shape[6], inputy_shape[0], inputy_shape[1], inputy_shape[2], inputy_shape[3], inputy_shape[4],
      inputy_shape[5], inputy_shape[6], output_shape[0], output_shape[1], output_shape[2], output_shape[3],
      output_shape[4], output_shape[5], output_shape[6], inputx, inputy, atol, rtol, equal_nan, output);
  }
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t IsClose<half>(size_t size, const half *inputx, const half *inputy,
                                                   const float atol, const float rtol, const bool equal_nan,
                                                   bool *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t IsClose<float>(size_t size, const float *inputx, const float *inputy,
                                                    const float atol, const float rtol, const bool equal_nan,
                                                    bool *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t IsClose<double>(size_t size, const double *inputx, const double *inputy,
                                                     const float atol, const float rtol, const bool equal_nan,
                                                     bool *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t IsClose<int8_t>(size_t size, const int8_t *inputx, const int8_t *inputy,
                                                     const float atol, const float rtol, const bool equal_nan,
                                                     bool *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t IsClose<int16_t>(size_t size, const int16_t *inputx, const int16_t *inputy,
                                                      const float atol, const float rtol, const bool equal_nan,
                                                      bool *output, const uint32_t &device_id,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t IsClose<int32_t>(size_t size, const int32_t *inputx, const int32_t *inputy,
                                                      const float atol, const float rtol, const bool equal_nan,
                                                      bool *output, const uint32_t &device_id,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t IsClose<int64_t>(size_t size, const int64_t *inputx, const int64_t *inputy,
                                                      const float atol, const float rtol, const bool equal_nan,
                                                      bool *output, const uint32_t &device_id,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t IsClose<uint8_t>(size_t size, const uint8_t *inputx, const uint8_t *inputy,
                                                      const float atol, const float rtol, const bool equal_nan,
                                                      bool *output, const uint32_t &device_id,
                                                      cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t BroadcastIsClose<half>(const std::vector<size_t> &inputx_shape,
                                                            const std::vector<size_t> &inputy_shape,
                                                            const std::vector<size_t> &output_shape, const half *inputx,
                                                            const half *inputy, const float atol, const float rtol,
                                                            const bool equal_nan, bool *output,
                                                            const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastIsClose<float>(const std::vector<size_t> &inputx_shape,
                                                             const std::vector<size_t> &inputy_shape,
                                                             const std::vector<size_t> &output_shape,
                                                             const float *inputx, const float *inputy, const float atol,
                                                             const float rtol, const bool equal_nan, bool *output,
                                                             const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastIsClose<double>(
  const std::vector<size_t> &inputx_shape, const std::vector<size_t> &inputy_shape,
  const std::vector<size_t> &output_shape, const double *inputx, const double *inputy, const float atol,
  const float rtol, const bool equal_nan, bool *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastIsClose<int8_t>(
  const std::vector<size_t> &inputx_shape, const std::vector<size_t> &inputy_shape,
  const std::vector<size_t> &output_shape, const int8_t *inputx, const int8_t *inputy, const float atol,
  const float rtol, const bool equal_nan, bool *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastIsClose<int16_t>(
  const std::vector<size_t> &inputx_shape, const std::vector<size_t> &inputy_shape,
  const std::vector<size_t> &output_shape, const int16_t *inputx, const int16_t *inputy, const float atol,
  const float rtol, const bool equal_nan, bool *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastIsClose<int32_t>(
  const std::vector<size_t> &inputx_shape, const std::vector<size_t> &inputy_shape,
  const std::vector<size_t> &output_shape, const int32_t *inputx, const int32_t *inputy, const float atol,
  const float rtol, const bool equal_nan, bool *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastIsClose<int64_t>(
  const std::vector<size_t> &inputx_shape, const std::vector<size_t> &inputy_shape,
  const std::vector<size_t> &output_shape, const int64_t *inputx, const int64_t *inputy, const float atol,
  const float rtol, const bool equal_nan, bool *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastIsClose<uint8_t>(
  const std::vector<size_t> &inputx_shape, const std::vector<size_t> &inputy_shape,
  const std::vector<size_t> &output_shape, const uint8_t *inputx, const uint8_t *inputy, const float atol,
  const float rtol, const bool equal_nan, bool *output, const uint32_t &device_id, cudaStream_t cuda_stream);
