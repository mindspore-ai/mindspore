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
#include "matrix_band_part_impl.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T>
__global__ void MatrixBandPartDiagonalKernel(const size_t size, const T *x_ptr, const size_t non_zero_len,
                                             const size_t m, const size_t n, T *output_ptr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    const size_t i = pos / non_zero_len;
    const size_t j = pos % non_zero_len;
    const size_t offset = i * m * n + j * n;
    // Diagonal
    output_ptr[offset + j] = x_ptr[offset + j];
  }
}

template <typename T>
__global__ void MatrixBandPartKernel(const int size, const T *x_ptr, const int m, const int n, const int lower,
                                     const int upper, T *output_ptr) {
  int start_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  int step = static_cast<int>(blockDim.x * gridDim.x);
  for (int pos = start_idx; pos < size; pos += step) {
    const int last_two_dim_offset = pos % (m * n);
    const int i = last_two_dim_offset / n;
    const int j = last_two_dim_offset % n;
    // Note: the type of i or j can not be size_t.
    if ((i - j) > lower || (j - i) > upper) {
      output_ptr[pos] = static_cast<T>(0.0);
    }
  }
}

__device__ __forceinline__ size_t Index(const size_t &index, const size_t &dim) { return dim == 1 ? 0 : index; }

template <typename T, typename LU>
__global__ void MatrixBandPartKernelBroadcast(const size_t size, size_t x0, size_t x1, size_t x2, size_t x3, size_t x4,
                                              size_t x5, size_t x6, size_t x7, size_t l0, size_t l1, size_t l2,
                                              size_t l3, size_t l4, size_t l5, size_t l6, size_t l7, size_t u0,
                                              size_t u1, size_t u2, size_t u3, size_t u4, size_t u5, size_t u6,
                                              size_t u7, size_t o0, size_t o1, size_t o2, size_t o3, size_t o4,
                                              size_t o5, size_t o6, size_t o7, const T *x_ptr, const size_t m,
                                              const size_t n, const LU *lower_ptr, const LU *upper_ptr, T *output_ptr) {
  auto zero = static_cast<T>(0.0);
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    size_t i = pos / (o1 * o2 * o3 * o4 * o5 * o6 * o7) % o0;
    size_t j = pos / (o2 * o3 * o4 * o5 * o6 * o7) % o1;
    size_t k = pos / (o3 * o4 * o5 * o6 * o7) % o2;
    size_t l = pos / (o4 * o5 * o6 * o7) % o3;
    size_t mm = pos / (o5 * o6 * o7) % o4;
    size_t nn = pos / (o6 * o7) % o5;
    size_t o = pos / o7 % o6;
    size_t p = pos % o7;

    size_t x_index = Index(i, x0) * x1 * x2 * x3 * x4 * x5 * x6 * x7;
    x_index += Index(j, x1) * x2 * x3 * x4 * x5 * x6 * x7;
    x_index += Index(k, x2) * x3 * x4 * x5 * x6 * x7;
    x_index += Index(l, x3) * x4 * x5 * x6 * x7;
    x_index += Index(mm, x4) * x5 * x6 * x7;
    x_index += Index(nn, x5) * x6 * x7;
    x_index += Index(o, x6) * x7;
    x_index += Index(p, x7);

    size_t l_index = Index(i, l0) * l1 * l2 * l3 * l4 * l5 * l6 * l7;
    l_index += Index(j, l1) * l2 * l3 * l4 * l5 * l6 * l7;
    l_index += Index(k, l2) * l3 * l4 * l5 * l6 * l7;
    l_index += Index(l, l3) * l4 * l5 * l6 * l7;
    l_index += Index(mm, l4) * l5 * l6 * l7;
    l_index += Index(nn, l5) * l6 * l7;
    l_index += Index(o, l6) * l7;
    l_index += Index(p, l7);

    size_t u_index = Index(i, u0) * u1 * u2 * u3 * u4 * u5 * u6 * u7;
    u_index += Index(j, u1) * u2 * u3 * u4 * u5 * u6 * u7;
    u_index += Index(k, u2) * u3 * u4 * u5 * u6 * u7;
    u_index += Index(l, u3) * u4 * u5 * u6 * u7;
    u_index += Index(mm, u4) * u5 * u6 * u7;
    u_index += Index(nn, u5) * u6 * u7;
    u_index += Index(o, u6) * u7;
    u_index += Index(p, u7);

    const size_t last_two_dim_offset = pos % (m * n);
    int64_t ii = static_cast<int64_t>(last_two_dim_offset / n);
    int64_t jj = static_cast<int64_t>(last_two_dim_offset % n);
    auto lower = lower_ptr[l_index];
    auto upper = upper_ptr[u_index];
    // Note: the type of ii or jj can not be size_t.
    if ((lower < 0 || (ii - jj) <= lower) && (upper < 0 || (jj - ii) <= upper)) {
      output_ptr[pos] = x_ptr[x_index];
    } else {
      output_ptr[pos] = zero;
    }
  }
}

template <typename T>
void MatrixBandPart(const size_t output_outer_size, const T *x_ptr, const size_t m, const size_t n, const int64_t lower,
                    const int64_t upper, T *output_ptr, const uint32_t &device_id, cudaStream_t cuda_stream) {
  if (lower == 0 && upper == 0) {
    // The non_zero_len is the length of the non zero element along the -2 axis, so it can skip the position with 0.
    size_t non_zero_len = std::min(m, lower + n);
    auto size = output_outer_size * non_zero_len;
    MatrixBandPartDiagonalKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      size, x_ptr, non_zero_len, m, n, output_ptr);
  } else {
    auto size = output_outer_size * m * n;
    MatrixBandPartKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      static_cast<int>(size), x_ptr, static_cast<int>(m), static_cast<int>(n), static_cast<int>(lower),
      static_cast<int>(upper), output_ptr);
  }
}

template <typename T, typename LU>
void MatrixBandPartBroadcast(const size_t output_element_num, const std::vector<size_t> &broadcast_x_shape,
                             const std::vector<size_t> &broadcast_lower_shape,
                             const std::vector<size_t> &broadcast_upper_shape,
                             const std::vector<size_t> &broadcast_output_shape, const T *x_ptr, const size_t m,
                             const size_t n, const LU *lower_ptr, const LU *upper_ptr, T *output_ptr,
                             const uint32_t &device_id, cudaStream_t cuda_stream) {
  MatrixBandPartKernelBroadcast<<<CUDA_BLOCKS(device_id, output_element_num), CUDA_THREADS(device_id), 0,
                                  cuda_stream>>>(
    output_element_num, broadcast_x_shape[0], broadcast_x_shape[1], broadcast_x_shape[2], broadcast_x_shape[3],
    broadcast_x_shape[4], broadcast_x_shape[5], broadcast_x_shape[6], broadcast_x_shape[7], broadcast_lower_shape[0],
    broadcast_lower_shape[1], broadcast_lower_shape[2], broadcast_lower_shape[3], broadcast_lower_shape[4],
    broadcast_lower_shape[5], broadcast_lower_shape[6], broadcast_lower_shape[7], broadcast_upper_shape[0],
    broadcast_upper_shape[1], broadcast_upper_shape[2], broadcast_upper_shape[3], broadcast_upper_shape[4],
    broadcast_upper_shape[5], broadcast_upper_shape[6], broadcast_upper_shape[7], broadcast_output_shape[0],
    broadcast_output_shape[1], broadcast_output_shape[2], broadcast_output_shape[3], broadcast_output_shape[4],
    broadcast_output_shape[5], broadcast_output_shape[6], broadcast_output_shape[7], x_ptr, m, n, lower_ptr, upper_ptr,
    output_ptr);
}

template CUDA_LIB_EXPORT void MatrixBandPart<char>(const size_t output_outer_size, const char *x_ptr, const size_t m,
                                                   const size_t n, const int64_t lower, const int64_t upper,
                                                   char *output_ptr, const uint32_t &device_id,
                                                   cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPart<int16_t>(const size_t output_outer_size, const int16_t *x_ptr,
                                                      const size_t m, const size_t n, const int64_t lower,
                                                      const int64_t upper, int16_t *output_ptr,
                                                      const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPart<int32_t>(const size_t output_outer_size, const int32_t *x_ptr,
                                                      const size_t m, const size_t n, const int64_t lower,
                                                      const int64_t upper, int32_t *output_ptr,
                                                      const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPart<int64_t>(const size_t output_outer_size, const int64_t *x_ptr,
                                                      const size_t m, const size_t n, const int64_t lower,
                                                      const int64_t upper, int64_t *output_ptr,
                                                      const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPart<unsigned char>(const size_t output_outer_size, const unsigned char *x_ptr,
                                                            const size_t m, const size_t n, const int64_t lower,
                                                            const int64_t upper, unsigned char *output_ptr,
                                                            const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPart<uint16_t>(const size_t output_outer_size, const uint16_t *x_ptr,
                                                       const size_t m, const size_t n, const int64_t lower,
                                                       const int64_t upper, uint16_t *output_ptr,
                                                       const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPart<uint32_t>(const size_t output_outer_size, const uint32_t *x_ptr,
                                                       const size_t m, const size_t n, const int64_t lower,
                                                       const int64_t upper, uint32_t *output_ptr,
                                                       const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPart<uint64_t>(const size_t output_outer_size, const uint64_t *x_ptr,
                                                       const size_t m, const size_t n, const int64_t lower,
                                                       const int64_t upper, uint64_t *output_ptr,
                                                       const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPart<half>(const size_t output_outer_size, const half *x_ptr, const size_t m,
                                                   const size_t n, const int64_t lower, const int64_t upper,
                                                   half *output_ptr, const uint32_t &device_id,
                                                   cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPart<float>(const size_t output_outer_size, const float *x_ptr, const size_t m,
                                                    const size_t n, const int64_t lower, const int64_t upper,
                                                    float *output_ptr, const uint32_t &device_id,
                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPart<double>(const size_t output_outer_size, const double *x_ptr,
                                                     const size_t m, const size_t n, const int64_t lower,
                                                     const int64_t upper, double *output_ptr, const uint32_t &device_id,
                                                     cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPart<Complex<float>>(const size_t output_outer_size,
                                                             const Complex<float> *x_ptr, const size_t m,
                                                             const size_t n, const int64_t lower, const int64_t upper,
                                                             Complex<float> *output_ptr, const uint32_t &device_id,
                                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPart<Complex<double>>(const size_t output_outer_size,
                                                              const Complex<double> *x_ptr, const size_t m,
                                                              const size_t n, const int64_t lower, const int64_t upper,
                                                              Complex<double> *output_ptr, const uint32_t &device_id,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPart<bool>(const size_t output_outer_size, const bool *x_ptr, const size_t m,
                                                   const size_t n, const int64_t lower, const int64_t upper,
                                                   bool *output_ptr, const uint32_t &device_id,
                                                   cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void MatrixBandPartBroadcast<char, int32_t>(
  const size_t output_element_num, const std::vector<size_t> &broadcast_x_shape,
  const std::vector<size_t> &broadcast_lower_shape, const std::vector<size_t> &broadcast_upper_shape,
  const std::vector<size_t> &broadcast_output_shape, const char *x_ptr, const size_t m, const size_t n,
  const int32_t *lower_ptr, const int32_t *upper_ptr, char *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPartBroadcast<int16_t, int32_t>(
  const size_t output_element_num, const std::vector<size_t> &broadcast_x_shape,
  const std::vector<size_t> &broadcast_lower_shape, const std::vector<size_t> &broadcast_upper_shape,
  const std::vector<size_t> &broadcast_output_shape, const int16_t *x_ptr, const size_t m, const size_t n,
  const int32_t *lower_ptr, const int32_t *upper_ptr, int16_t *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPartBroadcast<int32_t, int32_t>(
  const size_t output_element_num, const std::vector<size_t> &broadcast_x_shape,
  const std::vector<size_t> &broadcast_lower_shape, const std::vector<size_t> &broadcast_upper_shape,
  const std::vector<size_t> &broadcast_output_shape, const int32_t *x_ptr, const size_t m, const size_t n,
  const int32_t *lower_ptr, const int32_t *upper_ptr, int32_t *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPartBroadcast<int64_t, int32_t>(
  const size_t output_element_num, const std::vector<size_t> &broadcast_x_shape,
  const std::vector<size_t> &broadcast_lower_shape, const std::vector<size_t> &broadcast_upper_shape,
  const std::vector<size_t> &broadcast_output_shape, const int64_t *x_ptr, const size_t m, const size_t n,
  const int32_t *lower_ptr, const int32_t *upper_ptr, int64_t *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPartBroadcast<unsigned char, int32_t>(
  const size_t output_element_num, const std::vector<size_t> &broadcast_x_shape,
  const std::vector<size_t> &broadcast_lower_shape, const std::vector<size_t> &broadcast_upper_shape,
  const std::vector<size_t> &broadcast_output_shape, const unsigned char *x_ptr, const size_t m, const size_t n,
  const int32_t *lower_ptr, const int32_t *upper_ptr, unsigned char *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPartBroadcast<uint16_t, int32_t>(
  const size_t output_element_num, const std::vector<size_t> &broadcast_x_shape,
  const std::vector<size_t> &broadcast_lower_shape, const std::vector<size_t> &broadcast_upper_shape,
  const std::vector<size_t> &broadcast_output_shape, const uint16_t *x_ptr, const size_t m, const size_t n,
  const int32_t *lower_ptr, const int32_t *upper_ptr, uint16_t *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPartBroadcast<uint32_t, int32_t>(
  const size_t output_element_num, const std::vector<size_t> &broadcast_x_shape,
  const std::vector<size_t> &broadcast_lower_shape, const std::vector<size_t> &broadcast_upper_shape,
  const std::vector<size_t> &broadcast_output_shape, const uint32_t *x_ptr, const size_t m, const size_t n,
  const int32_t *lower_ptr, const int32_t *upper_ptr, uint32_t *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPartBroadcast<uint64_t, int32_t>(
  const size_t output_element_num, const std::vector<size_t> &broadcast_x_shape,
  const std::vector<size_t> &broadcast_lower_shape, const std::vector<size_t> &broadcast_upper_shape,
  const std::vector<size_t> &broadcast_output_shape, const uint64_t *x_ptr, const size_t m, const size_t n,
  const int32_t *lower_ptr, const int32_t *upper_ptr, uint64_t *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPartBroadcast<half, int32_t>(
  const size_t output_element_num, const std::vector<size_t> &broadcast_x_shape,
  const std::vector<size_t> &broadcast_lower_shape, const std::vector<size_t> &broadcast_upper_shape,
  const std::vector<size_t> &broadcast_output_shape, const half *x_ptr, const size_t m, const size_t n,
  const int32_t *lower_ptr, const int32_t *upper_ptr, half *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPartBroadcast<float, int32_t>(
  const size_t output_element_num, const std::vector<size_t> &broadcast_x_shape,
  const std::vector<size_t> &broadcast_lower_shape, const std::vector<size_t> &broadcast_upper_shape,
  const std::vector<size_t> &broadcast_output_shape, const float *x_ptr, const size_t m, const size_t n,
  const int32_t *lower_ptr, const int32_t *upper_ptr, float *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPartBroadcast<double, int32_t>(
  const size_t output_element_num, const std::vector<size_t> &broadcast_x_shape,
  const std::vector<size_t> &broadcast_lower_shape, const std::vector<size_t> &broadcast_upper_shape,
  const std::vector<size_t> &broadcast_output_shape, const double *x_ptr, const size_t m, const size_t n,
  const int32_t *lower_ptr, const int32_t *upper_ptr, double *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPartBroadcast<Complex<float>, int32_t>(
  const size_t output_element_num, const std::vector<size_t> &broadcast_x_shape,
  const std::vector<size_t> &broadcast_lower_shape, const std::vector<size_t> &broadcast_upper_shape,
  const std::vector<size_t> &broadcast_output_shape, const Complex<float> *x_ptr, const size_t m, const size_t n,
  const int32_t *lower_ptr, const int32_t *upper_ptr, Complex<float> *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPartBroadcast<Complex<double>, int32_t>(
  const size_t output_element_num, const std::vector<size_t> &broadcast_x_shape,
  const std::vector<size_t> &broadcast_lower_shape, const std::vector<size_t> &broadcast_upper_shape,
  const std::vector<size_t> &broadcast_output_shape, const Complex<double> *x_ptr, const size_t m, const size_t n,
  const int32_t *lower_ptr, const int32_t *upper_ptr, Complex<double> *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPartBroadcast<bool, int32_t>(
  const size_t output_element_num, const std::vector<size_t> &broadcast_x_shape,
  const std::vector<size_t> &broadcast_lower_shape, const std::vector<size_t> &broadcast_upper_shape,
  const std::vector<size_t> &broadcast_output_shape, const bool *x_ptr, const size_t m, const size_t n,
  const int32_t *lower_ptr, const int32_t *upper_ptr, bool *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void MatrixBandPartBroadcast<char, int64_t>(
  const size_t output_element_num, const std::vector<size_t> &broadcast_x_shape,
  const std::vector<size_t> &broadcast_lower_shape, const std::vector<size_t> &broadcast_upper_shape,
  const std::vector<size_t> &broadcast_output_shape, const char *x_ptr, const size_t m, const size_t n,
  const int64_t *lower_ptr, const int64_t *upper_ptr, char *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPartBroadcast<int16_t, int64_t>(
  const size_t output_element_num, const std::vector<size_t> &broadcast_x_shape,
  const std::vector<size_t> &broadcast_lower_shape, const std::vector<size_t> &broadcast_upper_shape,
  const std::vector<size_t> &broadcast_output_shape, const int16_t *x_ptr, const size_t m, const size_t n,
  const int64_t *lower_ptr, const int64_t *upper_ptr, int16_t *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPartBroadcast<int32_t, int64_t>(
  const size_t output_element_num, const std::vector<size_t> &broadcast_x_shape,
  const std::vector<size_t> &broadcast_lower_shape, const std::vector<size_t> &broadcast_upper_shape,
  const std::vector<size_t> &broadcast_output_shape, const int32_t *x_ptr, const size_t m, const size_t n,
  const int64_t *lower_ptr, const int64_t *upper_ptr, int32_t *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPartBroadcast<int64_t, int64_t>(
  const size_t output_element_num, const std::vector<size_t> &broadcast_x_shape,
  const std::vector<size_t> &broadcast_lower_shape, const std::vector<size_t> &broadcast_upper_shape,
  const std::vector<size_t> &broadcast_output_shape, const int64_t *x_ptr, const size_t m, const size_t n,
  const int64_t *lower_ptr, const int64_t *upper_ptr, int64_t *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPartBroadcast<unsigned char, int64_t>(
  const size_t output_element_num, const std::vector<size_t> &broadcast_x_shape,
  const std::vector<size_t> &broadcast_lower_shape, const std::vector<size_t> &broadcast_upper_shape,
  const std::vector<size_t> &broadcast_output_shape, const unsigned char *x_ptr, const size_t m, const size_t n,
  const int64_t *lower_ptr, const int64_t *upper_ptr, unsigned char *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPartBroadcast<uint16_t, int64_t>(
  const size_t output_element_num, const std::vector<size_t> &broadcast_x_shape,
  const std::vector<size_t> &broadcast_lower_shape, const std::vector<size_t> &broadcast_upper_shape,
  const std::vector<size_t> &broadcast_output_shape, const uint16_t *x_ptr, const size_t m, const size_t n,
  const int64_t *lower_ptr, const int64_t *upper_ptr, uint16_t *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPartBroadcast<uint32_t, int64_t>(
  const size_t output_element_num, const std::vector<size_t> &broadcast_x_shape,
  const std::vector<size_t> &broadcast_lower_shape, const std::vector<size_t> &broadcast_upper_shape,
  const std::vector<size_t> &broadcast_output_shape, const uint32_t *x_ptr, const size_t m, const size_t n,
  const int64_t *lower_ptr, const int64_t *upper_ptr, uint32_t *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPartBroadcast<uint64_t, int64_t>(
  const size_t output_element_num, const std::vector<size_t> &broadcast_x_shape,
  const std::vector<size_t> &broadcast_lower_shape, const std::vector<size_t> &broadcast_upper_shape,
  const std::vector<size_t> &broadcast_output_shape, const uint64_t *x_ptr, const size_t m, const size_t n,
  const int64_t *lower_ptr, const int64_t *upper_ptr, uint64_t *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPartBroadcast<half, int64_t>(
  const size_t output_element_num, const std::vector<size_t> &broadcast_x_shape,
  const std::vector<size_t> &broadcast_lower_shape, const std::vector<size_t> &broadcast_upper_shape,
  const std::vector<size_t> &broadcast_output_shape, const half *x_ptr, const size_t m, const size_t n,
  const int64_t *lower_ptr, const int64_t *upper_ptr, half *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPartBroadcast<float, int64_t>(
  const size_t output_element_num, const std::vector<size_t> &broadcast_x_shape,
  const std::vector<size_t> &broadcast_lower_shape, const std::vector<size_t> &broadcast_upper_shape,
  const std::vector<size_t> &broadcast_output_shape, const float *x_ptr, const size_t m, const size_t n,
  const int64_t *lower_ptr, const int64_t *upper_ptr, float *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPartBroadcast<double, int64_t>(
  const size_t output_element_num, const std::vector<size_t> &broadcast_x_shape,
  const std::vector<size_t> &broadcast_lower_shape, const std::vector<size_t> &broadcast_upper_shape,
  const std::vector<size_t> &broadcast_output_shape, const double *x_ptr, const size_t m, const size_t n,
  const int64_t *lower_ptr, const int64_t *upper_ptr, double *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPartBroadcast<Complex<float>, int64_t>(
  const size_t output_element_num, const std::vector<size_t> &broadcast_x_shape,
  const std::vector<size_t> &broadcast_lower_shape, const std::vector<size_t> &broadcast_upper_shape,
  const std::vector<size_t> &broadcast_output_shape, const Complex<float> *x_ptr, const size_t m, const size_t n,
  const int64_t *lower_ptr, const int64_t *upper_ptr, Complex<float> *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPartBroadcast<Complex<double>, int64_t>(
  const size_t output_element_num, const std::vector<size_t> &broadcast_x_shape,
  const std::vector<size_t> &broadcast_lower_shape, const std::vector<size_t> &broadcast_upper_shape,
  const std::vector<size_t> &broadcast_output_shape, const Complex<double> *x_ptr, const size_t m, const size_t n,
  const int64_t *lower_ptr, const int64_t *upper_ptr, Complex<double> *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MatrixBandPartBroadcast<bool, int64_t>(
  const size_t output_element_num, const std::vector<size_t> &broadcast_x_shape,
  const std::vector<size_t> &broadcast_lower_shape, const std::vector<size_t> &broadcast_upper_shape,
  const std::vector<size_t> &broadcast_output_shape, const bool *x_ptr, const size_t m, const size_t n,
  const int64_t *lower_ptr, const int64_t *upper_ptr, bool *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
