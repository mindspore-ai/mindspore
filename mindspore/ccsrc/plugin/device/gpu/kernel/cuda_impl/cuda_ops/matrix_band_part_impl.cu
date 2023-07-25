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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/broadcast_to_impl.cuh"

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
__global__ void MatrixBandPartKernelBroadcast(const size_t dim_size, const size_t size,
                                              TrinaryBroadcastStrideInfo strides, const T *x_ptr, const size_t m,
                                              const size_t n, const LU *lower_ptr, const LU *upper_ptr, T *output_ptr) {
  auto zero = static_cast<T>(0.0);
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int64_t cur_out_idx = 0;
    size_t cur_pos = pos;
    size_t x_index = 0;
    size_t l_index = 0;
    size_t u_index = 0;
    for (int idx = 0; idx < dim_size; ++idx) {
      cur_out_idx = cur_pos / strides.out_stride[idx];
      x_index += cur_out_idx * strides.in0_stride[idx];
      l_index += cur_out_idx * strides.in1_stride[idx];
      u_index += cur_out_idx * strides.in2_stride[idx];
      cur_pos -= cur_out_idx * strides.out_stride[idx];
    }
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
cudaError_t MatrixBandPart(const size_t output_outer_size, const T *x_ptr, const size_t m, const size_t n,
                           const int64_t lower, const int64_t upper, T *output_ptr, const uint32_t &device_id,
                           cudaStream_t cuda_stream) {
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
  return GetCudaStatus();
}

template <typename T, typename LU>
cudaError_t MatrixBandPartBroadcast(const size_t output_element_num, const std::vector<int64_t> &broadcast_x_shape,
                                    const std::vector<int64_t> &broadcast_lower_shape,
                                    const std::vector<int64_t> &broadcast_upper_shape,
                                    const std::vector<int64_t> &broadcast_output_shape, const T *x_ptr, const size_t m,
                                    const size_t n, const LU *lower_ptr, const LU *upper_ptr, T *output_ptr,
                                    const uint32_t &device_id, cudaStream_t cuda_stream) {
  const size_t dim_size = broadcast_output_shape.size();
  TrinaryBroadcastStrideInfo strides = TrinaryBroadcastCalStride(dim_size, broadcast_x_shape, broadcast_lower_shape,
                                                                 broadcast_upper_shape, broadcast_output_shape);
  MatrixBandPartKernelBroadcast<<<CUDA_BLOCKS(device_id, output_element_num), CUDA_THREADS(device_id), 0,
                                  cuda_stream>>>(dim_size, output_element_num, strides, x_ptr, m, n, lower_ptr,
                                                 upper_ptr, output_ptr);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t MatrixBandPart<char>(const size_t output_outer_size, const char *x_ptr,
                                                          const size_t m, const size_t n, const int64_t lower,
                                                          const int64_t upper, char *output_ptr,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPart<int16_t>(const size_t output_outer_size, const int16_t *x_ptr,
                                                             const size_t m, const size_t n, const int64_t lower,
                                                             const int64_t upper, int16_t *output_ptr,
                                                             const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPart<int32_t>(const size_t output_outer_size, const int32_t *x_ptr,
                                                             const size_t m, const size_t n, const int64_t lower,
                                                             const int64_t upper, int32_t *output_ptr,
                                                             const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPart<int64_t>(const size_t output_outer_size, const int64_t *x_ptr,
                                                             const size_t m, const size_t n, const int64_t lower,
                                                             const int64_t upper, int64_t *output_ptr,
                                                             const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPart<unsigned char>(const size_t output_outer_size,
                                                                   const unsigned char *x_ptr, const size_t m,
                                                                   const size_t n, const int64_t lower,
                                                                   const int64_t upper, unsigned char *output_ptr,
                                                                   const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPart<uint16_t>(const size_t output_outer_size, const uint16_t *x_ptr,
                                                              const size_t m, const size_t n, const int64_t lower,
                                                              const int64_t upper, uint16_t *output_ptr,
                                                              const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPart<uint32_t>(const size_t output_outer_size, const uint32_t *x_ptr,
                                                              const size_t m, const size_t n, const int64_t lower,
                                                              const int64_t upper, uint32_t *output_ptr,
                                                              const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPart<uint64_t>(const size_t output_outer_size, const uint64_t *x_ptr,
                                                              const size_t m, const size_t n, const int64_t lower,
                                                              const int64_t upper, uint64_t *output_ptr,
                                                              const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPart<half>(const size_t output_outer_size, const half *x_ptr,
                                                          const size_t m, const size_t n, const int64_t lower,
                                                          const int64_t upper, half *output_ptr,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPart<float>(const size_t output_outer_size, const float *x_ptr,
                                                           const size_t m, const size_t n, const int64_t lower,
                                                           const int64_t upper, float *output_ptr,
                                                           const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPart<double>(const size_t output_outer_size, const double *x_ptr,
                                                            const size_t m, const size_t n, const int64_t lower,
                                                            const int64_t upper, double *output_ptr,
                                                            const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPart<Complex<float>>(
  const size_t output_outer_size, const Complex<float> *x_ptr, const size_t m, const size_t n, const int64_t lower,
  const int64_t upper, Complex<float> *output_ptr, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPart<Complex<double>>(
  const size_t output_outer_size, const Complex<double> *x_ptr, const size_t m, const size_t n, const int64_t lower,
  const int64_t upper, Complex<double> *output_ptr, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPart<bool>(const size_t output_outer_size, const bool *x_ptr,
                                                          const size_t m, const size_t n, const int64_t lower,
                                                          const int64_t upper, bool *output_ptr,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t MatrixBandPartBroadcast<char, int32_t>(
  const size_t output_element_num, const std::vector<int64_t> &broadcast_x_shape,
  const std::vector<int64_t> &broadcast_lower_shape, const std::vector<int64_t> &broadcast_upper_shape,
  const std::vector<int64_t> &broadcast_output_shape, const char *x_ptr, const size_t m, const size_t n,
  const int32_t *lower_ptr, const int32_t *upper_ptr, char *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPartBroadcast<int16_t, int32_t>(
  const size_t output_element_num, const std::vector<int64_t> &broadcast_x_shape,
  const std::vector<int64_t> &broadcast_lower_shape, const std::vector<int64_t> &broadcast_upper_shape,
  const std::vector<int64_t> &broadcast_output_shape, const int16_t *x_ptr, const size_t m, const size_t n,
  const int32_t *lower_ptr, const int32_t *upper_ptr, int16_t *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPartBroadcast<int32_t, int32_t>(
  const size_t output_element_num, const std::vector<int64_t> &broadcast_x_shape,
  const std::vector<int64_t> &broadcast_lower_shape, const std::vector<int64_t> &broadcast_upper_shape,
  const std::vector<int64_t> &broadcast_output_shape, const int32_t *x_ptr, const size_t m, const size_t n,
  const int32_t *lower_ptr, const int32_t *upper_ptr, int32_t *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPartBroadcast<int64_t, int32_t>(
  const size_t output_element_num, const std::vector<int64_t> &broadcast_x_shape,
  const std::vector<int64_t> &broadcast_lower_shape, const std::vector<int64_t> &broadcast_upper_shape,
  const std::vector<int64_t> &broadcast_output_shape, const int64_t *x_ptr, const size_t m, const size_t n,
  const int32_t *lower_ptr, const int32_t *upper_ptr, int64_t *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPartBroadcast<unsigned char, int32_t>(
  const size_t output_element_num, const std::vector<int64_t> &broadcast_x_shape,
  const std::vector<int64_t> &broadcast_lower_shape, const std::vector<int64_t> &broadcast_upper_shape,
  const std::vector<int64_t> &broadcast_output_shape, const unsigned char *x_ptr, const size_t m, const size_t n,
  const int32_t *lower_ptr, const int32_t *upper_ptr, unsigned char *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPartBroadcast<uint16_t, int32_t>(
  const size_t output_element_num, const std::vector<int64_t> &broadcast_x_shape,
  const std::vector<int64_t> &broadcast_lower_shape, const std::vector<int64_t> &broadcast_upper_shape,
  const std::vector<int64_t> &broadcast_output_shape, const uint16_t *x_ptr, const size_t m, const size_t n,
  const int32_t *lower_ptr, const int32_t *upper_ptr, uint16_t *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPartBroadcast<uint32_t, int32_t>(
  const size_t output_element_num, const std::vector<int64_t> &broadcast_x_shape,
  const std::vector<int64_t> &broadcast_lower_shape, const std::vector<int64_t> &broadcast_upper_shape,
  const std::vector<int64_t> &broadcast_output_shape, const uint32_t *x_ptr, const size_t m, const size_t n,
  const int32_t *lower_ptr, const int32_t *upper_ptr, uint32_t *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPartBroadcast<uint64_t, int32_t>(
  const size_t output_element_num, const std::vector<int64_t> &broadcast_x_shape,
  const std::vector<int64_t> &broadcast_lower_shape, const std::vector<int64_t> &broadcast_upper_shape,
  const std::vector<int64_t> &broadcast_output_shape, const uint64_t *x_ptr, const size_t m, const size_t n,
  const int32_t *lower_ptr, const int32_t *upper_ptr, uint64_t *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPartBroadcast<half, int32_t>(
  const size_t output_element_num, const std::vector<int64_t> &broadcast_x_shape,
  const std::vector<int64_t> &broadcast_lower_shape, const std::vector<int64_t> &broadcast_upper_shape,
  const std::vector<int64_t> &broadcast_output_shape, const half *x_ptr, const size_t m, const size_t n,
  const int32_t *lower_ptr, const int32_t *upper_ptr, half *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPartBroadcast<float, int32_t>(
  const size_t output_element_num, const std::vector<int64_t> &broadcast_x_shape,
  const std::vector<int64_t> &broadcast_lower_shape, const std::vector<int64_t> &broadcast_upper_shape,
  const std::vector<int64_t> &broadcast_output_shape, const float *x_ptr, const size_t m, const size_t n,
  const int32_t *lower_ptr, const int32_t *upper_ptr, float *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPartBroadcast<double, int32_t>(
  const size_t output_element_num, const std::vector<int64_t> &broadcast_x_shape,
  const std::vector<int64_t> &broadcast_lower_shape, const std::vector<int64_t> &broadcast_upper_shape,
  const std::vector<int64_t> &broadcast_output_shape, const double *x_ptr, const size_t m, const size_t n,
  const int32_t *lower_ptr, const int32_t *upper_ptr, double *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPartBroadcast<Complex<float>, int32_t>(
  const size_t output_element_num, const std::vector<int64_t> &broadcast_x_shape,
  const std::vector<int64_t> &broadcast_lower_shape, const std::vector<int64_t> &broadcast_upper_shape,
  const std::vector<int64_t> &broadcast_output_shape, const Complex<float> *x_ptr, const size_t m, const size_t n,
  const int32_t *lower_ptr, const int32_t *upper_ptr, Complex<float> *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPartBroadcast<Complex<double>, int32_t>(
  const size_t output_element_num, const std::vector<int64_t> &broadcast_x_shape,
  const std::vector<int64_t> &broadcast_lower_shape, const std::vector<int64_t> &broadcast_upper_shape,
  const std::vector<int64_t> &broadcast_output_shape, const Complex<double> *x_ptr, const size_t m, const size_t n,
  const int32_t *lower_ptr, const int32_t *upper_ptr, Complex<double> *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPartBroadcast<bool, int32_t>(
  const size_t output_element_num, const std::vector<int64_t> &broadcast_x_shape,
  const std::vector<int64_t> &broadcast_lower_shape, const std::vector<int64_t> &broadcast_upper_shape,
  const std::vector<int64_t> &broadcast_output_shape, const bool *x_ptr, const size_t m, const size_t n,
  const int32_t *lower_ptr, const int32_t *upper_ptr, bool *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t MatrixBandPartBroadcast<char, int64_t>(
  const size_t output_element_num, const std::vector<int64_t> &broadcast_x_shape,
  const std::vector<int64_t> &broadcast_lower_shape, const std::vector<int64_t> &broadcast_upper_shape,
  const std::vector<int64_t> &broadcast_output_shape, const char *x_ptr, const size_t m, const size_t n,
  const int64_t *lower_ptr, const int64_t *upper_ptr, char *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPartBroadcast<int16_t, int64_t>(
  const size_t output_element_num, const std::vector<int64_t> &broadcast_x_shape,
  const std::vector<int64_t> &broadcast_lower_shape, const std::vector<int64_t> &broadcast_upper_shape,
  const std::vector<int64_t> &broadcast_output_shape, const int16_t *x_ptr, const size_t m, const size_t n,
  const int64_t *lower_ptr, const int64_t *upper_ptr, int16_t *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPartBroadcast<int32_t, int64_t>(
  const size_t output_element_num, const std::vector<int64_t> &broadcast_x_shape,
  const std::vector<int64_t> &broadcast_lower_shape, const std::vector<int64_t> &broadcast_upper_shape,
  const std::vector<int64_t> &broadcast_output_shape, const int32_t *x_ptr, const size_t m, const size_t n,
  const int64_t *lower_ptr, const int64_t *upper_ptr, int32_t *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPartBroadcast<int64_t, int64_t>(
  const size_t output_element_num, const std::vector<int64_t> &broadcast_x_shape,
  const std::vector<int64_t> &broadcast_lower_shape, const std::vector<int64_t> &broadcast_upper_shape,
  const std::vector<int64_t> &broadcast_output_shape, const int64_t *x_ptr, const size_t m, const size_t n,
  const int64_t *lower_ptr, const int64_t *upper_ptr, int64_t *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPartBroadcast<unsigned char, int64_t>(
  const size_t output_element_num, const std::vector<int64_t> &broadcast_x_shape,
  const std::vector<int64_t> &broadcast_lower_shape, const std::vector<int64_t> &broadcast_upper_shape,
  const std::vector<int64_t> &broadcast_output_shape, const unsigned char *x_ptr, const size_t m, const size_t n,
  const int64_t *lower_ptr, const int64_t *upper_ptr, unsigned char *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPartBroadcast<uint16_t, int64_t>(
  const size_t output_element_num, const std::vector<int64_t> &broadcast_x_shape,
  const std::vector<int64_t> &broadcast_lower_shape, const std::vector<int64_t> &broadcast_upper_shape,
  const std::vector<int64_t> &broadcast_output_shape, const uint16_t *x_ptr, const size_t m, const size_t n,
  const int64_t *lower_ptr, const int64_t *upper_ptr, uint16_t *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPartBroadcast<uint32_t, int64_t>(
  const size_t output_element_num, const std::vector<int64_t> &broadcast_x_shape,
  const std::vector<int64_t> &broadcast_lower_shape, const std::vector<int64_t> &broadcast_upper_shape,
  const std::vector<int64_t> &broadcast_output_shape, const uint32_t *x_ptr, const size_t m, const size_t n,
  const int64_t *lower_ptr, const int64_t *upper_ptr, uint32_t *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPartBroadcast<uint64_t, int64_t>(
  const size_t output_element_num, const std::vector<int64_t> &broadcast_x_shape,
  const std::vector<int64_t> &broadcast_lower_shape, const std::vector<int64_t> &broadcast_upper_shape,
  const std::vector<int64_t> &broadcast_output_shape, const uint64_t *x_ptr, const size_t m, const size_t n,
  const int64_t *lower_ptr, const int64_t *upper_ptr, uint64_t *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPartBroadcast<half, int64_t>(
  const size_t output_element_num, const std::vector<int64_t> &broadcast_x_shape,
  const std::vector<int64_t> &broadcast_lower_shape, const std::vector<int64_t> &broadcast_upper_shape,
  const std::vector<int64_t> &broadcast_output_shape, const half *x_ptr, const size_t m, const size_t n,
  const int64_t *lower_ptr, const int64_t *upper_ptr, half *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPartBroadcast<float, int64_t>(
  const size_t output_element_num, const std::vector<int64_t> &broadcast_x_shape,
  const std::vector<int64_t> &broadcast_lower_shape, const std::vector<int64_t> &broadcast_upper_shape,
  const std::vector<int64_t> &broadcast_output_shape, const float *x_ptr, const size_t m, const size_t n,
  const int64_t *lower_ptr, const int64_t *upper_ptr, float *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPartBroadcast<double, int64_t>(
  const size_t output_element_num, const std::vector<int64_t> &broadcast_x_shape,
  const std::vector<int64_t> &broadcast_lower_shape, const std::vector<int64_t> &broadcast_upper_shape,
  const std::vector<int64_t> &broadcast_output_shape, const double *x_ptr, const size_t m, const size_t n,
  const int64_t *lower_ptr, const int64_t *upper_ptr, double *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPartBroadcast<Complex<float>, int64_t>(
  const size_t output_element_num, const std::vector<int64_t> &broadcast_x_shape,
  const std::vector<int64_t> &broadcast_lower_shape, const std::vector<int64_t> &broadcast_upper_shape,
  const std::vector<int64_t> &broadcast_output_shape, const Complex<float> *x_ptr, const size_t m, const size_t n,
  const int64_t *lower_ptr, const int64_t *upper_ptr, Complex<float> *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPartBroadcast<Complex<double>, int64_t>(
  const size_t output_element_num, const std::vector<int64_t> &broadcast_x_shape,
  const std::vector<int64_t> &broadcast_lower_shape, const std::vector<int64_t> &broadcast_upper_shape,
  const std::vector<int64_t> &broadcast_output_shape, const Complex<double> *x_ptr, const size_t m, const size_t n,
  const int64_t *lower_ptr, const int64_t *upper_ptr, Complex<double> *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixBandPartBroadcast<bool, int64_t>(
  const size_t output_element_num, const std::vector<int64_t> &broadcast_x_shape,
  const std::vector<int64_t> &broadcast_lower_shape, const std::vector<int64_t> &broadcast_upper_shape,
  const std::vector<int64_t> &broadcast_output_shape, const bool *x_ptr, const size_t m, const size_t n,
  const int64_t *lower_ptr, const int64_t *upper_ptr, bool *output_ptr, const uint32_t &device_id,
  cudaStream_t cuda_stream);
