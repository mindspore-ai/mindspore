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

#include <complex.h>
#include "tril_triu_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

template <typename R>
using Complex = mindspore::utils::Complex<R>;

template <typename T>
__global__ void Tril(const size_t size, const T *input, const int diagonal, const int64_t matrix_row,
                     const int64_t matrix_col, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int matrix_size = matrix_row * matrix_col;
    int row = pos % matrix_size / matrix_col;
    int col = pos % matrix_size % matrix_col;
    output[pos] = row + diagonal >= col ? input[pos] : static_cast<T>(0.0);
  }
  return;
}

template <typename T>
__global__ void Triu(const size_t size, const T *input, const int diagonal, const int64_t matrix_row,
                     const int64_t matrix_col, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int matrix_size = matrix_row * matrix_col;
    int row = pos % matrix_size / matrix_col;
    int col = pos % matrix_size % matrix_col;
    output[pos] = row + diagonal <= col ? input[pos] : static_cast<T>(0.0);
  }
  return;
}

template <>
__global__ void Triu(const size_t size, const Complex<float> *input, const int diagonal, const int64_t matrix_row,
                     const int64_t matrix_col, Complex<float> *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int matrix_size = matrix_row * matrix_col;
    int row = pos % matrix_size / matrix_col;
    int col = pos % matrix_size % matrix_col;
    float rs_real = row + diagonal <= col ? input[pos].real() : static_cast<float>(0.0);
    float rs_imag = row + diagonal <= col ? input[pos].imag() : static_cast<float>(0.0);
    output[pos].real(rs_real);
    output[pos].imag(rs_imag);
  }
  return;
}

template <>
__global__ void Triu(const size_t size, const Complex<double> *input, const int diagonal, const int64_t matrix_row,
                     const int64_t matrix_col, Complex<double> *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int matrix_size = matrix_row * matrix_col;
    int row = pos % matrix_size / matrix_col;
    int col = pos % matrix_size % matrix_col;
    double rs_real = row + diagonal <= col ? input[pos].real() : static_cast<double>(0.0);
    double rs_imag = row + diagonal <= col ? input[pos].imag() : static_cast<double>(0.0);
    output[pos].real(rs_real);
    output[pos].imag(rs_imag);
  }
  return;
}

template <typename T>
void CalTril(const size_t size, const T *input, const int diagonal, const int64_t matrix_row, const int64_t matrix_col,
             T *output, const uint32_t &device_id, cudaStream_t cuda_stream) {
  Tril<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input, diagonal, matrix_row,
                                                                                  matrix_col, output);
  return;
}

template <typename T>
void CalTriu(const size_t size, const T *input, const int diagonal, const int64_t matrix_row, const int64_t matrix_col,
             T *output, const uint32_t &device_id, cudaStream_t cuda_stream) {
  Triu<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input, diagonal, matrix_row,
                                                                                  matrix_col, output);
  return;
}

template
CUDA_LIB_EXPORT void CalTril<uint8_t>(const size_t size, const uint8_t *input, const int diagonal,
                                      const int64_t matrix_row, const int64_t matrix_col, uint8_t *output,
                                      const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalTril<uint16_t>(const size_t size, const uint16_t *input, const int diagonal,
                                       const int64_t matrix_row, const int64_t matrix_col, uint16_t *output,
                                       const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalTril<uint32_t>(const size_t size, const uint32_t *input, const int diagonal,
                                       const int64_t matrix_row, const int64_t matrix_col, uint32_t *output,
                                       const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalTril<uint64_t>(const size_t size, const uint64_t *input, const int diagonal,
                                       const int64_t matrix_row, const int64_t matrix_col, uint64_t *output,
                                       const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalTril<int8_t>(const size_t size, const int8_t *input, const int diagonal,
                                     const int64_t matrix_row, const int64_t matrix_col, int8_t *output,
                                     const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalTril<int16_t>(const size_t size, const int16_t *input, const int diagonal,
                                      const int64_t matrix_row, const int64_t matrix_col, int16_t *output,
                                      const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalTril<int>(const size_t size, const int *input, const int diagonal,
                                  const int64_t matrix_row, const int64_t matrix_col, int *output,
                                  const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalTril<int64_t>(const size_t size, const int64_t *input, const int diagonal,
                                      const int64_t matrix_row, const int64_t matrix_col, int64_t *output,
                                      const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalTril<half>(const size_t size, const half *input, const int diagonal,
                                   const int64_t matrix_row, const int64_t matrix_col, half *output,
                                   const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalTril<float>(const size_t size, const float *input, const int diagonal,
                                    const int64_t matrix_row, const int64_t matrix_col, float *output,
                                    const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalTril<double>(const size_t size, const double *input, const int diagonal,
                                     const int64_t matrix_row, const int64_t matrix_col, double *output,
                                     const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalTril<bool>(const size_t size, const bool *input, const int diagonal,
                                   const int64_t matrix_row, const int64_t matrix_col, bool *output,
                                   const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalTriu<uint8_t>(const size_t size, const uint8_t *input, const int diagonal,
                                      const int64_t matrix_row, const int64_t matrix_col, uint8_t *output,
                                      const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalTriu<uint16_t>(const size_t size, const uint16_t *input, const int diagonal,
                                       const int64_t matrix_row, const int64_t matrix_col, uint16_t *output,
                                       const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalTriu<uint32_t>(const size_t size, const uint32_t *input, const int diagonal,
                                       const int64_t matrix_row, const int64_t matrix_col, uint32_t *output,
                                       const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalTriu<uint64_t>(const size_t size, const uint64_t *input, const int diagonal,
                                       const int64_t matrix_row, const int64_t matrix_col, uint64_t *output,
                                       const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalTriu<int8_t>(const size_t size, const int8_t *input, const int diagonal,
                                     const int64_t matrix_row, const int64_t matrix_col, int8_t *output,
                                     const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalTriu<int16_t>(const size_t size, const int16_t *input, const int diagonal,
                                      const int64_t matrix_row, const int64_t matrix_col, int16_t *output,
                                      const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalTriu<int>(const size_t size, const int *input, const int diagonal,
                                  const int64_t matrix_row, const int64_t matrix_col, int *output,
                                  const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalTriu<int64_t>(const size_t size, const int64_t *input, const int diagonal,
                                      const int64_t matrix_row, const int64_t matrix_col, int64_t *output,
                                      const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalTriu<half>(const size_t size, const half *input, const int diagonal,
                                   const int64_t matrix_row, const int64_t matrix_col, half *output,
                                   const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalTriu<float>(const size_t size, const float *input, const int diagonal,
                                    const int64_t matrix_row, const int64_t matrix_col, float *output,
                                    const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalTriu<double>(const size_t size, const double *input, const int diagonal,
                                     const int64_t matrix_row, const int64_t matrix_col, double *output,
                                     const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalTriu<Complex<float>>(const size_t size, const Complex<float> *input, const int diagonal,
                                             const int64_t matrix_row, const int64_t matrix_col, Complex<float> *output,
                                             const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalTriu<Complex<double>>(const size_t size, const Complex<double> *input, const int diagonal,
                                              const int64_t matrix_row, const int64_t matrix_col,
                                              Complex<double> *output, const uint32_t &device_id,
                                              cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalTriu<bool>(const size_t size, const bool *input, const int diagonal,
                                   const int64_t matrix_row, const int64_t matrix_col, bool *output,
                                   const uint32_t &device_id, cudaStream_t cuda_stream);
