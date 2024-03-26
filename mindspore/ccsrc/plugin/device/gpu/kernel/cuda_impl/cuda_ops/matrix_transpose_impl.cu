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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/matrix_transpose_impl.cuh"
#include <cuda_runtime.h>
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T>
__global__ void MatrixTransposeKernel(const T *input, int elements, int row, int col, T *output) {
template <typename T>
cudaError_t MatrixTranspose(const T *input, int elements, int row, int col, T *output, uint32_t device_id,
                            cudaStream_t cuda_stream) {
  if (col < 0 || row < 0 ) {
    return cudaErrorInvalidValue;
  }
  MatrixTransposeKernel<<<CUDA_BLOCKS(device_id, elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input, elements, row, col, output);
  return GetCudaStatus();
}
    return;
  }
  const int matrix_size = row * col;
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < elements; pos += blockDim.x * gridDim.x) {
    const int b = pos / matrix_size;
    const int b_stride = b * matrix_size;
    const int r = (pos - b_stride) / col;
    const int c = (pos - b_stride) % col;
    // For output,  new position is  b_stride + c * row + r.
    output[b_stride + c * row + r] = input[pos];
  }
}

template <typename T>
cudaError_t MatrixTranspose(const T *input, int elements, int row, int col, T *output, uint32_t device_id,
                            cudaStream_t cuda_stream) {
  MatrixTransposeKernel<<<CUDA_BLOCKS(device_id, elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input, elements, row, col, output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t MatrixTranspose<bool>(const bool *input, int elements, int row, int col,
                                                           bool *output, uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixTranspose<double>(const double *input, int elements, int row, int col,
                                                             double *output, uint32_t device_id,
                                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixTranspose<float>(const float *input, int elements, int row, int col,
                                                            float *output, uint32_t device_id,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixTranspose<half>(const half *input, int elements, int row, int col,
                                                           half *output, uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixTranspose<int64_t>(const int64_t *input, int elements, int row, int col,
                                                              int64_t *output, uint32_t device_id,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixTranspose<int>(const int *input, int elements, int row, int col, int *output,
                                                          uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixTranspose<int16_t>(const int16_t *input, int elements, int row, int col,
                                                              int16_t *output, uint32_t device_id,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixTranspose<int8_t>(const int8_t *input, int elements, int row, int col,
                                                             int8_t *output, uint32_t device_id,
                                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixTranspose<uint64_t>(const uint64_t *input, int elements, int row, int col,
                                                               uint64_t *output, uint32_t device_id,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixTranspose<uint32_t>(const uint32_t *input, int elements, int row, int col,
                                                               uint32_t *output, uint32_t device_id,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixTranspose<uint16_t>(const uint16_t *input, int elements, int row, int col,
                                                               uint16_t *output, uint32_t device_id,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixTranspose<uint8_t>(const uint8_t *input, int elements, int row, int col,
                                                              uint8_t *output, uint32_t device_id,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixTranspose<Complex<float>>(const Complex<float> *input, int elements, int row,
                                                                     int col, Complex<float> *output,
                                                                     uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MatrixTranspose<Complex<double>>(const Complex<double> *input, int elements,
                                                                      int row, int col, Complex<double> *output,
                                                                      uint32_t device_id, cudaStream_t cuda_stream);
