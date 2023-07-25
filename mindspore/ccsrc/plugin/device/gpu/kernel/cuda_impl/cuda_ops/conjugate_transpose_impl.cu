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
#include <typeinfo>
#include "include/cuda_fp16.h"
#include "conjugate_transpose_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

template <typename U>
using Complex = mindspore::utils::Complex<U>;

template <typename T, typename S>
__global__ void ConjugateTranspose(const size_t size, const T *input, const size_t *input_stride,
                                   const size_t *output_stride, const S *input_axis, const size_t shape_size,
                                   T *output) {
  size_t pos_size;
  size_t temp_pos;
  size_t ratio;

  // for example 4-D: pos = posArray[0] * input_shape[1] * input_shape[2] * input_shape[3] +
  //                        posArray[1] * input_shape[2] * input_shape[3] +
  //                        posArray[2] * input_shape[3] +
  //                        posArray[3]
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    temp_pos = pos;
    pos_size = 0;
    for (size_t i = 0; i < shape_size; ++i) {
      ratio = temp_pos / output_stride[i];
      temp_pos -= ratio * output_stride[i];
      pos_size += ratio * input_stride[input_axis[i]];
    }

    output[pos] = input[pos_size];
  }
}

template <typename T, typename S>
__global__ void ConjugateTransposeComplex(const size_t size, const T *input, const size_t *input_stride,
                                          const size_t *output_stride, const S *input_axis, const size_t shape_size,
                                          T *output) {
  size_t pos_size;
  size_t temp_pos;
  size_t ratio;

  // for example 4-D: pos = posArray[0] * input_shape[1] * input_shape[2] * input_shape[3] +
  //                        posArray[1] * input_shape[2] * input_shape[3] +
  //                        posArray[2] * input_shape[3] +
  //                        posArray[3]
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    temp_pos = pos;
    pos_size = 0;
    for (size_t i = 0; i < shape_size; ++i) {
      ratio = temp_pos / output_stride[i];
      temp_pos -= ratio * output_stride[i];
      pos_size += ratio * input_stride[input_axis[i]];
    }
    T temp_complex = T(input[pos_size].real(), -input[pos_size].imag());
    output[pos] = temp_complex;
  }
}

template <typename T, typename S>
cudaError_t CalConjugateTranspose(const size_t size, const T *input, const size_t *input_stride,
                                  const size_t *output_stride, const S *input_axis, const size_t shape_size, T *output,
                                  const uint32_t &device_id, cudaStream_t cuda_stream) {
  ConjugateTranspose<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    size, input, input_stride, output_stride, input_axis, shape_size, output);
  return GetCudaStatus();
}

template <typename T, typename S>
cudaError_t CalConjugateTransposeComplex(const size_t size, const T *input, const size_t *input_stride,
                                         const size_t *output_stride, const S *input_axis, const size_t shape_size,
                                         T *output, const uint32_t &device_id, cudaStream_t cuda_stream) {
  ConjugateTransposeComplex<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    size, input, input_stride, output_stride, input_axis, shape_size, output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t
CalConjugateTranspose<bool, int64_t>(const size_t size, const bool *input, const size_t *input_stride,
                                     const size_t *output_stride, const int64_t *input_axis, const size_t shape_size,
                                     bool *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t
CalConjugateTranspose<double, int64_t>(const size_t size, const double *input, const size_t *input_stride,
                                       const size_t *output_stride, const int64_t *input_axis, const size_t shape_size,
                                       double *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t
CalConjugateTranspose<float, int64_t>(const size_t size, const float *input, const size_t *input_stride,
                                      const size_t *output_stride, const int64_t *input_axis, const size_t shape_size,
                                      float *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t
CalConjugateTranspose<half, int64_t>(const size_t size, const half *input, const size_t *input_stride,
                                     const size_t *output_stride, const int64_t *input_axis, const size_t shape_size,
                                     half *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t
CalConjugateTranspose<int64_t, int64_t>(const size_t size, const int64_t *input, const size_t *input_stride,
                                        const size_t *output_stride, const int64_t *input_axis, const size_t shape_size,
                                        int64_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalConjugateTranspose<int, int64_t>(
  const size_t size, const int *input, const size_t *input_stride, const size_t *output_stride,
  const int64_t *input_axis, const size_t shape_size, int *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t
CalConjugateTranspose<int16_t, int64_t>(const size_t size, const int16_t *input, const size_t *input_stride,
                                        const size_t *output_stride, const int64_t *input_axis, const size_t shape_size,
                                        int16_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t
CalConjugateTranspose<int8_t, int64_t>(const size_t size, const int8_t *input, const size_t *input_stride,
                                       const size_t *output_stride, const int64_t *input_axis, const size_t shape_size,
                                       int8_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalConjugateTranspose<uint64_t, int64_t>(
  const size_t size, const uint64_t *input, const size_t *input_stride, const size_t *output_stride,
  const int64_t *input_axis, const size_t shape_size, uint64_t *output, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalConjugateTranspose<uint32_t, int64_t>(
  const size_t size, const uint32_t *input, const size_t *input_stride, const size_t *output_stride,
  const int64_t *input_axis, const size_t shape_size, uint32_t *output, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalConjugateTranspose<uint16_t, int64_t>(
  const size_t size, const uint16_t *input, const size_t *input_stride, const size_t *output_stride,
  const int64_t *input_axis, const size_t shape_size, uint16_t *output, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t
CalConjugateTranspose<uint8_t, int64_t>(const size_t size, const uint8_t *input, const size_t *input_stride,
                                        const size_t *output_stride, const int64_t *input_axis, const size_t shape_size,
                                        uint8_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalConjugateTransposeComplex<Complex<float>, int64_t>(
  const size_t size, const Complex<float> *input, const size_t *input_stride, const size_t *output_stride,
  const int64_t *input_axis, const size_t shape_size, Complex<float> *output, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalConjugateTransposeComplex<Complex<double>, int64_t>(
  const size_t size, const Complex<double> *input, const size_t *input_stride, const size_t *output_stride,
  const int64_t *input_axis, const size_t shape_size, Complex<double> *output, const uint32_t &device_id,
  cudaStream_t cuda_stream);
