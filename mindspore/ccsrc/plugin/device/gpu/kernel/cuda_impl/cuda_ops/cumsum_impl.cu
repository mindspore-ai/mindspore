/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include <cuda_fp16.h>
#include "cumsum_impl.cuh"

template <typename T>
__global__ void Copy(T *input, T *output, size_t size) {
  size_t step = blockDim.x * gridDim.x;
  for (size_t write_index = blockIdx.x * blockDim.x + threadIdx.x; write_index < size; write_index += step) {
    input[write_index] = output[write_index];
  }
}

template <typename T>
__global__ void LeftMoveSum(const T *input, T *output, size_t dim0, size_t dim1, size_t dim2, size_t stride,
                            size_t stride2) {
  size_t num = dim0 * dim2;
  size_t i, k, offset;
  size_t step = blockDim.x * gridDim.x;
  for (size_t write_index = blockIdx.x * blockDim.x + threadIdx.x; write_index < num; write_index += step) {
    i = write_index / dim2 % dim0;
    k = write_index % dim2;
    offset = i * stride + k;
    for (size_t j = 0; j < dim1; ++j) {
      size_t read_index = j * stride2 + offset;
      if (j == 0) {
        output[read_index] = 0;
      } else {
        size_t read_index2 = (j - 1) * stride2 + offset;
        output[read_index] = input[read_index2];
      }
    }
  }
}

template <typename T>
__global__ void RightMoveSum(const T *input, T *output, size_t dim0, size_t dim1, size_t dim2, size_t stride,
                             size_t stride2) {
  size_t num = dim0 * dim2;
  size_t i, k, offset;
  size_t step = blockDim.x * gridDim.x;
  for (size_t write_index = blockIdx.x * blockDim.x + threadIdx.x; write_index < num; write_index += step) {
    i = write_index / dim2 % dim0;
    k = write_index % dim2;
    offset = i * stride + k;
    for (int j = dim1 - 1; j >= 0; --j) {
      size_t read_index = j * stride2 + offset;
      if (j == dim1 - 1) {
        output[read_index] = 0;
      } else {
        size_t read_index2 = (j + 1) * stride2 + offset;
        output[read_index] = input[read_index2];
      }
    }
  }
}
template <typename T>
__global__ void CumSumKernelReverse(const T *input, T *output, size_t dim0, size_t dim1, size_t dim2, size_t stride,
                                    size_t stride2) {
  size_t num = dim0 * dim2;
  size_t i, k, offset;
  size_t step = blockDim.x * gridDim.x;
  for (size_t write_index = blockIdx.x * blockDim.x + threadIdx.x; write_index < num; write_index += step) {
    i = write_index / dim2 % dim0;
    k = write_index % dim2;
    offset = i * stride + k;
    for (int j = dim1 - 1; j >= 0; --j) {
      size_t read_index = j * stride2 + offset;
      if (j == dim1 - 1) {
        output[read_index] = input[read_index];
      } else {
        size_t read_index2 = (j + 1) * stride2 + offset;
        output[read_index] = output[read_index2] + input[read_index];
      }
    }
  }
}

template <typename T>
__global__ void CumSumKernel(const T *input, T *output, size_t dim0, size_t dim1, size_t dim2, size_t stride,
                             size_t stride2) {
  size_t num = dim0 * dim2;
  size_t i, k, offset;
  size_t step = blockDim.x * gridDim.x;
  for (size_t write_index = blockIdx.x * blockDim.x + threadIdx.x; write_index < num; write_index += step) {
    i = write_index / dim2 % dim0;
    k = write_index % dim2;
    offset = i * stride + k;
    for (size_t j = 0; j < dim1; ++j) {
      size_t read_index = j * stride2 + offset;
      if (j == 0) {
        output[read_index] = input[read_index];
      } else {
        size_t read_index2 = (j - 1) * stride2 + offset;
        output[read_index] = output[read_index2] + input[read_index];
      }
    }
  }
}
template <typename T>
void CumSum(const T *input, T *output, T *workspace, size_t dim0, size_t dim1, size_t dim2, size_t stride,
            size_t stride2, bool exclusive_, bool reverse_, const uint32_t &device_id, cudaStream_t stream) {
  int size = dim0 * dim2;
  if (exclusive_) {
    if (reverse_) {
      RightMoveSum<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, stream>>>(input, output, dim0, dim1,
                                                                                         dim2, stride, stride2);
      Copy<<<CUDA_BLOCKS(device_id, size * dim1), CUDA_THREADS(device_id), 0, stream>>>(workspace, output, size * dim1);
      CumSumKernelReverse<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, stream>>>(
        workspace, output, dim0, dim1, dim2, stride, stride2);
    } else {
      LeftMoveSum<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, stream>>>(input, output, dim0, dim1, dim2,
                                                                                        stride, stride2);
      Copy<<<CUDA_BLOCKS(device_id, size * dim1), CUDA_THREADS(device_id), 0, stream>>>(workspace, output, size * dim1);
      CumSumKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, stream>>>(workspace, output, dim0, dim1,
                                                                                         dim2, stride, stride2);
    }
  } else {
    if (reverse_) {
      CumSumKernelReverse<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, stream>>>(
        input, output, dim0, dim1, dim2, stride, stride2);
    } else {
      CumSumKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, stream>>>(input, output, dim0, dim1,
                                                                                         dim2, stride, stride2);
    }
  }
  return;
}

template CUDA_LIB_EXPORT void CumSum<int8_t>(const int8_t *input, int8_t *output, int8_t *workspace, size_t dim0,
                                             size_t dim1, size_t dim2, size_t stride, size_t stride2, bool exclusive_,
                                             bool reverse_, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void CumSum<int16_t>(const int16_t *input, int16_t *output, int16_t *workspace, size_t dim0,
                                              size_t dim1, size_t dim2, size_t stride, size_t stride2, bool exclusive_,
                                              bool reverse_, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void CumSum<int32_t>(const int32_t *input, int32_t *output, int32_t *workspace, size_t dim0,
                                              size_t dim1, size_t dim2, size_t stride, size_t stride2, bool exclusive_,
                                              bool reverse_, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void CumSum<int64_t>(const int64_t *input, int64_t *output, int64_t *workspace, size_t dim0,
                                              size_t dim1, size_t dim2, size_t stride, size_t stride2, bool exclusive_,
                                              bool reverse_, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void CumSum<uint8_t>(const uint8_t *input, uint8_t *output, uint8_t *workspace, size_t dim0,
                                              size_t dim1, size_t dim2, size_t stride, size_t stride2, bool exclusive_,
                                              bool reverse_, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void CumSum<uint16_t>(const uint16_t *input, uint16_t *output, uint16_t *workspace,
                                               size_t dim0, size_t dim1, size_t dim2, size_t stride, size_t stride2,
                                               bool exclusive_, bool reverse_, const uint32_t &device_id,
                                               cudaStream_t stream);
template CUDA_LIB_EXPORT void CumSum<uint32_t>(const uint32_t *input, uint32_t *output, uint32_t *workspace,
                                               size_t dim0, size_t dim1, size_t dim2, size_t stride, size_t stride2,
                                               bool exclusive_, bool reverse_, const uint32_t &device_id,
                                               cudaStream_t stream);
template CUDA_LIB_EXPORT void CumSum<uint64_t>(const uint64_t *input, uint64_t *output, uint64_t *workspace,
                                               size_t dim0, size_t dim1, size_t dim2, size_t stride, size_t stride2,
                                               bool exclusive_, bool reverse_, const uint32_t &device_id,
                                               cudaStream_t stream);
template CUDA_LIB_EXPORT void CumSum<double>(const double *input, double *output, double *workspace, size_t dim0,
                                             size_t dim1, size_t dim2, size_t stride, size_t stride2, bool exclusive_,
                                             bool reverse_, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void CumSum<float>(const float *input, float *output, float *workspace, size_t dim0,
                                            size_t dim1, size_t dim2, size_t stride, size_t stride2, bool exclusive_,
                                            bool reverse_, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void CumSum<half>(const half *input, half *output, half *workspace, size_t dim0, size_t dim1,
                                           size_t dim2, size_t stride, size_t stride2, bool exclusive_, bool reverse_,
                                           const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void CumSum<Complex<float>>(const Complex<float> *input, Complex<float> *output,
                                                     Complex<float> *workspace, size_t dim0, size_t dim1, size_t dim2,
                                                     size_t stride, size_t stride2, bool exclusive_, bool reverse_,
                                                     const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void CumSum<Complex<double>>(const Complex<double> *input, Complex<double> *output,
                                                      Complex<double> *workspace, size_t dim0, size_t dim1, size_t dim2,
                                                      size_t stride, size_t stride2, bool exclusive_, bool reverse_,
                                                      const uint32_t &device_id, cudaStream_t stream);
