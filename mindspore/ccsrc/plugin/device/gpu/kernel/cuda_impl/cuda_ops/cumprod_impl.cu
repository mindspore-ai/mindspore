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

#include "cumprod_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__global__ void Copy(T *input, T *output, size_t size) {
  size_t step = blockDim.x * gridDim.x;
  for (size_t write_index = blockIdx.x * blockDim.x + threadIdx.x; write_index < size; write_index += step) {
    input[write_index] = output[write_index];
  }
}

template <typename T>
__global__ void LeftMoveProd(const T *input, T *output, size_t dim0, size_t dim1, size_t dim2, size_t stride,
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
        output[read_index] = 1;
      } else {
        size_t read_index2 = (j - 1) * stride2 + offset;
        output[read_index] = input[read_index2];
      }
    }
  }
}

template <typename T>
__global__ void RightMoveProd(const T *input, T *output, size_t dim0, size_t dim1, size_t dim2, size_t stride,
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
        output[read_index] = 1;
      } else {
        size_t read_index2 = (j + 1) * stride2 + offset;
        output[read_index] = input[read_index2];
      }
    }
  }
}
template <typename T>
__global__ void CumProdKernelReverse(const T *input, T *output, size_t dim0, size_t dim1, size_t dim2, size_t stride,
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
        output[read_index] = output[read_index2] * input[read_index];
      }
    }
  }
}

template <typename T>
__global__ void CumProdKernel(const T *input, T *output, size_t dim0, size_t dim1, size_t dim2, size_t stride,
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
        output[read_index] = output[read_index2] * input[read_index];
      }
    }
  }
}
template <typename T>
void CumProd(const T *input, T *output, T *workspace, size_t dim0, size_t dim1, size_t dim2, size_t stride,
             size_t stride2, bool exclusive_, bool reverse_, cudaStream_t stream) {
  int size = dim0 * dim2;
  if (exclusive_) {
    if (reverse_) {
      RightMoveProd<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(input, output, dim0, dim1, dim2, stride, stride2);
      Copy<<<GET_BLOCKS(size * dim1), GET_THREADS, 0, stream>>>(workspace, output, size * dim1);
      CumProdKernelReverse<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(workspace, output, dim0, dim1, dim2, stride,
                                                                         stride2);
    } else {
      LeftMoveProd<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(input, output, dim0, dim1, dim2, stride, stride2);
      Copy<<<GET_BLOCKS(size * dim1), GET_THREADS, 0, stream>>>(workspace, output, size * dim1);
      CumProdKernel<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(workspace, output, dim0, dim1, dim2, stride, stride2);
    }
  } else {
    if (reverse_) {
      CumProdKernelReverse<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(input, output, dim0, dim1, dim2, stride,
                                                                         stride2);
    } else {
      CumProdKernel<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(input, output, dim0, dim1, dim2, stride, stride2);
    }
  }
  return;
}

template CUDA_LIB_EXPORT void CumProd<uint8_t>(const uint8_t *input, uint8_t *output, uint8_t *workspace, size_t dim0,
                                               size_t dim1, size_t dim2, size_t stride, size_t stride2, bool exclusive_,
                                               bool reverse_, cudaStream_t stream);
template CUDA_LIB_EXPORT void CumProd<uint16_t>(const uint16_t *input, uint16_t *output, uint16_t *workspace,
                                                size_t dim0, size_t dim1, size_t dim2, size_t stride, size_t stride2,
                                                bool exclusive_, bool reverse_, cudaStream_t stream);
template CUDA_LIB_EXPORT void CumProd<uint32_t>(const uint32_t *input, uint32_t *output, uint32_t *workspace,
                                                size_t dim0, size_t dim1, size_t dim2, size_t stride, size_t stride2,
                                                bool exclusive_, bool reverse_, cudaStream_t stream);
template CUDA_LIB_EXPORT void CumProd<uint64_t>(const uint64_t *input, uint64_t *output, uint64_t *workspace,
                                                size_t dim0, size_t dim1, size_t dim2, size_t stride, size_t stride2,
                                                bool exclusive_, bool reverse_, cudaStream_t stream);
template CUDA_LIB_EXPORT void CumProd<int8_t>(const int8_t *input, int8_t *output, int8_t *workspace, size_t dim0,
                                              size_t dim1, size_t dim2, size_t stride, size_t stride2, bool exclusive_,
                                              bool reverse_, cudaStream_t stream);
template CUDA_LIB_EXPORT void CumProd<int16_t>(const int16_t *input, int16_t *output, int16_t *workspace, size_t dim0,
                                               size_t dim1, size_t dim2, size_t stride, size_t stride2, bool exclusive_,
                                               bool reverse_, cudaStream_t stream);
template CUDA_LIB_EXPORT void CumProd<int32_t>(const int32_t *input, int32_t *output, int32_t *workspace, size_t dim0,
                                               size_t dim1, size_t dim2, size_t stride, size_t stride2, bool exclusive_,
                                               bool reverse_, cudaStream_t stream);
template CUDA_LIB_EXPORT void CumProd<int64_t>(const int64_t *input, int64_t *output, int64_t *workspace, size_t dim0,
                                               size_t dim1, size_t dim2, size_t stride, size_t stride2, bool exclusive_,
                                               bool reverse_, cudaStream_t stream);
template CUDA_LIB_EXPORT void CumProd<double>(const double *input, double *output, double *workspace, size_t dim0,
                                              size_t dim1, size_t dim2, size_t stride, size_t stride2, bool exclusive_,
                                              bool reverse_, cudaStream_t stream);
template CUDA_LIB_EXPORT void CumProd<float>(const float *input, float *output, float *workspace, size_t dim0,
                                             size_t dim1, size_t dim2, size_t stride, size_t stride2, bool exclusive_,
                                             bool reverse_, cudaStream_t stream);
template CUDA_LIB_EXPORT void CumProd<half>(const half *input, half *output, half *workspace, size_t dim0, size_t dim1,
                                            size_t dim2, size_t stride, size_t stride2, bool exclusive_, bool reverse_,
                                            cudaStream_t stream);
