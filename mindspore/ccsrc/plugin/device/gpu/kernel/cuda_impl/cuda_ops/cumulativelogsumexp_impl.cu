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

#include "cumulativelogsumexp_impl.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <limits>

inline __device__ double logT(double x) { return log(x); }
inline __device__ float logT(float x) { return logf(x); }
inline __device__ half logT(half x) { return hlog(x); }

inline __device__ double expT(double x) { return exp(x); }
inline __device__ float expT(float x) { return exp(x); }
inline __device__ half expT(half x) { return hexp(x); }

template <typename T>
inline __device__ T neg_infT() {
  return -std::numeric_limits<T>::infinity();
}
template <>
inline __device__ half neg_infT() {
  return __float2half(-INFINITY);
}

template <typename T>
inline __device__ T neg_maxT() {
  return -std::numeric_limits<T>::max();
}
template <>
inline __device__ half neg_maxT() {
  return __float2half(-6.550e+04);
}

template <typename T>
__global__ void CumulativeLogsumexpKernelReverse(const T *input, T *output, size_t dim0, size_t dim1, size_t dim2,
                                                 size_t stride, size_t stride2) {
  size_t num = dim0 * dim2;
  size_t i, k, offset;
  size_t step = blockDim.x * gridDim.x;
  for (size_t write_index = blockIdx.x * blockDim.x + threadIdx.x; write_index < num; write_index += step) {
    i = write_index / dim2 % dim0;
    k = write_index % dim2;
    offset = i * stride + k;

    size_t read_index = (dim1 - 1) * stride2 + offset;
    output[read_index] = input[read_index];
    for (int j = dim1 - 2; j >= 0; --j) {
      read_index = j * stride2 + offset;
      output[read_index] = logT(expT(output[read_index + stride2]) + expT(input[read_index]));
    }
  }
}

template <typename T>
__global__ void CumulativeLogsumexpKernel(const T *input, T *output, size_t dim0, size_t dim1, size_t dim2,
                                          size_t stride, size_t stride2) {
  size_t num = dim0 * dim2;
  size_t i, k, offset;
  size_t step = blockDim.x * gridDim.x;
  for (size_t write_index = blockIdx.x * blockDim.x + threadIdx.x; write_index < num; write_index += step) {
    i = write_index / dim2 % dim0;
    k = write_index % dim2;
    offset = i * stride + k;

    size_t read_index = offset;
    output[read_index] = input[read_index];
    for (size_t j = 1; j < dim1; ++j) {
      read_index = j * stride2 + offset;
      output[read_index] = logT(expT(output[read_index - stride2]) + expT(input[read_index]));
    }
  }
}

template <typename T>
__global__ void CumulativeLogsumexpKernelExclusive(const T *input, T *output, size_t dim0, size_t dim1, size_t dim2,
                                                   size_t stride, size_t stride2) {
  size_t num = dim0 * dim2;
  size_t i, k, offset;
  size_t step = blockDim.x * gridDim.x;
  for (size_t write_index = blockIdx.x * blockDim.x + threadIdx.x; write_index < num; write_index += step) {
    i = write_index / dim2 % dim0;
    k = write_index % dim2;
    offset = i * stride + k;

    size_t read_index = offset;
    output[read_index] = neg_infT<T>();
    for (size_t j = 1; j < dim1; ++j) {
      read_index = j * stride2 + offset;
      output[read_index] = logT(expT(output[read_index - stride2]) + expT(input[read_index - stride2]));
    }
    output[offset] = neg_maxT<T>();
  }
}
template <typename T>
__global__ void CumulativeLogsumexpKernelReverseExclusive(const T *input, T *output, size_t dim0, size_t dim1,
                                                          size_t dim2, size_t stride, size_t stride2) {
  size_t num = dim0 * dim2;
  size_t i, k, offset;
  size_t step = blockDim.x * gridDim.x;
  for (size_t write_index = blockIdx.x * blockDim.x + threadIdx.x; write_index < num; write_index += step) {
    i = write_index / dim2 % dim0;
    k = write_index % dim2;
    offset = i * stride + k;

    size_t read_index = (dim1 - 1) * stride2 + offset;
    output[read_index] = neg_infT<T>();
    for (int j = dim1 - 2; j >= 0; --j) {
      read_index = j * stride2 + offset;
      output[read_index] = logT(expT(output[read_index + stride2]) + expT(input[read_index + stride2]));
    }
    output[(dim1 - 1) * stride2 + offset] = neg_maxT<T>();
  }
}

template <typename T>
void CumulativeLogsumexp(const T *input, T *output, size_t dim0, size_t dim1, size_t dim2, size_t stride,
                         size_t stride2, bool exclusive_, bool reverse_, const uint32_t &device_id,
                         cudaStream_t stream) {
  int size = dim0 * dim2;
  if (exclusive_) {
    if (reverse_) {
      CumulativeLogsumexpKernelReverseExclusive<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(input, output, dim0, dim1,
                                                                                              dim2, stride, stride2);
    } else {
      CumulativeLogsumexpKernelExclusive<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(input, output, dim0, dim1, dim2,
                                                                                       stride, stride2);
    }
  } else {
    if (reverse_) {
      CumulativeLogsumexpKernelReverse<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(input, output, dim0, dim1, dim2,
                                                                                     stride, stride2);
    } else {
      CumulativeLogsumexpKernel<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(input, output, dim0, dim1, dim2, stride,
                                                                              stride2);
    }
  }
  return;
}

template CUDA_LIB_EXPORT void CumulativeLogsumexp<double>(const double *input, double *output, size_t dim0, size_t dim1,
                                                          size_t dim2, size_t stride, size_t stride2, bool exclusive_,
                                                          bool reverse_, const uint32_t &device_id,
                                                          cudaStream_t stream);
template CUDA_LIB_EXPORT void CumulativeLogsumexp<float>(const float *input, float *output, size_t dim0, size_t dim1,
                                                         size_t dim2, size_t stride, size_t stride2, bool exclusive_,
                                                         bool reverse_, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void CumulativeLogsumexp<half>(const half *input, half *output, size_t dim0, size_t dim1,
                                                        size_t dim2, size_t stride, size_t stride2, bool exclusive_,
                                                        bool reverse_, const uint32_t &device_id, cudaStream_t stream);
