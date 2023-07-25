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

#include "eye_impl.cuh"
#include <iostream>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T>
__global__ void BatchEyeKernel(const size_t size, const size_t dim, T *output_addr) {
  for (size_t pointIdx = blockIdx.x * blockDim.x + threadIdx.x; pointIdx < (size); pointIdx += blockDim.x * gridDim.x) {
    size_t batchIdx = pointIdx / (dim * dim);
    size_t dst_x = (pointIdx - batchIdx * dim * dim) / dim;
    size_t dst_y = (pointIdx - batchIdx * dim * dim) % dim;
    if (dst_x == dst_y) {
      output_addr[pointIdx] = 1;
    } else {
      output_addr[pointIdx] = 0;
    }
  }
}

// for common situations where nums may not equal to cols and the dim of the output tensor is 2
template <typename T>
__global__ void CudaEyeKernel(const int64_t num_min, const int64_t cols, T *out) {
  for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < num_min; id += blockDim.x * gridDim.x) {
    out[id * cols + id] = static_cast<T>(1);
  }
}

template <typename T>
cudaError_t BatchEye(const size_t size, const size_t dim, T *output_addr, cudaStream_t cuda_stream) {
  BatchEyeKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, dim, output_addr);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CudaEye(const size_t out_size, const int64_t nums, const int64_t cols, T *out, cudaStream_t cuda_stream) {
  const int64_t num_min = nums > cols ? cols : nums;
  cudaMemset(static_cast<void *>(out), 0, out_size);
  CudaEyeKernel<<<GET_BLOCKS(num_min), GET_THREADS, 0, cuda_stream>>>(num_min, cols, out);
  cudaStreamSynchronize(cuda_stream);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t BatchEye<float>(const size_t size, const size_t dim, float *output_addr,
                                                     cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t BatchEye<double>(const size_t size, const size_t dim, double *output_addr,
                                                      cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CudaEye(const size_t out_size, const int64_t nums, const int64_t cols, bool *out,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CudaEye(const size_t out_size, const int64_t nums, const int64_t cols, half *out,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CudaEye(const size_t out_size, const int64_t nums, const int64_t cols, float *out,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CudaEye(const size_t out_size, const int64_t nums, const int64_t cols, double *out,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CudaEye(const size_t out_size, const int64_t nums, const int64_t cols, int8_t *out,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CudaEye(const size_t out_size, const int64_t nums, const int64_t cols,
                                             int16_t *out, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CudaEye(const size_t out_size, const int64_t nums, const int64_t cols,
                                             int32_t *out, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CudaEye(const size_t out_size, const int64_t nums, const int64_t cols,
                                             int64_t *out, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CudaEye(const size_t out_size, const int64_t nums, const int64_t cols,
                                             uint8_t *out, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CudaEye(const size_t out_size, const int64_t nums, const int64_t cols,
                                             uint16_t *out, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CudaEye(const size_t out_size, const int64_t nums, const int64_t cols,
                                             uint32_t *out, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CudaEye(const size_t out_size, const int64_t nums, const int64_t cols,
                                             uint64_t *out, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CudaEye(const size_t out_size, const int64_t nums, const int64_t cols,
                                             Complex<float> *out, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CudaEye(const size_t out_size, const int64_t nums, const int64_t cols,
                                             Complex<double> *out, cudaStream_t cuda_stream);
