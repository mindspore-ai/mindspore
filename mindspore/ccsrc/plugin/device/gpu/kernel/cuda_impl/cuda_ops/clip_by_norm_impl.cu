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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/clip_by_norm_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__global__ void AbsKernel(const size_t size, const T *in, T *out) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    out[i] = (in[i] >= 0) ? in[i] : -in[i];
  }
}

template <>
__global__ void AbsKernel(const size_t size, const float *in, float *out) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    out[i] = fabs(in[i]);
  }
}

template <>
__global__ void AbsKernel(const size_t size, const half *in, half *out) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    float zero = 0;
    out[i] = (in[i] >= __float2half(zero)) ? in[i] : -in[i];
  }
}

template <typename T>
__global__ void CompKernel(const size_t size, const T *x, const T *temp_output_addr, float *output_addr) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    if (temp_output_addr[i] * x[i] >= 0) {
      output_addr[i] = (temp_output_addr[i] * temp_output_addr[i]) > (x[i] * x[i]) ? x[i] : temp_output_addr[i];
    } else {
      output_addr[i] = temp_output_addr[i];
    }
  }
}

template <>
__global__ void CompKernel(const size_t size, const half *x, const half *temp_output_addr, float *output_addr) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    float zero = 0;
    if (temp_output_addr[i] * x[i] >= __float2half(zero)) {
      output_addr[i] = (temp_output_addr[i] * temp_output_addr[i]) > (x[i] * x[i]) ? __half2float(x[i])
                                                                                   : __half2float(temp_output_addr[i]);
    } else {
      output_addr[i] = __half2float(temp_output_addr[i]);
    }
  }
}

template <>
__global__ void CompKernel(const size_t size, const float *x, const float *temp_output_addr, float *output_addr) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    if (temp_output_addr[i] * x[i] >= 0) {
      output_addr[i] = (temp_output_addr[i] * temp_output_addr[i]) > (x[i] * x[i]) ? x[i] : temp_output_addr[i];
    } else {
      output_addr[i] = temp_output_addr[i];
    }
  }
}

template <typename T>
cudaError_t AbsOp(const size_t size, const T *in, T *out, cudaStream_t cuda_stream) {
  AbsKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, in, out);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t AbsOp<float>(const size_t size, const float *in, float *out,
                                                  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t AbsOp<half>(const size_t size, const half *in, half *out,
                                                 cudaStream_t cuda_stream);

template <typename T>
cudaError_t CompOp(const size_t size, const T *x, const T *temp_output_addr, float *output_addr,
                   cudaStream_t cuda_stream) {
  CompKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, x, temp_output_addr, output_addr);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CompOp<float>(const size_t size, const float *x, const float *temp_output_addr,
                                                   float *output_addr, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CompOp<half>(const size_t size, const half *x, const half *temp_output_addr,
                                                  float *output_addr, cudaStream_t cuda_stream);
