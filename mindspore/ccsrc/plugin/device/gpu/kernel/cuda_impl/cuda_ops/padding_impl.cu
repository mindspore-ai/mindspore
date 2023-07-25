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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/padding_impl.cuh"
#include "include/cuda_runtime.h"
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T>
__global__ void CalculatePaddingKernel(const T *input_ptr, size_t output_outer_size_, size_t pad_dim_size,
                                       T *output_ptr) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < output_outer_size_; i += blockDim.x * gridDim.x) {
    output_ptr[i * pad_dim_size] = input_ptr[i];
  }
}

template <typename T>
cudaError_t CalculatePadding(const T *input_ptr, size_t output_outer_size_, size_t pad_dim_size, T *output_ptr,
                             const uint32_t &device_id, cudaStream_t cuda_stream) {
  CalculatePaddingKernel<<<CUDA_BLOCKS(device_id, output_outer_size_), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input_ptr, output_outer_size_, pad_dim_size, output_ptr);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalculatePadding<int8_t>(const int8_t *input_ptr, size_t output_outer_size_,
                                                              size_t pad_dim_size, int8_t *output_ptr,
                                                              const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalculatePadding<int16_t>(const int16_t *input_ptr, size_t output_outer_size_,
                                                               size_t pad_dim_size, int16_t *output_ptr,
                                                               const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalculatePadding<int32_t>(const int32_t *input_ptr, size_t output_outer_size_,
                                                               size_t pad_dim_size, int32_t *output_ptr,
                                                               const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalculatePadding<int64_t>(const int64_t *input_ptr, size_t output_outer_size_,
                                                               size_t pad_dim_size, int64_t *output_ptr,
                                                               const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalculatePadding<uint8_t>(const uint8_t *input_ptr, size_t output_outer_size_,
                                                               size_t pad_dim_size, uint8_t *output_ptr,
                                                               const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalculatePadding<uint16_t>(const uint16_t *input_ptr, size_t output_outer_size_,
                                                                size_t pad_dim_size, uint16_t *output_ptr,
                                                                const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalculatePadding<uint32_t>(const uint32_t *input_ptr, size_t output_outer_size_,
                                                                size_t pad_dim_size, uint32_t *output_ptr,
                                                                const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalculatePadding<uint64_t>(const uint64_t *input_ptr, size_t output_outer_size_,
                                                                size_t pad_dim_size, uint64_t *output_ptr,
                                                                const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalculatePadding<half>(const half *input_ptr, size_t output_outer_size_,
                                                            size_t pad_dim_size, half *output_ptr,
                                                            const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalculatePadding<float>(const float *input_ptr, size_t output_outer_size_,
                                                             size_t pad_dim_size, float *output_ptr,
                                                             const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalculatePadding<double>(const double *input_ptr, size_t output_outer_size_,
                                                              size_t pad_dim_size, double *output_ptr,
                                                              const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalculatePadding<Complex<float>>(const Complex<float> *input_ptr,
                                                                      size_t output_outer_size_, size_t pad_dim_size,
                                                                      Complex<float> *output_ptr,
                                                                      const uint32_t &device_id,
                                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalculatePadding<Complex<double>>(const Complex<double> *input_ptr,
                                                                       size_t output_outer_size_, size_t pad_dim_size,
                                                                       Complex<double> *output_ptr,
                                                                       const uint32_t &device_id,
                                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalculatePadding<bool>(const bool *input_ptr, size_t output_outer_size_,
                                                            size_t pad_dim_size, bool *output_ptr,
                                                            const uint32_t &device_id, cudaStream_t cuda_stream);
