/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#define MAXDIM 10

#include <cuda_runtime.h>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/roll_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"

__constant__ int64_t stride_d[MAXDIM];
__constant__ int64_t kernel_shift_d[MAXDIM];
__constant__ int64_t dim_size_d[MAXDIM];

template <typename T>
__global__ void Roll(const int64_t nthreads, const int64_t dims, const T *input, T *outputs) {
  for (int out_idx = blockIdx.x * blockDim.x + threadIdx.x; out_idx < nthreads; out_idx += gridDim.x * blockDim.x) {
    int64_t offset = 0;
    for (int i = 0; i < dims; i++) {
      const int64_t indx = (out_idx / stride_d[i]) % dim_size_d[i];
      int64_t shifted_indx = (indx + kernel_shift_d[i]) % dim_size_d[i];
      offset += (shifted_indx - indx) * stride_d[i];
    }
    outputs[out_idx + offset] = input[out_idx];
  }
  return;
}

template <typename T>
cudaError_t CalRoll(const T *input, T *outputs, int64_t *stride, int64_t *kernel_shift, int64_t *dim_size,
                    const size_t outer_size, const int64_t dims, const uint32_t &device_id, cudaStream_t cuda_stream) {
  cudaMemcpyToSymbol(stride_d, &stride[0], dims * sizeof(int64_t));
  cudaMemcpyToSymbol(kernel_shift_d, &kernel_shift[0], dims * sizeof(int64_t));
  cudaMemcpyToSymbol(dim_size_d, &dim_size[0], dims * sizeof(int64_t));

  Roll<<<CUDA_BLOCKS(device_id, outer_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(outer_size, dims, input,
                                                                                        outputs);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalRoll<int32_t>(const int32_t *input, int32_t *outputs, int64_t *stride,
                                                      int64_t *kernel_shift, int64_t *dim_size, const size_t outer_size,
                                                      const int64_t dims, const uint32_t &device_id,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalRoll<float>(const float *input, float *outputs, int64_t *stride,
                                                    int64_t *kernel_shift, int64_t *dim_size, const size_t outer_size,
                                                    const int64_t dims, const uint32_t &device_id,
                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalRoll<half>(const half *input, half *outputs, int64_t *stride,
                                                   int64_t *kernel_shift, int64_t *dim_size, const size_t outer_size,
                                                   const int64_t dims, const uint32_t &device_id,
                                                   cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalRoll<double>(const double *input, double *outputs, int64_t *stride,
                                                     int64_t *kernel_shift, int64_t *dim_size, const size_t outer_size,
                                                     const int64_t dims, const uint32_t &device_id,
                                                     cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalRoll<int8_t>(const int8_t *input, int8_t *outputs, int64_t *stride,
                                                     int64_t *kernel_shift, int64_t *dim_size, const size_t outer_size,
                                                     const int64_t dims, const uint32_t &device_id,
                                                     cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalRoll<uint8_t>(const uint8_t *input, uint8_t *outputs, int64_t *stride,
                                                      int64_t *kernel_shift, int64_t *dim_size, const size_t outer_size,
                                                      const int64_t dims, const uint32_t &device_id,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalRoll<uint32_t>(const uint32_t *input, uint32_t *outputs, int64_t *stride,
                                                       int64_t *kernel_shift, int64_t *dim_size,
                                                       const size_t outer_size, const int64_t dims,
                                                       const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalRoll<int16_t>(const int16_t *input, int16_t *outputs, int64_t *stride,
                                                      int64_t *kernel_shift, int64_t *dim_size, const size_t outer_size,
                                                      const int64_t dims, const uint32_t &device_id,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalRoll<int64_t>(const int64_t *input, int64_t *outputs, int64_t *stride,
                                                      int64_t *kernel_shift, int64_t *dim_size, const size_t outer_size,
                                                      const int64_t dims, const uint32_t &device_id,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalRoll<bool>(const bool *input, bool *outputs, int64_t *stride,
                                                   int64_t *kernel_shift, int64_t *dim_size, const size_t outer_size,
                                                   const int64_t dims, const uint32_t &device_id,
                                                   cudaStream_t cuda_stream);
