/**
 * Copyright 2020-2024 Huawei Technologies Co., Ltd
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

#include <iostream>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/gatherd.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "include/cuda_fp16.h"

template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T, typename S>
__global__ void GatherDKernel(const T *input, const S *index, T *output, size_t dim_before_axis_index,
                              size_t dim_at_axis_index, size_t dim_after_axis_index, size_t dim_at_axis_input,
                              size_t dim_after_axis_input, size_t num) {
  for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < num; id += blockDim.x * gridDim.x) {
    S j = index[id];
    if (j < 0) {
      j += static_cast<S>(dim_at_axis_input);
    }
    CUDA_KERNEL_ASSERT(j >= 0);
    size_t j_read = static_cast<size_t>(j);
    CUDA_KERNEL_ASSERT(j_read < dim_at_axis_input);
    size_t offset = id % dim_after_axis_index + j_read * dim_after_axis_input +
                    ((id / (dim_after_axis_index * dim_at_axis_index)) % dim_before_axis_index) *
                      (dim_at_axis_input * dim_after_axis_input);
    output[id] = input[offset];
  }
  return;
}

template <typename T, typename S>
cudaError_t GatherD(const T *input, const S *index, T *output, size_t dim_before_axis_index, size_t dim_at_axis_index,
                    size_t dim_after_axis_index, size_t dim_at_axis_input, size_t dim_after_axis_input, size_t num,
                    cudaStream_t stream, uint32_t device_id) {
  GatherDKernel<<<CUDA_BLOCKS(device_id, num), CUDA_THREADS(device_id), 0, stream>>>(
    input, index, output, dim_before_axis_index, dim_at_axis_index, dim_after_axis_index, dim_at_axis_input,
    dim_after_axis_input, num);
  return GetCudaStatus();
}

#define SPECIALIZE_KERNEL(T, S)                                                                        \
  template CUDA_LIB_EXPORT cudaError_t GatherD<T, S>(                                                  \
    const T *input, const S *index, T *output, size_t dim_before_axis_index, size_t dim_at_axis_index, \
    size_t dim_after_axis_index, size_t dim_at_axis_input, size_t dim_after_axis_input, size_t num,    \
    cudaStream_t stream, uint32_t device_id);

SPECIALIZE_KERNEL(float, int64_t)
SPECIALIZE_KERNEL(Complex<double>, int)
SPECIALIZE_KERNEL(Complex<double>, int64_t)
SPECIALIZE_KERNEL(Complex<float>, int)
SPECIALIZE_KERNEL(Complex<float>, int64_t)
SPECIALIZE_KERNEL(double, int)
SPECIALIZE_KERNEL(double, int64_t)
SPECIALIZE_KERNEL(float, int)
SPECIALIZE_KERNEL(half, int)
SPECIALIZE_KERNEL(half, int64_t)
SPECIALIZE_KERNEL(int64_t, int)
SPECIALIZE_KERNEL(int64_t, int64_t)
SPECIALIZE_KERNEL(int, int)
SPECIALIZE_KERNEL(int, int64_t)
SPECIALIZE_KERNEL(int16_t, int)
SPECIALIZE_KERNEL(int16_t, int64_t)
SPECIALIZE_KERNEL(int8_t, int)
SPECIALIZE_KERNEL(int8_t, int64_t)
SPECIALIZE_KERNEL(unsigned char, int)
SPECIALIZE_KERNEL(unsigned char, int64_t)
SPECIALIZE_KERNEL(bool, int)
SPECIALIZE_KERNEL(bool, int64_t)
SPECIALIZE_KERNEL(uint16_t, int)
SPECIALIZE_KERNEL(uint16_t, int64_t)
SPECIALIZE_KERNEL(uint32_t, int)
SPECIALIZE_KERNEL(uint32_t, int64_t)
SPECIALIZE_KERNEL(uint64_t, int)
SPECIALIZE_KERNEL(uint64_t, int64_t)

#undef SPECIALIZE_KERNEL
