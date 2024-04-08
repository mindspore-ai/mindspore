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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/gather_grad.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T, typename S>
__global__ void GatherGradKernel(const T *index, const S *grad, S *output, size_t dim, size_t num, size_t rank,
                                 const ShapeHelper output_shape, const ShapeHelper index_shape) {
  for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < num; id += blockDim.x * gridDim.x) {
    T j = index[id];
    if (j < 0) {
      j += static_cast<T>(output_shape.shape[dim]);
    }
    CUDA_KERNEL_ASSERT(j >= 0);
    size_t j_read = static_cast<size_t>(j);
    CUDA_KERNEL_ASSERT(j_read < output_shape.shape[dim]);
    size_t offset = 0;
    size_t moved_id = id;
    size_t moved_offset = 1;
    for (size_t i = rank; i > 0; i--) {
      auto real_i = i - 1;
      auto cur_idx = moved_id % index_shape.shape[real_i];
      moved_id = moved_id / index_shape.shape[real_i];
      auto cur_input_idx = real_i == dim ? j_read : cur_idx;
      offset += cur_input_idx * moved_offset;
      moved_offset *= output_shape.shape[real_i];
    }
    MsAtomicAdd(output + offset, grad[id]);
  }
  return;
}

template <typename T, typename S>
cudaError_t GatherGrad(const T *index, const S *grad, S *output, size_t dim, size_t num, size_t rank,
                       const ShapeHelper &output_shape, const ShapeHelper &index_shape, cudaStream_t stream) {
  GatherGradKernel<<<GET_BLOCKS(num), GET_THREADS, 0, stream>>>(index, grad, output, dim, num, rank, output_shape,
                                                                index_shape);
  return GetCudaStatus();
}

#define SPECIALIZE_KERNEL(T, S)                                                                                   \
  template CUDA_LIB_EXPORT cudaError_t GatherGrad<T, S>(const T *index, const S *grad, S *output, size_t dim,     \
                                                        size_t num, size_t rank, const ShapeHelper &output_shape, \
                                                        const ShapeHelper &index_shape, cudaStream_t stream);

SPECIALIZE_KERNEL(int, Complex<double>)
SPECIALIZE_KERNEL(int64_t, Complex<double>)
SPECIALIZE_KERNEL(int, Complex<float>)
SPECIALIZE_KERNEL(int64_t, Complex<float>)
SPECIALIZE_KERNEL(int, double)
SPECIALIZE_KERNEL(int64_t, double)
SPECIALIZE_KERNEL(int, float)
SPECIALIZE_KERNEL(int64_t, float)
SPECIALIZE_KERNEL(int, half)
SPECIALIZE_KERNEL(int64_t, half)
SPECIALIZE_KERNEL(int, int)
SPECIALIZE_KERNEL(int64_t, int)
SPECIALIZE_KERNEL(int, int8_t)
SPECIALIZE_KERNEL(int64_t, int8_t)
SPECIALIZE_KERNEL(int, int16_t)
SPECIALIZE_KERNEL(int64_t, int16_t)
SPECIALIZE_KERNEL(int, int64_t)
SPECIALIZE_KERNEL(int64_t, int64_t)
SPECIALIZE_KERNEL(int, unsigned char)
SPECIALIZE_KERNEL(int64_t, unsigned char)
SPECIALIZE_KERNEL(int, uint64_t)
SPECIALIZE_KERNEL(int64_t, uint64_t)
SPECIALIZE_KERNEL(int, uint32_t)
SPECIALIZE_KERNEL(int64_t, uint32_t)
SPECIALIZE_KERNEL(int, uint16_t)
SPECIALIZE_KERNEL(int64_t, uint16_t)
SPECIALIZE_KERNEL(int, bool)
SPECIALIZE_KERNEL(int64_t, bool)

#undef SPECIALIZE_KERNEL
