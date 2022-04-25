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

#include "cum_minmax_impl.cuh"
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <algorithm>
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T>
__device__ bool IsNan(const T &x) {
  return isnan(x);
}

__device__ bool IsNan(const half &x) { return __hisnan(x); }

template <typename T, typename OP>
struct binary_op {
  const T *input_ptr_;
  size_t axis_inner_size_;
  size_t axis_size_;
  size_t inner_size_;
  OP op;

  __thrust_exec_check_disable__ __device__ size_t operator()(const size_t &lhs, const size_t &rhs) const {
    if (rhs % axis_size_) {
      size_t batch_idx = rhs / axis_size_;
      size_t axis_idx = rhs - batch_idx * axis_size_;
      size_t outer_idx = batch_idx / inner_size_;
      size_t inner_idx = batch_idx - outer_idx * inner_size_;
      size_t fix_part = outer_idx * axis_inner_size_ + inner_idx;
      size_t lhs_idx = fix_part + lhs * inner_size_;
      size_t rhs_idx = fix_part + axis_idx * inner_size_;
      return IsNan(input_ptr_[lhs_idx]) || op(input_ptr_[lhs_idx], input_ptr_[rhs_idx]) ? lhs : axis_idx;
    } else {
      return 0;
    }
  }
};

template <typename T, typename S>
__global__ void DecodeKernel(const T *input_ptr, const size_t *workspace_ptr, T *value_ptr, S *index_ptr,
                             size_t element_size, size_t axis_inner_size, size_t axis_size, size_t inner_size) {
  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < element_size; tid += blockDim.x * gridDim.x) {
    size_t batch_idx = tid / axis_size;
    size_t axis_idx = tid - batch_idx * axis_size;
    size_t outer_idx = batch_idx / inner_size;
    size_t inner_idx = batch_idx - outer_idx * inner_size;
    size_t fix_part = outer_idx * axis_inner_size + inner_idx;
    size_t real_idx = fix_part + axis_idx * inner_size;
    size_t cum_idx = fix_part + workspace_ptr[tid] * inner_size;
    value_ptr[real_idx] = input_ptr[cum_idx];
    index_ptr[real_idx] = static_cast<S>(workspace_ptr[tid]);
  }
}

template <typename T, typename S>
void CumMinMax(enum CumOpType op_type, const T *input_ptr, size_t *workspace_ptr, T *value_ptr, S *index_ptr,
               size_t element_size, size_t axis_size, size_t inner_size, cudaStream_t cuda_stream) {
  // Cummin/Cummax cuda algorithm:
  // 1. Generate a sequence from 0 to element_size-1;
  // 2. Using thrust:inclusive_scan to get the cumulative maximum/minimum result of transposed array.
  //    Note that 1. Segmentation of array is done within binary_op of inclusive_scan;
  //              2. it's not necessary to directly transpose the original array, but using the mapping rule;
  // 3. Restore the transposed array using DecodeKernel, and also with the help of mapping rule.
  auto device = thrust::cuda::par.on(cuda_stream);
  auto thrust_ptr = thrust::device_pointer_cast(workspace_ptr);
  thrust::sequence(device, thrust_ptr, thrust_ptr + element_size);
  auto axis_inner_size = axis_size * inner_size;
  switch (op_type) {
    case CUMMIN: {
      binary_op<T, thrust::less<T>> op{input_ptr, axis_inner_size, axis_size, inner_size};
      thrust::inclusive_scan(device, thrust_ptr, thrust_ptr + element_size, thrust_ptr, op);
      break;
    }
    case CUMMAX: {
      binary_op<T, thrust::greater<T>> op{input_ptr, axis_inner_size, axis_size, inner_size};
      thrust::inclusive_scan(device, thrust_ptr, thrust_ptr + element_size, thrust_ptr, op);
      break;
    }
    default:
      break;
  }

  DecodeKernel<<<GET_BLOCKS(element_size), GET_THREADS, 0, cuda_stream>>>(
    input_ptr, workspace_ptr, value_ptr, index_ptr, element_size, axis_inner_size, axis_size, inner_size);
}

template CUDA_LIB_EXPORT void CumMinMax<int8_t, int32_t>(enum CumOpType op_type, const int8_t *input_ptr,
                                                         size_t *workspace_ptr, int8_t *value_ptr, int32_t *index_ptr,
                                                         size_t element_size, size_t axis_size, size_t inner_size,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<int16_t, int32_t>(enum CumOpType op_type, const int16_t *input_ptr,
                                                          size_t *workspace_ptr, int16_t *value_ptr, int32_t *index_ptr,
                                                          size_t element_size, size_t axis_size, size_t inner_size,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<int32_t, int32_t>(enum CumOpType op_type, const int32_t *input_ptr,
                                                          size_t *workspace_ptr, int32_t *value_ptr, int32_t *index_ptr,
                                                          size_t element_size, size_t axis_size, size_t inner_size,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<int64_t, int32_t>(enum CumOpType op_type, const int64_t *input_ptr,
                                                          size_t *workspace_ptr, int64_t *value_ptr, int32_t *index_ptr,
                                                          size_t element_size, size_t axis_size, size_t inner_size,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<uint8_t, int32_t>(enum CumOpType op_type, const uint8_t *input_ptr,
                                                          size_t *workspace_ptr, uint8_t *value_ptr, int32_t *index_ptr,
                                                          size_t element_size, size_t axis_size, size_t inner_size,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<uint16_t, int32_t>(enum CumOpType op_type, const uint16_t *input_ptr,
                                                           size_t *workspace_ptr, uint16_t *value_ptr,
                                                           int32_t *index_ptr, size_t element_size, size_t axis_size,
                                                           size_t inner_size, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<uint32_t, int32_t>(enum CumOpType op_type, const uint32_t *input_ptr,
                                                           size_t *workspace_ptr, uint32_t *value_ptr,
                                                           int32_t *index_ptr, size_t element_size, size_t axis_size,
                                                           size_t inner_size, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<uint64_t, int32_t>(enum CumOpType op_type, const uint64_t *input_ptr,
                                                           size_t *workspace_ptr, uint64_t *value_ptr,
                                                           int32_t *index_ptr, size_t element_size, size_t axis_size,
                                                           size_t inner_size, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<half, int32_t>(enum CumOpType op_type, const half *input_ptr,
                                                       size_t *workspace_ptr, half *value_ptr, int32_t *index_ptr,
                                                       size_t element_size, size_t axis_size, size_t inner_size,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<float, int32_t>(enum CumOpType op_type, const float *input_ptr,
                                                        size_t *workspace_ptr, float *value_ptr, int32_t *index_ptr,
                                                        size_t element_size, size_t axis_size, size_t inner_size,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<double, int32_t>(enum CumOpType op_type, const double *input_ptr,
                                                         size_t *workspace_ptr, double *value_ptr, int32_t *index_ptr,
                                                         size_t element_size, size_t axis_size, size_t inner_size,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<int8_t, int64_t>(enum CumOpType op_type, const int8_t *input_ptr,
                                                         size_t *workspace_ptr, int8_t *value_ptr, int64_t *index_ptr,
                                                         size_t element_size, size_t axis_size, size_t inner_size,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<int16_t, int64_t>(enum CumOpType op_type, const int16_t *input_ptr,
                                                          size_t *workspace_ptr, int16_t *value_ptr, int64_t *index_ptr,
                                                          size_t element_size, size_t axis_size, size_t inner_size,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<int32_t, int64_t>(enum CumOpType op_type, const int32_t *input_ptr,
                                                          size_t *workspace_ptr, int32_t *value_ptr, int64_t *index_ptr,
                                                          size_t element_size, size_t axis_size, size_t inner_size,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<int64_t, int64_t>(enum CumOpType op_type, const int64_t *input_ptr,
                                                          size_t *workspace_ptr, int64_t *value_ptr, int64_t *index_ptr,
                                                          size_t element_size, size_t axis_size, size_t inner_size,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<uint8_t, int64_t>(enum CumOpType op_type, const uint8_t *input_ptr,
                                                          size_t *workspace_ptr, uint8_t *value_ptr, int64_t *index_ptr,
                                                          size_t element_size, size_t axis_size, size_t inner_size,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<uint16_t, int64_t>(enum CumOpType op_type, const uint16_t *input_ptr,
                                                           size_t *workspace_ptr, uint16_t *value_ptr,
                                                           int64_t *index_ptr, size_t element_size, size_t axis_size,
                                                           size_t inner_size, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<uint32_t, int64_t>(enum CumOpType op_type, const uint32_t *input_ptr,
                                                           size_t *workspace_ptr, uint32_t *value_ptr,
                                                           int64_t *index_ptr, size_t element_size, size_t axis_size,
                                                           size_t inner_size, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<uint64_t, int64_t>(enum CumOpType op_type, const uint64_t *input_ptr,
                                                           size_t *workspace_ptr, uint64_t *value_ptr,
                                                           int64_t *index_ptr, size_t element_size, size_t axis_size,
                                                           size_t inner_size, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<half, int64_t>(enum CumOpType op_type, const half *input_ptr,
                                                       size_t *workspace_ptr, half *value_ptr, int64_t *index_ptr,
                                                       size_t element_size, size_t axis_size, size_t inner_size,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<float, int64_t>(enum CumOpType op_type, const float *input_ptr,
                                                        size_t *workspace_ptr, float *value_ptr, int64_t *index_ptr,
                                                        size_t element_size, size_t axis_size, size_t inner_size,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<double, int64_t>(enum CumOpType op_type, const double *input_ptr,
                                                         size_t *workspace_ptr, double *value_ptr, int64_t *index_ptr,
                                                         size_t element_size, size_t axis_size, size_t inner_size,
                                                         cudaStream_t cuda_stream);
