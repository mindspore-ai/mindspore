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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cum_minmax_impl.cuh"
#include <cub/cub.cuh>
#include <thrust/functional.h>
#include <algorithm>
#include "include/cuda_fp16.h"

template <typename DataType>
__device__ bool IsNan(const DataType &x) {
  return isnan(x);
}

__device__ bool IsNan(const half &x) { return __hisnan(x); }

template <typename DataType, typename OP>
struct BinaryOp {
  const DataType *input_ptr_;
  size_t axis_inner_size_;
  size_t axis_size_;
  size_t inner_size_;
  OP op;

  __device__ size_t operator()(const size_t &pre_trans_idx, const size_t &trans_idx) const {
    size_t axis_idx = trans_idx % axis_size_;
    if (axis_idx == 0) {
      return axis_idx;
    } else {
      size_t axis_inner_idx = trans_idx % axis_inner_size_;
      size_t outer_part = trans_idx - axis_inner_idx;
      size_t inner_idx = axis_inner_idx / axis_size_;
      size_t pre_axis_idx = pre_trans_idx % axis_size_;
      DataType lhs = input_ptr_[outer_part + pre_axis_idx * inner_size_ + inner_idx];
      DataType rhs = input_ptr_[outer_part + axis_idx * inner_size_ + inner_idx];
      return IsNan(rhs) || (!IsNan(lhs) && op(rhs, lhs)) ? trans_idx : pre_trans_idx;
    }
  }
};

template <typename DataType, typename IndexType>
__global__ void ArgMinMaxKernel(const DataType *input_ptr, const size_t *workspace_ptr, DataType *value_ptr,
                                IndexType *index_ptr, size_t element_size, size_t axis_inner_size, size_t axis_size,
                                size_t inner_size) {
  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < element_size; tid += blockDim.x * gridDim.x) {
    size_t axis_inner_idx = tid % axis_inner_size;
    size_t outer_part = tid - axis_inner_idx;
    size_t axis_idx = axis_inner_idx % axis_size;
    size_t inner_idx = axis_inner_idx / axis_size;
    size_t real_idx = outer_part + axis_idx * inner_size + inner_idx;
    size_t real_arg = workspace_ptr[tid] % axis_size;
    value_ptr[real_idx] = input_ptr[outer_part + real_arg * inner_size + inner_idx];
    index_ptr[real_idx] = static_cast<IndexType>(real_arg);
  }
}

template <typename DataType, typename IndexType>
void CumMinMax(CumOpType cum_op_type, const DataType *input_ptr, size_t *workspace_ptr, DataType *value_ptr,
               IndexType *index_ptr, size_t element_size, size_t axis_size, size_t inner_size,
               const uint32_t &device_id, cudaStream_t cuda_stream) {
  // Cummin/Cummax cuda algorithm:
  // 1. Generate a counting iterator from 0 to element_size-1;
  // 2. Using inclusive scan api to get the cumulative maximum/minimum result of transposed array.
  //    Note that 1. Segmentation of array is done within scan_op of inclusive scan api;
  //              2. it's not necessary to directly transpose the original array, but using the mapping rule;
  // 3. Restore the transposed array using ArgMinMaxKernel, and also with the help of mapping rule.
  auto axis_inner_size = axis_size * inner_size;
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::CountingInputIterator<size_t> count_iter(0);
  switch (cum_op_type) {
    case CUMMIN: {
      BinaryOp<DataType, thrust::less_equal<DataType>> scan_op{input_ptr, axis_inner_size, axis_size, inner_size};
      cub::DeviceScan::InclusiveScan(nullptr, temp_storage_bytes, count_iter, workspace_ptr, scan_op, element_size,
                                     cuda_stream);
      (void)cudaMalloc(&d_temp_storage, temp_storage_bytes);
      cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, count_iter, workspace_ptr, scan_op,
                                     element_size, cuda_stream);
      break;
    }
    case CUMMAX: {
      BinaryOp<DataType, thrust::greater_equal<DataType>> scan_op{input_ptr, axis_inner_size, axis_size, inner_size};
      cub::DeviceScan::InclusiveScan(nullptr, temp_storage_bytes, count_iter, workspace_ptr, scan_op, element_size,
                                     cuda_stream);
      (void)cudaMalloc(&d_temp_storage, temp_storage_bytes);
      cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, count_iter, workspace_ptr, scan_op,
                                     element_size, cuda_stream);
      break;
    }
    default:
      break;
  }

  ArgMinMaxKernel<<<CUDA_BLOCKS(device_id, element_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input_ptr, workspace_ptr, value_ptr, index_ptr, element_size, axis_inner_size, axis_size, inner_size);

  // Since cudaGetLastError can return the last error from a runtime call,
  // we catch the error in Launch function.
  (void)cudaFree(d_temp_storage);
}

template CUDA_LIB_EXPORT void CumMinMax<int8_t, int32_t>(CumOpType cum_op_type, const int8_t *input_ptr,
                                                         size_t *workspace_ptr, int8_t *value_ptr, int32_t *index_ptr,
                                                         size_t element_size, size_t axis_size, size_t inner_size,
                                                         const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<int16_t, int32_t>(CumOpType cum_op_type, const int16_t *input_ptr,
                                                          size_t *workspace_ptr, int16_t *value_ptr, int32_t *index_ptr,
                                                          size_t element_size, size_t axis_size, size_t inner_size,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<int32_t, int32_t>(CumOpType cum_op_type, const int32_t *input_ptr,
                                                          size_t *workspace_ptr, int32_t *value_ptr, int32_t *index_ptr,
                                                          size_t element_size, size_t axis_size, size_t inner_size,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<int64_t, int32_t>(CumOpType cum_op_type, const int64_t *input_ptr,
                                                          size_t *workspace_ptr, int64_t *value_ptr, int32_t *index_ptr,
                                                          size_t element_size, size_t axis_size, size_t inner_size,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<uint8_t, int32_t>(CumOpType cum_op_type, const uint8_t *input_ptr,
                                                          size_t *workspace_ptr, uint8_t *value_ptr, int32_t *index_ptr,
                                                          size_t element_size, size_t axis_size, size_t inner_size,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<uint16_t, int32_t>(CumOpType cum_op_type, const uint16_t *input_ptr,
                                                           size_t *workspace_ptr, uint16_t *value_ptr,
                                                           int32_t *index_ptr, size_t element_size, size_t axis_size,
                                                           size_t inner_size, const uint32_t &device_id,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<uint32_t, int32_t>(CumOpType cum_op_type, const uint32_t *input_ptr,
                                                           size_t *workspace_ptr, uint32_t *value_ptr,
                                                           int32_t *index_ptr, size_t element_size, size_t axis_size,
                                                           size_t inner_size, const uint32_t &device_id,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<uint64_t, int32_t>(CumOpType cum_op_type, const uint64_t *input_ptr,
                                                           size_t *workspace_ptr, uint64_t *value_ptr,
                                                           int32_t *index_ptr, size_t element_size, size_t axis_size,
                                                           size_t inner_size, const uint32_t &device_id,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<half, int32_t>(CumOpType cum_op_type, const half *input_ptr,
                                                       size_t *workspace_ptr, half *value_ptr, int32_t *index_ptr,
                                                       size_t element_size, size_t axis_size, size_t inner_size,
                                                       const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<float, int32_t>(CumOpType cum_op_type, const float *input_ptr,
                                                        size_t *workspace_ptr, float *value_ptr, int32_t *index_ptr,
                                                        size_t element_size, size_t axis_size, size_t inner_size,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<double, int32_t>(CumOpType cum_op_type, const double *input_ptr,
                                                         size_t *workspace_ptr, double *value_ptr, int32_t *index_ptr,
                                                         size_t element_size, size_t axis_size, size_t inner_size,
                                                         const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<int8_t, int64_t>(CumOpType cum_op_type, const int8_t *input_ptr,
                                                         size_t *workspace_ptr, int8_t *value_ptr, int64_t *index_ptr,
                                                         size_t element_size, size_t axis_size, size_t inner_size,
                                                         const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<int16_t, int64_t>(CumOpType cum_op_type, const int16_t *input_ptr,
                                                          size_t *workspace_ptr, int16_t *value_ptr, int64_t *index_ptr,
                                                          size_t element_size, size_t axis_size, size_t inner_size,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<int32_t, int64_t>(CumOpType cum_op_type, const int32_t *input_ptr,
                                                          size_t *workspace_ptr, int32_t *value_ptr, int64_t *index_ptr,
                                                          size_t element_size, size_t axis_size, size_t inner_size,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<int64_t, int64_t>(CumOpType cum_op_type, const int64_t *input_ptr,
                                                          size_t *workspace_ptr, int64_t *value_ptr, int64_t *index_ptr,
                                                          size_t element_size, size_t axis_size, size_t inner_size,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<uint8_t, int64_t>(CumOpType cum_op_type, const uint8_t *input_ptr,
                                                          size_t *workspace_ptr, uint8_t *value_ptr, int64_t *index_ptr,
                                                          size_t element_size, size_t axis_size, size_t inner_size,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<uint16_t, int64_t>(CumOpType cum_op_type, const uint16_t *input_ptr,
                                                           size_t *workspace_ptr, uint16_t *value_ptr,
                                                           int64_t *index_ptr, size_t element_size, size_t axis_size,
                                                           size_t inner_size, const uint32_t &device_id,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<uint32_t, int64_t>(CumOpType cum_op_type, const uint32_t *input_ptr,
                                                           size_t *workspace_ptr, uint32_t *value_ptr,
                                                           int64_t *index_ptr, size_t element_size, size_t axis_size,
                                                           size_t inner_size, const uint32_t &device_id,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<uint64_t, int64_t>(CumOpType cum_op_type, const uint64_t *input_ptr,
                                                           size_t *workspace_ptr, uint64_t *value_ptr,
                                                           int64_t *index_ptr, size_t element_size, size_t axis_size,
                                                           size_t inner_size, const uint32_t &device_id,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<half, int64_t>(CumOpType cum_op_type, const half *input_ptr,
                                                       size_t *workspace_ptr, half *value_ptr, int64_t *index_ptr,
                                                       size_t element_size, size_t axis_size, size_t inner_size,
                                                       const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<float, int64_t>(CumOpType cum_op_type, const float *input_ptr,
                                                        size_t *workspace_ptr, float *value_ptr, int64_t *index_ptr,
                                                        size_t element_size, size_t axis_size, size_t inner_size,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<double, int64_t>(CumOpType cum_op_type, const double *input_ptr,
                                                         size_t *workspace_ptr, double *value_ptr, int64_t *index_ptr,
                                                         size_t element_size, size_t axis_size, size_t inner_size,
                                                         const uint32_t &device_id, cudaStream_t cuda_stream);
