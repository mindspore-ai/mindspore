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
#include <algorithm>
#include "include/cuda_fp16.h"

template <typename DataType>
__device__ bool IsNan(const DataType &x) {
  return isnan(x);
}

__device__ bool IsNan(const half &x) { return __hisnan(x); }

template <typename DataType, typename IndexType, typename BinaryOp>
__global__ void CumMinMaxKernel(BinaryOp op, const DataType *input_ptr, DataType *value_ptr, IndexType *index_ptr,
                                int axis_size, int inner_size, int axis_inner_size, int outer_inner_size) {
  int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  int step = static_cast<int>(blockDim.x * gridDim.x);
  for (int tid = idx; tid < outer_inner_size; tid += step) {
    int outer_idx = (tid / inner_size) * axis_inner_size;
    int inner_idx = tid % inner_size;
    int offset = (outer_idx + inner_idx);
    auto cur_input_ptr = input_ptr + offset;
    auto cur_value_ptr = value_ptr + offset;
    auto cur_index_ptr = index_ptr + offset;
    DataType out_val = *cur_value_ptr = *cur_input_ptr;
    IndexType out_idx = *cur_index_ptr = 0;
    for (int j = 1; j < axis_size; j++) {
      cur_input_ptr += inner_size;
      cur_value_ptr += inner_size;
      cur_index_ptr += inner_size;
      DataType cur_val = *cur_input_ptr;
      if (IsNan(cur_val) || (!IsNan(out_val) && op(cur_val, out_val))) {
        out_val = cur_val;
        out_idx = static_cast<IndexType>(j);
      }
      *cur_value_ptr = out_val;
      *cur_index_ptr = out_idx;
    }
  }
}

template <typename DataType, typename IndexType>
void CumMinMax(CumOpType cum_op_type, const DataType *input_ptr, DataType *value_ptr, IndexType *index_ptr,
               size_t outer_size_st, size_t axis_size_st, size_t inner_size_st, const uint32_t &device_id,
               cudaStream_t cuda_stream) {
  auto outer_size = static_cast<int>(outer_size_st);
  auto inner_size = static_cast<int>(inner_size_st);
  auto axis_size = static_cast<int>(axis_size_st);
  auto outer_inner_size = outer_size * inner_size;
  auto axis_inner_size = axis_size * inner_size;
  switch (cum_op_type) {
    case CUMMIN: {
      CumMinMaxKernel<<<CUDA_BLOCKS(device_id, outer_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
        thrust::less_equal<DataType>(), input_ptr, value_ptr, index_ptr, axis_size, inner_size, axis_inner_size,
        outer_inner_size);
      break;
    }
    case CUMMAX: {
      CumMinMaxKernel<<<CUDA_BLOCKS(device_id, outer_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
        thrust::greater_equal<DataType>(), input_ptr, value_ptr, index_ptr, axis_size, inner_size, axis_inner_size,
        outer_inner_size);
      break;
    }
    default:
      break;
  }
}

template CUDA_LIB_EXPORT void CumMinMax<int8_t, int32_t>(CumOpType cum_op_type, const int8_t *input_ptr,
                                                         int8_t *value_ptr, int32_t *index_ptr, size_t outer_size,
                                                         size_t axis_size, size_t inner_size, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<int16_t, int32_t>(CumOpType cum_op_type, const int16_t *input_ptr,
                                                          int16_t *value_ptr, int32_t *index_ptr, size_t outer_size,
                                                          size_t axis_size, size_t inner_size,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<int32_t, int32_t>(CumOpType cum_op_type, const int32_t *input_ptr,
                                                          int32_t *value_ptr, int32_t *index_ptr, size_t outer_size,
                                                          size_t axis_size, size_t inner_size,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<int64_t, int32_t>(CumOpType cum_op_type, const int64_t *input_ptr,
                                                          int64_t *value_ptr, int32_t *index_ptr, size_t outer_size,
                                                          size_t axis_size, size_t inner_size,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<uint8_t, int32_t>(CumOpType cum_op_type, const uint8_t *input_ptr,
                                                          uint8_t *value_ptr, int32_t *index_ptr, size_t outer_size,
                                                          size_t axis_size, size_t inner_size,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<uint16_t, int32_t>(CumOpType cum_op_type, const uint16_t *input_ptr,
                                                           uint16_t *value_ptr, int32_t *index_ptr, size_t outer_size,
                                                           size_t axis_size, size_t inner_size,
                                                           const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<uint32_t, int32_t>(CumOpType cum_op_type, const uint32_t *input_ptr,
                                                           uint32_t *value_ptr, int32_t *index_ptr, size_t outer_size,
                                                           size_t axis_size, size_t inner_size,
                                                           const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<uint64_t, int32_t>(CumOpType cum_op_type, const uint64_t *input_ptr,
                                                           uint64_t *value_ptr, int32_t *index_ptr, size_t outer_size,
                                                           size_t axis_size, size_t inner_size,
                                                           const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<half, int32_t>(CumOpType cum_op_type, const half *input_ptr, half *value_ptr,
                                                       int32_t *index_ptr, size_t outer_size, size_t axis_size,
                                                       size_t inner_size, const uint32_t &device_id,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<float, int32_t>(CumOpType cum_op_type, const float *input_ptr, float *value_ptr,
                                                        int32_t *index_ptr, size_t outer_size, size_t axis_size,
                                                        size_t inner_size, const uint32_t &device_id,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<double, int32_t>(CumOpType cum_op_type, const double *input_ptr,
                                                         double *value_ptr, int32_t *index_ptr, size_t outer_size,
                                                         size_t axis_size, size_t inner_size, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<int8_t, int64_t>(CumOpType cum_op_type, const int8_t *input_ptr,
                                                         int8_t *value_ptr, int64_t *index_ptr, size_t outer_size,
                                                         size_t axis_size, size_t inner_size, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<int16_t, int64_t>(CumOpType cum_op_type, const int16_t *input_ptr,
                                                          int16_t *value_ptr, int64_t *index_ptr, size_t outer_size,
                                                          size_t axis_size, size_t inner_size,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<int32_t, int64_t>(CumOpType cum_op_type, const int32_t *input_ptr,
                                                          int32_t *value_ptr, int64_t *index_ptr, size_t outer_size,
                                                          size_t axis_size, size_t inner_size,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<int64_t, int64_t>(CumOpType cum_op_type, const int64_t *input_ptr,
                                                          int64_t *value_ptr, int64_t *index_ptr, size_t outer_size,
                                                          size_t axis_size, size_t inner_size,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<uint8_t, int64_t>(CumOpType cum_op_type, const uint8_t *input_ptr,
                                                          uint8_t *value_ptr, int64_t *index_ptr, size_t outer_size,
                                                          size_t axis_size, size_t inner_size,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<uint16_t, int64_t>(CumOpType cum_op_type, const uint16_t *input_ptr,
                                                           uint16_t *value_ptr, int64_t *index_ptr, size_t outer_size,
                                                           size_t axis_size, size_t inner_size,
                                                           const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<uint32_t, int64_t>(CumOpType cum_op_type, const uint32_t *input_ptr,
                                                           uint32_t *value_ptr, int64_t *index_ptr, size_t outer_size,
                                                           size_t axis_size, size_t inner_size,
                                                           const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<uint64_t, int64_t>(CumOpType cum_op_type, const uint64_t *input_ptr,
                                                           uint64_t *value_ptr, int64_t *index_ptr, size_t outer_size,
                                                           size_t axis_size, size_t inner_size,
                                                           const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<half, int64_t>(CumOpType cum_op_type, const half *input_ptr, half *value_ptr,
                                                       int64_t *index_ptr, size_t outer_size, size_t axis_size,
                                                       size_t inner_size, const uint32_t &device_id,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<float, int64_t>(CumOpType cum_op_type, const float *input_ptr, float *value_ptr,
                                                        int64_t *index_ptr, size_t outer_size, size_t axis_size,
                                                        size_t inner_size, const uint32_t &device_id,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CumMinMax<double, int64_t>(CumOpType cum_op_type, const double *input_ptr,
                                                         double *value_ptr, int64_t *index_ptr, size_t outer_size,
                                                         size_t axis_size, size_t inner_size, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
