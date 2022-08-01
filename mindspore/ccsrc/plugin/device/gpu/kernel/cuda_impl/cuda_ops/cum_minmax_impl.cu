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
#include <limits>
#include <algorithm>
#include "include/cuda_fp16.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

namespace {
using uint = unsigned int;
}

int GetMaxSharedMemoryPerBlock(const uint32_t &device_id) {
  int max_size = 128;
  (void)cudaDeviceGetAttribute(&max_size, cudaDevAttrMaxSharedMemoryPerBlock, static_cast<int>(device_id));
  return max_size;
}

int GetMaxThreadsPerBlock(const uint32_t &device_id) {
  int max_size = 128;
  (void)cudaDeviceGetAttribute(&max_size, cudaDevAttrMaxThreadsPerBlock, static_cast<int>(device_id));
  return max_size;
}

int GetMaxGridDimX(const uint32_t &device_id) {
  int max_size = 128;
  (void)cudaDeviceGetAttribute(&max_size, cudaDevAttrMaxGridDimX, static_cast<int>(device_id));
  return max_size;
}

template <typename DataType>
__device__ __forceinline__ bool IsNan(const DataType &x) {
  return isnan(x);
}

__device__ __forceinline__ bool IsNan(const half &x) { return __hisnan(x); }

template <typename DataType>
DataType NumericMax() {
  return std::numeric_limits<DataType>::max();
}

template <typename DataType>
DataType NumericMin() {
  return std::numeric_limits<DataType>::lowest();
}

template <>
half NumericMax<half>() {
  constexpr uint16_t x = 0x7BFF;
  return half(__half_raw{x});
}

template <>
half NumericMin<half>() {
  constexpr uint16_t x = 0xFBFF;
  return half(__half_raw{x});
}

template <typename BinaryFunctor, typename DataType, typename IndexType>
__device__ __forceinline__ void Update(BinaryFunctor fun, DataType *dst_data, IndexType *dst_index, DataType src_data,
                                       IndexType src_index) {
  if (fun(src_data, *dst_data)) {
    *dst_data = src_data;
    *dst_index = src_index;
  }
}

template <typename BinaryFunctor, typename DataType, typename IndexType>
__global__ void CumMinMaxKernel(BinaryFunctor fun, const DataType *input_ptr, DataType *value_ptr, IndexType *index_ptr,
                                uint axis_size, uint inner_size, uint axis_inner_size, uint outer_inner_size,
                                DataType init) {
  uint tid = threadIdx.y;
  uint tid_d = tid << 1;  // The suffix `d` represents double.
  uint scan_per_block = blockDim.y * 2;
  extern __shared__ char share_data[];
  auto total_value_size = sizeof(DataType) * blockDim.x * scan_per_block;
  auto share_value_ptr = reinterpret_cast<DataType *>(share_data) + threadIdx.x * scan_per_block;
  auto share_index_ptr = reinterpret_cast<IndexType *>(share_data + total_value_size) + threadIdx.x * scan_per_block;
  for (uint bid = threadIdx.x + blockIdx.x * blockDim.x; bid < outer_inner_size; bid += blockDim.x * gridDim.x) {
    uint outer_idx = bid / inner_size;
    uint inner_idx = bid % inner_size;
    uint outer_inner_offset = outer_idx * axis_inner_size + inner_idx;
    auto cur_input_ptr = input_ptr + outer_inner_offset;
    auto cur_value_ptr = value_ptr + outer_inner_offset;
    auto cur_index_ptr = index_ptr + outer_inner_offset;
    DataType block_value = init;
    IndexType block_index = 0;
    // Each iteration processes (2 * blockDim.y) elements, since share memory typically larger than thread number of
    // each block.
    for (uint cid = 0; cid < axis_size; cid += scan_per_block) {
      // The following parallel scan algorithm refers to:
      // Figure 9.7 from David B. Kirk, et al. 'Programming Massively Parallel Processors'.
      uint axis_idx = cid + tid_d;
      uint axis_offset = axis_idx * inner_size;
      // Initializing share memory with input value.
      if (axis_idx < axis_size) {
        share_value_ptr[tid_d] = cur_input_ptr[axis_offset];
        share_index_ptr[tid_d] = axis_idx;
      } else {
        share_value_ptr[tid_d] = init;
      }
      if (axis_idx + 1 < axis_size) {
        share_value_ptr[tid_d + 1] = cur_input_ptr[axis_offset + inner_size];
        share_index_ptr[tid_d + 1] = axis_idx + 1;
      } else {
        share_value_ptr[tid_d + 1] = init;
      }
      // update with previous block result.
      if (tid == 0) {
        Update(fun, share_value_ptr, share_index_ptr, block_value, block_index);
      }
      // up-sweep
      for (uint stride = 1; stride < scan_per_block; stride <<= 1) {
        __syncthreads();
        uint index = (tid + 1) * (stride << 1) - 1;
        if (index < scan_per_block) {
          Update(fun, share_value_ptr + index, share_index_ptr + index, share_value_ptr[index - stride],
                 share_index_ptr[index - stride]);
        }
      }
      // down-sweep
      for (uint stride = scan_per_block >> 2; stride > 0; stride >>= 1) {
        __syncthreads();
        uint index = (tid + 1) * (stride << 1) - 1;
        if (index + stride < scan_per_block) {
          Update(fun, share_value_ptr + (index + stride), share_index_ptr + (index + stride), share_value_ptr[index],
                 share_index_ptr[index]);
        }
      }
      // write to output.
      __syncthreads();
      if (axis_idx < axis_size) {
        cur_value_ptr[axis_offset] = share_value_ptr[tid_d];
        cur_index_ptr[axis_offset] = share_index_ptr[tid_d];
      }
      if (axis_idx + 1 < axis_size) {
        cur_value_ptr[axis_offset + inner_size] = share_value_ptr[tid_d + 1];
        cur_index_ptr[axis_offset + inner_size] = share_index_ptr[tid_d + 1];
      }
      // update block_value & block_index
      if (tid == 0) {
        block_value = share_value_ptr[scan_per_block - 1];
        block_index = share_index_ptr[scan_per_block - 1];
      }
      __syncthreads();
    }
  }
}

template <typename BinaryFunctor, typename DataType, typename IndexType>
struct IndexFunctor {
  const DataType *input_ptr_;
  BinaryFunctor functor_;
  explicit IndexFunctor(const DataType *input_ptr, BinaryFunctor functor) : input_ptr_(input_ptr), functor_(functor) {}
  __device__ __forceinline__ IndexType operator()(IndexType x, IndexType y) {
    auto lhs = input_ptr_[x];
    auto rhs = input_ptr_[y];
    return functor_(lhs, rhs) ? x : y;
  }
};

template <typename BinaryFunctor, typename DataType>
struct ValueFunctor {
  BinaryFunctor functor_;
  explicit ValueFunctor(BinaryFunctor functor) : functor_(functor) {}
  __device__ __forceinline__ DataType operator()(DataType lhs, DataType rhs) { return functor_(lhs, rhs) ? lhs : rhs; }
};

template <typename BinaryOp, typename DataType>
struct BinaryFunctor {
  BinaryOp binary_op_;
  __device__ __forceinline__ bool operator()(DataType lhs, DataType rhs) {
    return !IsNan(rhs) && (IsNan(lhs) || !binary_op_(rhs, lhs));
  }
};

template <typename BinaryFunctor, typename DataType, typename IndexType>
__global__ void CumMinMaxSlowKernel(BinaryFunctor functor, const DataType *input_ptr, DataType *value_ptr,
                                    IndexType *index_ptr, uint axis_size, uint inner_size, uint axis_inner_size,
                                    uint outer_inner_size) {
  for (uint tid = blockIdx.x * blockDim.x + threadIdx.x; tid < outer_inner_size; tid += blockDim.x * gridDim.x) {
    uint outer_idx = tid / inner_size;
    uint inner_idx = tid % inner_size;
    uint offset = outer_idx * axis_inner_size + inner_idx;
    auto cur_input_ptr = input_ptr + offset;
    auto cur_value_ptr = value_ptr + offset;
    auto cur_index_ptr = index_ptr + offset;
    DataType out_val = *cur_value_ptr = *cur_input_ptr;
    IndexType out_idx = *cur_index_ptr = 0;
    for (uint j = 1; j < axis_size; j++) {
      cur_input_ptr += inner_size;
      cur_value_ptr += inner_size;
      cur_index_ptr += inner_size;
      DataType cur_val = *cur_input_ptr;
      if (!functor(out_val, cur_val)) {
        out_val = cur_val;
        out_idx = static_cast<IndexType>(j);
      }
      *cur_value_ptr = out_val;
      *cur_index_ptr = out_idx;
    }
  }
}

template <typename BinaryFunctor, typename DataType, typename IndexType>
void KernelHelper(BinaryFunctor fun, DataType init, const DataType *input_ptr, DataType *value_ptr,
                  IndexType *index_ptr, size_t outer_size_st, size_t axis_size_st, size_t inner_size_st,
                  const uint32_t &device_id, cudaStream_t cuda_stream) {
  if (outer_size_st == 1 && inner_size_st == 1) {
    // Special case where only one dimension that needs to compute, so using cub library is the most efficient way.
    ValueFunctor<BinaryFunctor, DataType> value_fun{fun};
    IndexFunctor<BinaryFunctor, DataType, IndexType> index_fun{input_ptr, fun};
    size_t value_storage_bytes = 0;
    size_t index_storage_bytes = 0;
    cub::CountingInputIterator<IndexType> count_iter(0);
    (void)cub::DeviceScan::InclusiveScan(nullptr, value_storage_bytes, input_ptr, value_ptr, value_fun, axis_size_st,
                                         cuda_stream);
    (void)cub::DeviceScan::InclusiveScan(nullptr, index_storage_bytes, count_iter, index_ptr, index_fun, axis_size_st,
                                         cuda_stream);
    // Here only allocate once.
    char *temp_storage_ptr = nullptr;
    (void)cudaMalloc(&temp_storage_ptr, value_storage_bytes + index_storage_bytes);
    void *value_storage_ptr = reinterpret_cast<void *>(temp_storage_ptr);
    void *index_storage_ptr = reinterpret_cast<void *>(temp_storage_ptr + value_storage_bytes);

    (void)cub::DeviceScan::InclusiveScan(value_storage_ptr, value_storage_bytes, input_ptr, value_ptr, value_fun,
                                         axis_size_st, cuda_stream);
    (void)cub::DeviceScan::InclusiveScan(index_storage_ptr, index_storage_bytes, count_iter, index_ptr, index_fun,
                                         axis_size_st, cuda_stream);
    (void)cudaFree(temp_storage_ptr);
  } else {
    auto outer_size = static_cast<uint>(outer_size_st);
    auto inner_size = static_cast<uint>(inner_size_st);
    auto axis_size = static_cast<uint>(axis_size_st);
    auto outer_inner_size = outer_size * inner_size;
    auto axis_inner_size = axis_size * inner_size;
    if (inner_size_st == 1) {
      // The partitioning strategy is as follows:
      // 1. The block has two dimensions, the y dimension with max size is 128, scan an array with axis_size, while the
      // other one is used to process batch dimension on parallel, and the specific size depends on the max size of
      // shared memory and max threads number.
      // 2. The gird has only one dimension, which requires to take over the remaining batch dimension.
      constexpr uint max_block_y = 128;
      uint max_share_size = GetMaxSharedMemoryPerBlock(device_id);
      uint max_thread_size = GetMaxThreadsPerBlock(device_id);
      uint max_grid_size = GetMaxGridDimX(device_id);
      uint axis_power2 = 1u << Log2Ceil(axis_size);
      uint block_y = std::min(max_block_y, axis_power2);
      uint has_allocate = block_y * 2 * (sizeof(DataType) + sizeof(IndexType));
      uint block_x = std::min(max_thread_size / block_y, max_share_size / has_allocate);
      uint grid_x = std::min(max_grid_size, UP_DIV(outer_inner_size, block_x));
      dim3 block = {block_x, block_y};
      dim3 grid = {grid_x};
      uint share_size = block_x * has_allocate;
      CumMinMaxKernel<BinaryFunctor, DataType, IndexType><<<grid, block, share_size, cuda_stream>>>(
        fun, input_ptr, value_ptr, index_ptr, axis_size, inner_size, axis_inner_size, outer_inner_size, init);
    } else {
      // A useless case. If you don't like this branch, please delete it.
      CumMinMaxSlowKernel<BinaryFunctor, DataType, IndexType>
        <<<CUDA_BLOCKS(device_id, outer_inner_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
          fun, input_ptr, value_ptr, index_ptr, axis_size, inner_size, axis_inner_size, outer_inner_size);
    }
  }
}

template <typename DataType, typename IndexType>
CUDA_LIB_EXPORT void CumMinMax(CumOpType cum_op_type, const DataType *input_ptr, DataType *value_ptr,
                               IndexType *index_ptr, size_t outer_size_st, size_t axis_size_st, size_t inner_size_st,
                               const uint32_t &device_id, cudaStream_t cuda_stream) {
  switch (cum_op_type) {
    case CUMMIN: {
      KernelHelper(BinaryFunctor<thrust::less_equal<DataType>, DataType>{}, NumericMax<DataType>(), input_ptr,
                   value_ptr, index_ptr, outer_size_st, axis_size_st, inner_size_st, device_id, cuda_stream);
      break;
    }
    case CUMMAX: {
      KernelHelper(BinaryFunctor<thrust::greater_equal<DataType>, DataType>{}, NumericMin<DataType>(), input_ptr,
                   value_ptr, index_ptr, outer_size_st, axis_size_st, inner_size_st, device_id, cuda_stream);
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
