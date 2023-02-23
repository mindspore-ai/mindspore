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
#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <algorithm>
#include <limits>
#include "include/cuda_fp16.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"

namespace {

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

int GetMaxGridDimY(const uint32_t &device_id) {
  int max_size = 1 << 16;
  (void)cudaDeviceGetAttribute(&max_size, cudaDevAttrMaxGridDimY, static_cast<int>(device_id));
  return max_size;
}
}  // namespace

template <typename DataType>
__device__ __forceinline__ bool IsNan(const DataType &x) {
  return isnan(x);
}

__device__ __forceinline__ bool IsNan(const half &x) { return __hisnan(x); }

template <typename BinaryOp, typename DataType>
struct BinaryFunctor {
  BinaryOp op_;
  __device__ __forceinline__ bool operator()(DataType lhs, DataType rhs) {
    return (IsNan(lhs) || !op_(rhs, lhs)) && !IsNan(rhs);
  }
};

template <typename BinaryFunctor, typename TupleType>
struct BlockScanFunctor {
  BinaryFunctor functor_;
  explicit BlockScanFunctor(BinaryFunctor functor) : functor_(functor) {}
  __device__ __forceinline__ TupleType operator()(TupleType lhs, TupleType rhs) {
    return functor_(thrust::get<0>(lhs), thrust::get<0>(rhs)) ? lhs : rhs;
  }
};

// Inspired by cub documentation.
template <typename BlockScanFunctor, typename TupleType>
struct BlockPrefixCallbackFunctor {
  BlockScanFunctor functor_;
  TupleType block_aggregate_;
  // Constructor
  __device__ BlockPrefixCallbackFunctor(BlockScanFunctor functor, TupleType block_aggregate)
      : functor_(functor), block_aggregate_(block_aggregate) {}
  // Callback operator to be entered by the first warp of threads in the block.
  // Thread-0 is responsible for returning a value for seeding the block-wide scan.
  __device__ __forceinline__ TupleType operator()(TupleType block_aggregate) {
    TupleType old_block_aggregate = block_aggregate_;
    block_aggregate_ = functor_(old_block_aggregate, block_aggregate);
    return old_block_aggregate;
  }
};

#ifndef _WIN32
template <typename BlockScanFunctor, typename ValueType, typename IndexType, uint BlockDim>
__global__ void LargeBlockScanKernel(BlockScanFunctor functor, const ValueType *input_ptr, ValueType *value_ptr,
                                     IndexType *index_ptr, uint axis_size, uint inner_size, uint axis_inner_size,
                                     uint outer_inner_size, ValueType init) {
  typedef thrust::tuple<ValueType, IndexType> DataType;
  typedef cub::BlockScan<DataType, BlockDim> BlockScan;
  __shared__ typename BlockScan::TempStorage share_data;
  for (uint bid = blockIdx.x; bid < outer_inner_size; bid += gridDim.x) {
    uint outer_idx = bid / inner_size;
    uint inner_idx = bid % inner_size;
    DataType init_data{init, 0};
    BlockPrefixCallbackFunctor<BlockScanFunctor, DataType> cb_functor{functor, init_data};
    uint axis_idx = threadIdx.x;
    uint axis_offset = outer_idx * axis_inner_size + inner_idx + axis_idx * inner_size;
    for (uint block_offset = 0; block_offset < axis_size; block_offset += BlockDim) {
      DataType thread_data = init_data;
      if (axis_idx < axis_size) {
        thread_data = thrust::make_tuple(input_ptr[axis_offset], axis_idx);
      }
      BlockScan(share_data).template InclusiveScan(thread_data, thread_data, functor, cb_functor);
      __syncthreads();
      if (axis_idx < axis_size) {
        thrust::tie(value_ptr[axis_offset], index_ptr[axis_offset]) = thread_data;
      }
      axis_idx += BlockDim;
      axis_offset += BlockDim * inner_size;
    }
  }
}
#endif

template <typename BlockScanFunctor, typename ValueType, typename IndexType, uint BlockDimX, uint BlockDimY>
__global__ void ScanInnerMostDimKernel(BlockScanFunctor functor, const ValueType *input_ptr, ValueType *value_ptr,
                                       IndexType *index_ptr, uint outer_size, uint axis_size, ValueType init) {
  typedef thrust::tuple<ValueType, IndexType> DataType;
  constexpr uint scan_per_block = BlockDimX * 2;
  __shared__ ValueType share_value[BlockDimY][scan_per_block];
  __shared__ IndexType share_index[BlockDimY][scan_per_block];
  auto share_value_ptr = share_value[threadIdx.y];
  auto share_index_ptr = share_index[threadIdx.y];
  for (uint bid = blockIdx.x * blockDim.y; bid < outer_size; bid += gridDim.x * blockDim.y) {
    uint outer_idx = bid + threadIdx.y;
    bool is_valid = outer_idx < outer_size;
    uint offset = outer_idx * axis_size;
    DataType block_data{init, 0};
    // The following parallel scan algorithm refers to:
    // Figure 9.7 from David B. Kirk, et al. 'Programming Massively Parallel Processors'.
    for (uint i = 0; i < axis_size; i += scan_per_block) {
      // Initializing share memory with input value, and each thread process two elements.
      uint idx1 = threadIdx.x + i;
      uint idx2 = idx1 + BlockDimX;
      if (is_valid) {
        if (idx1 < axis_size) {
          share_value_ptr[threadIdx.x] = input_ptr[offset + idx1];
          share_index_ptr[threadIdx.x] = idx1;
        } else {
          share_value_ptr[threadIdx.x] = init;
        }
        if (idx2 < axis_size) {
          share_value_ptr[threadIdx.x + BlockDimX] = input_ptr[offset + idx2];
          share_index_ptr[threadIdx.x + BlockDimX] = idx2;
        } else {
          share_value_ptr[threadIdx.x + BlockDimX] = init;
        }
        // update with previous block result.
        if (threadIdx.x == 0) {
          thrust::tie(share_value_ptr[0], share_index_ptr[0]) =
            functor(thrust::make_tuple(share_value_ptr[0], share_index_ptr[0]), block_data);
        }
      }
      // up-sweep
      for (uint stride = 1; stride < scan_per_block; stride <<= 1) {
        uint index = (threadIdx.x + 1) * (stride << 1) - 1;
        if (is_valid && index < scan_per_block) {
          thrust::tie(share_value_ptr[index], share_index_ptr[index]) =
            functor(thrust::make_tuple(share_value_ptr[index - stride], share_index_ptr[index - stride]),
                    thrust::make_tuple(share_value_ptr[index], share_index_ptr[index]));
        }
      }
      // down-sweep
      for (uint stride = scan_per_block >> 2; stride > 0; stride >>= 1) {
        uint index = (threadIdx.x + 1) * (stride << 1) - 1;
        if (is_valid && index + stride < scan_per_block) {
          thrust::tie(share_value_ptr[index + stride], share_index_ptr[index + stride]) =
            functor(thrust::make_tuple(share_value_ptr[index], share_index_ptr[index]),
                    thrust::make_tuple(share_value_ptr[index + stride], share_index_ptr[index + stride]));
        }
      }
      // write to output.
      if (is_valid) {
        if (idx1 < axis_size) {
          value_ptr[offset + idx1] = share_value_ptr[threadIdx.x];
          index_ptr[offset + idx1] = share_index_ptr[threadIdx.x];
        }
        if (idx2 < axis_size) {
          value_ptr[offset + idx2] = share_value_ptr[threadIdx.x + BlockDimX];
          index_ptr[offset + idx2] = share_index_ptr[threadIdx.x + BlockDimX];
        }
        // update block_data
        block_data = thrust::make_tuple(share_value_ptr[scan_per_block - 1], share_index_ptr[scan_per_block - 1]);
      }
    }
  }
}

template <typename BlockScanFunctor, typename ValueType, typename IndexType>
__global__ void ScanOuterDimKernel(BlockScanFunctor functor, const ValueType *input_ptr, ValueType *value_ptr,
                                   IndexType *index_ptr, uint axis_size, uint inner_size, uint axis_inner_size,
                                   uint outer_inner_size, ValueType init) {
  typedef thrust::tuple<ValueType, IndexType> DataType;
  for (uint bid = blockIdx.x * blockDim.x + threadIdx.x; bid < outer_inner_size; bid += gridDim.x * blockDim.x) {
    uint outer_idx = bid / inner_size;
    uint inner_idx = bid % inner_size;
    DataType out{init, 0};
    uint offset = outer_idx * axis_inner_size + inner_idx;
    for (uint i = 0; i < axis_size; i++) {
      DataType thread_data = thrust::make_tuple(input_ptr[offset], i);
      out = functor(out, thread_data);
      thrust::tie(value_ptr[offset], index_ptr[offset]) = out;
      offset += inner_size;
    }
  }
}

template <typename BinaryFunctor, typename ValueType, typename IndexType>
void KernelHelper(BinaryFunctor functor, ValueType init, const ValueType *input_ptr, ValueType *value_ptr,
                  IndexType *index_ptr, size_t outer_size_st, size_t axis_size_st, size_t inner_size_st,
                  const uint32_t &device_id, cudaStream_t cuda_stream) {
  auto outer_size = static_cast<uint>(outer_size_st);
  auto inner_size = static_cast<uint>(inner_size_st);
  auto axis_size = static_cast<uint>(axis_size_st);
  auto outer_inner_size = outer_size * inner_size;
  auto axis_inner_size = axis_size * inner_size;
  uint max_grid_size = GetMaxGridDimY(device_id);
  typedef BlockScanFunctor<BinaryFunctor, thrust::tuple<ValueType, IndexType>> BlockScanFunctor;
  BlockScanFunctor scan_op{functor};
#if defined(CUB_VERSION) && (CUB_VERSION > 100800) && !defined(_WIN32)
  // Special case where only one dimension that needs to compute, so using cub library is the most efficient way.
  if (outer_size == 1 && inner_size == 1) {
    // Using thrust::zip_iterator to make an iterator for (ValueType, IndexType).
    cub::CountingInputIterator<IndexType> count_iter(0);
    typedef typename thrust::detail::normal_iterator<const ValueType *> InputValueIterator;
    typedef cub::CountingInputIterator<IndexType> InputIndexIterator;
    typedef thrust::zip_iterator<thrust::tuple<InputValueIterator, InputIndexIterator>> InputZipIterator;
    InputZipIterator input_iter(thrust::make_tuple(input_ptr, count_iter));

    typedef typename thrust::detail::normal_iterator<ValueType *> OutputValueIterator;
    typedef typename thrust::detail::normal_iterator<IndexType *> OutputIndexIterator;
    typedef thrust::zip_iterator<thrust::tuple<OutputValueIterator, OutputIndexIterator>> OutputZipIterator;
    OutputZipIterator output_iter(thrust::make_tuple(value_ptr, index_ptr));

    // Calculate the size of temporary storage.
    size_t temp_storage_bytes = 0;
    (void)cub::DeviceScan::InclusiveScan(nullptr, temp_storage_bytes, input_iter, output_iter, scan_op, axis_size,
                                         cuda_stream);
    // Allocate temporary storage.
    char *temp_storage_ptr = nullptr;
    (void)cudaMalloc(&temp_storage_ptr, temp_storage_bytes);
    // Core computation process.
    (void)cub::DeviceScan::InclusiveScan(temp_storage_ptr, temp_storage_bytes, input_iter, output_iter, scan_op,
                                         axis_size, cuda_stream);
    (void)cudaFree(temp_storage_ptr);
    return;
  }
  // When computing capacity of CUDA is not recommended (<7), we instead use self-implemented scan algorithm.
  // Otherwise, we use cub::BlockScan, which is faster than self-implemented one.
  const int major_sm = GET_MAJOR_SM;
  const bool check_sm = mindspore::device::gpu::CudaCommon::GetInstance().check_sm();
  constexpr uint threshold_large_scan_dim = 500;
  if (!(check_sm && major_sm < RECOMMEND_SM) && axis_size > threshold_large_scan_dim) {
    constexpr uint block_dim = 512;
    uint grid_x = std::min(outer_inner_size, max_grid_size);
    dim3 block{block_dim};
    dim3 grid{grid_x};
    LargeBlockScanKernel<BlockScanFunctor, ValueType, IndexType, block_dim><<<grid, block, 0, cuda_stream>>>(
      scan_op, input_ptr, value_ptr, index_ptr, axis_size, inner_size, axis_inner_size, outer_inner_size, init);
    return;
  }
#endif
  if (inner_size == 1) {
    constexpr uint block_dim_x = 32;
    constexpr uint block_dim_y = 16;
    // The reason why x-dimension of block is set to 32:
    // Each thread process 2 elements, so each x-dimension of block process 64 elements. An obvious advantage is no
    // bank conflict. In addition, we don't need `__syncthreads`, since 32 is equal to warp size.
    uint grid_x = std::min(UP_DIV(outer_size, block_dim_y), max_grid_size);
    dim3 block = {block_dim_x, block_dim_y};
    dim3 grid = {grid_x};
    ScanInnerMostDimKernel<BlockScanFunctor, ValueType, IndexType, block_dim_x, block_dim_y>
      <<<grid, block, 0, cuda_stream>>>(scan_op, input_ptr, value_ptr, index_ptr, outer_size, axis_size, init);
  } else {
    constexpr uint block_dim = 512;
    uint grid_x = std::min(UP_DIV(outer_inner_size, block_dim), max_grid_size);
    dim3 block{block_dim};
    dim3 grid{grid_x};
    ScanOuterDimKernel<<<grid, block, 0, cuda_stream>>>(scan_op, input_ptr, value_ptr, index_ptr, axis_size, inner_size,
                                                        axis_inner_size, outer_inner_size, init);
  }
}

template <typename DataType, typename IndexType>
cudaError_t CumMinMax(CumOpType cum_op_type, const DataType *input_ptr, DataType *value_ptr, IndexType *index_ptr,
                      size_t outer_size_st, size_t axis_size_st, size_t inner_size_st, const uint32_t &device_id,
                      cudaStream_t cuda_stream) {
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
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template CUDA_LIB_EXPORT cudaError_t CumMinMax<int8_t, int32_t>(CumOpType cum_op_type, const int8_t *input_ptr,
                                                                int8_t *value_ptr, int32_t *index_ptr,
                                                                size_t outer_size, size_t axis_size, size_t inner_size,
                                                                const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CumMinMax<int16_t, int32_t>(CumOpType cum_op_type, const int16_t *input_ptr,
                                                                 int16_t *value_ptr, int32_t *index_ptr,
                                                                 size_t outer_size, size_t axis_size, size_t inner_size,
                                                                 const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CumMinMax<int32_t, int32_t>(CumOpType cum_op_type, const int32_t *input_ptr,
                                                                 int32_t *value_ptr, int32_t *index_ptr,
                                                                 size_t outer_size, size_t axis_size, size_t inner_size,
                                                                 const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CumMinMax<int64_t, int32_t>(CumOpType cum_op_type, const int64_t *input_ptr,
                                                                 int64_t *value_ptr, int32_t *index_ptr,
                                                                 size_t outer_size, size_t axis_size, size_t inner_size,
                                                                 const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CumMinMax<uint8_t, int32_t>(CumOpType cum_op_type, const uint8_t *input_ptr,
                                                                 uint8_t *value_ptr, int32_t *index_ptr,
                                                                 size_t outer_size, size_t axis_size, size_t inner_size,
                                                                 const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CumMinMax<uint16_t, int32_t>(CumOpType cum_op_type, const uint16_t *input_ptr,
                                                                  uint16_t *value_ptr, int32_t *index_ptr,
                                                                  size_t outer_size, size_t axis_size,
                                                                  size_t inner_size, const uint32_t &device_id,
                                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CumMinMax<uint32_t, int32_t>(CumOpType cum_op_type, const uint32_t *input_ptr,
                                                                  uint32_t *value_ptr, int32_t *index_ptr,
                                                                  size_t outer_size, size_t axis_size,
                                                                  size_t inner_size, const uint32_t &device_id,
                                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CumMinMax<uint64_t, int32_t>(CumOpType cum_op_type, const uint64_t *input_ptr,
                                                                  uint64_t *value_ptr, int32_t *index_ptr,
                                                                  size_t outer_size, size_t axis_size,
                                                                  size_t inner_size, const uint32_t &device_id,
                                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CumMinMax<half, int32_t>(CumOpType cum_op_type, const half *input_ptr,
                                                              half *value_ptr, int32_t *index_ptr, size_t outer_size,
                                                              size_t axis_size, size_t inner_size,
                                                              const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CumMinMax<float, int32_t>(CumOpType cum_op_type, const float *input_ptr,
                                                               float *value_ptr, int32_t *index_ptr, size_t outer_size,
                                                               size_t axis_size, size_t inner_size,
                                                               const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CumMinMax<double, int32_t>(CumOpType cum_op_type, const double *input_ptr,
                                                                double *value_ptr, int32_t *index_ptr,
                                                                size_t outer_size, size_t axis_size, size_t inner_size,
                                                                const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CumMinMax<int8_t, int64_t>(CumOpType cum_op_type, const int8_t *input_ptr,
                                                                int8_t *value_ptr, int64_t *index_ptr,
                                                                size_t outer_size, size_t axis_size, size_t inner_size,
                                                                const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CumMinMax<int16_t, int64_t>(CumOpType cum_op_type, const int16_t *input_ptr,
                                                                 int16_t *value_ptr, int64_t *index_ptr,
                                                                 size_t outer_size, size_t axis_size, size_t inner_size,
                                                                 const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CumMinMax<int32_t, int64_t>(CumOpType cum_op_type, const int32_t *input_ptr,
                                                                 int32_t *value_ptr, int64_t *index_ptr,
                                                                 size_t outer_size, size_t axis_size, size_t inner_size,
                                                                 const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CumMinMax<int64_t, int64_t>(CumOpType cum_op_type, const int64_t *input_ptr,
                                                                 int64_t *value_ptr, int64_t *index_ptr,
                                                                 size_t outer_size, size_t axis_size, size_t inner_size,
                                                                 const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CumMinMax<uint8_t, int64_t>(CumOpType cum_op_type, const uint8_t *input_ptr,
                                                                 uint8_t *value_ptr, int64_t *index_ptr,
                                                                 size_t outer_size, size_t axis_size, size_t inner_size,
                                                                 const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CumMinMax<uint16_t, int64_t>(CumOpType cum_op_type, const uint16_t *input_ptr,
                                                                  uint16_t *value_ptr, int64_t *index_ptr,
                                                                  size_t outer_size, size_t axis_size,
                                                                  size_t inner_size, const uint32_t &device_id,
                                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CumMinMax<uint32_t, int64_t>(CumOpType cum_op_type, const uint32_t *input_ptr,
                                                                  uint32_t *value_ptr, int64_t *index_ptr,
                                                                  size_t outer_size, size_t axis_size,
                                                                  size_t inner_size, const uint32_t &device_id,
                                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CumMinMax<uint64_t, int64_t>(CumOpType cum_op_type, const uint64_t *input_ptr,
                                                                  uint64_t *value_ptr, int64_t *index_ptr,
                                                                  size_t outer_size, size_t axis_size,
                                                                  size_t inner_size, const uint32_t &device_id,
                                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CumMinMax<half, int64_t>(CumOpType cum_op_type, const half *input_ptr,
                                                              half *value_ptr, int64_t *index_ptr, size_t outer_size,
                                                              size_t axis_size, size_t inner_size,
                                                              const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CumMinMax<float, int64_t>(CumOpType cum_op_type, const float *input_ptr,
                                                               float *value_ptr, int64_t *index_ptr, size_t outer_size,
                                                               size_t axis_size, size_t inner_size,
                                                               const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CumMinMax<double, int64_t>(CumOpType cum_op_type, const double *input_ptr,
                                                                double *value_ptr, int64_t *index_ptr,
                                                                size_t outer_size, size_t axis_size, size_t inner_size,
                                                                const uint32_t &device_id, cudaStream_t cuda_stream);
