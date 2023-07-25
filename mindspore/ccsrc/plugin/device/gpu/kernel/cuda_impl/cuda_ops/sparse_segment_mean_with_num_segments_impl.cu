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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sparse_segment_mean_with_num_segments_impl.cuh"
#include <algorithm>
#include "plugin/device/cpu/kernel/nnacl/op_base.h"

template <typename IndexType>
__global__ void SparseSegmentPosKernel(const IndexType *segment_ids_ptr, size_t *segment_pos_ptr, size_t indices_size,
                                       size_t segment_size) {
  for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id <= indices_size; id += blockDim.x * gridDim.x) {
    // We don't check whether the segment indices is sorted
    // and ignore the segment indices that are out of range[0, segment_size),
    // while in CPU platform kernel raises runtime error.
    const IndexType max_size = static_cast<IndexType>(segment_size);
    const IndexType min_size = IndexType(0);
    IndexType beg_idx = (id == 0) ? min_size : segment_ids_ptr[id - 1] + 1;
    IndexType end_idx = (id >= indices_size) ? max_size : segment_ids_ptr[id];
    beg_idx = max(min_size, min(max_size, beg_idx));
    end_idx = max(min_size, min(max_size, end_idx));
    for (IndexType i = beg_idx; i <= end_idx; i++) {
      segment_pos_ptr[i] = id;
    }
  }
}

template <typename DataType>
__device__ DataType ReduceWithinBlock(const DataType &value) {
  // Refer to reduce3 from Mark Harris, et al. 'Optimizing Parallel Reduction in CUDA'.
  extern __shared__ __align__(16) char share_data[];
  DataType *share_data_ptr = reinterpret_cast<DataType *>(share_data);
  const unsigned int x = threadIdx.x;
  const unsigned int y = threadIdx.y;
  const unsigned int tid = y * blockDim.x + x;
  share_data_ptr[tid] = value;
  __syncthreads();
  // Reduce over the y dimension of the block.
  for (unsigned k = blockDim.y / 2; k > 0; k /= 2) {
    if (y < k) {
      share_data_ptr[tid] += share_data_ptr[(y + k) * blockDim.x + x];
    }
    __syncthreads();
  }
  return share_data_ptr[tid];
}

template <typename DataType, typename IndexType>
__global__ void SparseSegmentMeanWithNumSegmentsKernel(const DataType *x_ptr, const IndexType *indices_ptr,
                                                       const size_t *segment_pos_ptr, DataType *y_ptr,
                                                       size_t outer_size, size_t inner_size, size_t segment_size) {
  size_t num_blocks = (inner_size - 1) / blockDim.x + 1;
  for (size_t bid = blockIdx.x; bid < num_blocks; bid += gridDim.x) {
    size_t inner_idx = threadIdx.x + bid * blockDim.x;
    bool inner_valid = inner_idx < inner_size;
    for (size_t sid = blockIdx.y; sid < segment_size; sid += gridDim.y) {
      size_t beg_pos = segment_pos_ptr[sid];
      size_t end_pos = segment_pos_ptr[sid + 1];
      // Store the mean of a segment.
      DataType segment_sum = 0;
      DataType segment_len = DataType(static_cast<double>(end_pos - beg_pos));
      for (size_t pos = beg_pos; pos < end_pos; pos += blockDim.y) {
        size_t index_id = pos + threadIdx.y;
        bool index_valid = index_id < end_pos;
        DataType reduce_result = 0;
        IndexType index = inner_valid && index_valid ? indices_ptr[index_id] : outer_size;
        // Similarly, we ignore the invalid index and don't raise error.
        if (index >= 0 && index < outer_size) {
          reduce_result = x_ptr[index * inner_size + inner_idx];
        }
        reduce_result = inner_valid ? reduce_result : ReduceWithinBlock(reduce_result);
        if (threadIdx.y == 0 && inner_valid) {
          segment_sum += reduce_result;
        }
      }
      if (threadIdx.y == 0 && inner_valid) {
        y_ptr[sid * inner_size + inner_idx] = beg_pos == end_pos ? DataType(0) : segment_sum / segment_len;
      }
    }
  }
}

inline int Log2Floor(uint32_t n) {
  if (n == 0) return -1;
  int log = 0;
  for (int i = 4; i >= 0; --i) {
    int shift = (1 << i);
    uint32_t x = n >> shift;
    if (x) {
      n = x;
      log += shift;
    }
  }
  return log;
}

inline int Log2Floor64(uint64_t n) {
  // Scan n first high 32 then low 32 bits.
  const uint32_t high_32_bit = static_cast<uint32_t>(n >> 32);
  if (high_32_bit == 0) {
    return Log2Floor(static_cast<uint32_t>(n));
  } else {
    return 32 + Log2Floor(high_32_bit);
  }
}

inline int Log2Ceil64(uint64_t n) {
  int floor = Log2Floor64(n);
  if (n == (n & ~(n - 1)))
    return floor;
  else
    return floor + 1;
}

template <typename IndexType>
__global__ void InputValidCheck(IndexType *indices_ptr, IndexType *segment_ids_ptr, IndexType *num_segments_ptr,
                                size_t outer_size, int *ret_flag, size_t indices_size) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indices_size; i += gridDim.x * blockDim.x) {
    if ((i != indices_size - 1) && (segment_ids_ptr[i] > segment_ids_ptr[i + 1])) {
      *ret_flag = 1;
      return;
    }
    if (segment_ids_ptr[indices_size - 1] >= num_segments_ptr[0]) {
      *ret_flag = 2;
      return;
    }
    if (indices_ptr[i] >= outer_size) {
      *ret_flag = 3;
      return;
    }
  }
}

template <typename DataType, typename IndexType>
CUDA_LIB_EXPORT cudaError_t CalSparseSegmentMeanWithNumSegments(
  const DataType *x_ptr, const IndexType *indices_ptr, const IndexType *segment_ids_ptr,
  const IndexType *num_segments_ptr, size_t *segment_pos_ptr, DataType *y_ptr, size_t outer_size, size_t inner_size,
  size_t indices_size, size_t segment_size, size_t x_size, size_t y_size, size_t batch_size, int *ret_flag_device,
  uint32_t device_id, cudaStream_t cuda_stream, int *ret_flag_host) {
  // Get start position of each segment and set to segment_pos_ptr.
  // The last element of segment_pos_ptr must equal to segment_size.
  int thread_num = indices_size + 1 > 256 ? 256 : (indices_size + 1);
  (void)cudaMemsetAsync(ret_flag_device, 0, sizeof(int), cuda_stream);
  InputValidCheck<<<CUDA_BLOCKS_CAL(device_id, indices_size + 1, thread_num), thread_num, 0, cuda_stream>>>(
    indices_ptr, segment_ids_ptr, num_segments_ptr, outer_size, ret_flag_device, indices_size);
  (void)cudaMemcpyAsync(ret_flag_host, ret_flag_device, sizeof(int), cudaMemcpyDeviceToHost, cuda_stream);
  cudaStreamSynchronize(cuda_stream);
  if (*ret_flag_host != 0) {
    return cudaErrorNotReady;
  }
  SparseSegmentPosKernel<<<CUDA_BLOCKS_CAL(device_id, indices_size + 1, thread_num), thread_num, 0, cuda_stream>>>(
    segment_ids_ptr, segment_pos_ptr, indices_size, segment_size);

  const unsigned int max_grid_x = (1u << 31) - 1;
  const unsigned int max_grid_y = (1u << 16) - 1;
  const unsigned int max_block_x = 64;
  const unsigned int max_block_y = 8;
  unsigned int inner_power2 = 1u << Log2Ceil64(inner_size);
  unsigned int avg_reduce_size = UP_DIV(outer_size, segment_size);
  unsigned int avg_reduce_size_power2 = 1u << Log2Ceil64(avg_reduce_size);
  unsigned int block_x = std::min(inner_power2, max_block_x);
  unsigned int block_y = std::min(avg_reduce_size_power2, UP_DIV(max_block_y, block_x));
  unsigned int grid_x = std::min(static_cast<unsigned int>(UP_DIV(inner_size, block_x)), max_grid_x);
  unsigned int grid_y = std::min(static_cast<unsigned int>(segment_size), max_grid_y);
  dim3 block(block_x, block_y);
  dim3 grid(grid_x, grid_y);
  unsigned int shared_memory_size = block_x * block_y * sizeof(DataType);
  // Reduce each segment along the indices of first dimension.
  for (size_t i = 0; i < batch_size; i++) {
    auto batch_x_ptr = x_ptr + i * x_size;
    auto batch_y_ptr = y_ptr + i * y_size;
    auto batch_indices_ptr = indices_ptr + i * indices_size;
    SparseSegmentMeanWithNumSegmentsKernel<<<grid, block, shared_memory_size, cuda_stream>>>(
      batch_x_ptr, batch_indices_ptr, segment_pos_ptr, batch_y_ptr, outer_size, inner_size, segment_size);
  }
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalSparseSegmentMeanWithNumSegments<half, int32_t>(
  const half *x_ptr, const int32_t *indices_ptr, const int32_t *segment_ids_ptr, const int32_t *num_segments_ptr,
  size_t *segment_pos_ptr, half *y_ptr, size_t outer_size, size_t inner_size, size_t indices_size, size_t segment_size,
  size_t x_size, size_t y_size, size_t batch_size, int *ret_flag_device, uint32_t device_id, cudaStream_t cuda_stream,
  int *ret_flag_host);
template CUDA_LIB_EXPORT cudaError_t CalSparseSegmentMeanWithNumSegments<float, int32_t>(
  const float *x_ptr, const int32_t *indices_ptr, const int32_t *segment_ids_ptr, const int32_t *num_segments_ptr,
  size_t *segment_pos_ptr, float *y_ptr, size_t outer_size, size_t inner_size, size_t indices_size, size_t segment_size,
  size_t x_size, size_t y_size, size_t batch_size, int *ret_flag_device, uint32_t device_id, cudaStream_t cuda_stream,
  int *ret_flag_host);
template CUDA_LIB_EXPORT cudaError_t CalSparseSegmentMeanWithNumSegments<double, int32_t>(
  const double *x_ptr, const int32_t *indices_ptr, const int32_t *segment_ids_ptr, const int32_t *num_segments_ptr,
  size_t *segment_pos_ptr, double *y_ptr, size_t outer_size, size_t inner_size, size_t indices_size,
  size_t segment_size, size_t x_size, size_t y_size, size_t batch_size, int *ret_flag_device, uint32_t device_id,
  cudaStream_t cuda_stream, int *ret_flag_host);
template CUDA_LIB_EXPORT cudaError_t CalSparseSegmentMeanWithNumSegments<half, int64_t>(
  const half *x_ptr, const int64_t *indices_ptr, const int64_t *segment_ids_ptr, const int64_t *num_segments_ptr,
  size_t *segment_pos_ptr, half *y_ptr, size_t outer_size, size_t inner_size, size_t indices_size, size_t segment_size,
  size_t x_size, size_t y_size, size_t batch_size, int *ret_flag_device, uint32_t device_id, cudaStream_t cuda_stream,
  int *ret_flag_host);
template CUDA_LIB_EXPORT cudaError_t CalSparseSegmentMeanWithNumSegments<float, int64_t>(
  const float *x_ptr, const int64_t *indices_ptr, const int64_t *segment_ids_ptr, const int64_t *num_segments_ptr,
  size_t *segment_pos_ptr, float *y_ptr, size_t outer_size, size_t inner_size, size_t indices_size, size_t segment_size,
  size_t x_size, size_t y_size, size_t batch_size, int *ret_flag_device, uint32_t device_id, cudaStream_t cuda_stream,
  int *ret_flag_host);
template CUDA_LIB_EXPORT cudaError_t CalSparseSegmentMeanWithNumSegments<double, int64_t>(
  const double *x_ptr, const int64_t *indices_ptr, const int64_t *segment_ids_ptr, const int64_t *num_segments_ptr,
  size_t *segment_pos_ptr, double *y_ptr, size_t outer_size, size_t inner_size, size_t indices_size,
  size_t segment_size, size_t x_size, size_t y_size, size_t batch_size, int *ret_flag_device, uint32_t device_id,
  cudaStream_t cuda_stream, int *ret_flag_host);
