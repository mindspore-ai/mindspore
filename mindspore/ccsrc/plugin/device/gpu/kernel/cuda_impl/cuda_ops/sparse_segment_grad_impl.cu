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

#include <algorithm>
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sparse_segment_grad_impl.cuh"

template <typename S>
__global__ void SparseSegmentPosKernel(const S *indices_ptr, size_t *indices_pos_ptr, size_t idx_seg_size,
                                       size_t outer_size) {
  for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id <= idx_seg_size; id += blockDim.x * gridDim.x) {
    const S max_size = static_cast<S>(outer_size);
    const S min_size = S(0);
    S beg_idx = (id == 0) ? min_size : indices_ptr[id - 1] + 1;
    S end_idx = (id >= idx_seg_size) ? max_size : indices_ptr[id];
    beg_idx = max(min_size, min(max_size, beg_idx));
    end_idx = max(min_size, min(max_size, end_idx));
    for (S i = beg_idx; i <= end_idx; i++) {
      indices_pos_ptr[i] = id;
    }
  }
}

template <typename R, typename S>
__global__ void SparseSegmentSumGradKernel(const R *grad_ptr, const S *indices_ptr, const S *segment_ids_ptr,
                                           const size_t *indices_pos_ptr, size_t outer_size, size_t inner_size,
                                           size_t output_dim0, R *y_ptr) {
  size_t num_blocks = (inner_size - 1) / blockDim.x + 1;
  for (size_t bid = blockIdx.x; bid < num_blocks; bid += gridDim.x) {
    size_t inner_idx = threadIdx.x + bid * blockDim.x;
    bool inner_valid = inner_idx < inner_size;
    for (size_t inid = blockIdx.y; inid < outer_size; inid += gridDim.y) {
      size_t beg_pos = indices_pos_ptr[inid];
      size_t end_pos = indices_pos_ptr[inid + 1];
      for (size_t pos = beg_pos; pos < end_pos; pos += 1) {
        double reduce_result = 0;
        S index = inner_valid ? indices_ptr[pos] : outer_size;
        if (index >= 0 && index < outer_size) {
          reduce_result = static_cast<double>(grad_ptr[index * inner_size + inner_idx]);
        }
        if (threadIdx.y == 0 && inner_valid) {
          R *out_pos = y_ptr + segment_ids_ptr[pos] * inner_size + inner_idx;
          MsAtomicAdd(out_pos, static_cast<R>(reduce_result));
        }
      }
    }
  }
}

template <typename S>
__global__ void SparseSegmentSumGradKernel(const half *grad_ptr, const S *indices_ptr, const S *segment_ids_ptr,
                                           const size_t *indices_pos_ptr, size_t outer_size, size_t inner_size,
                                           size_t output_dim0, half *y_ptr) {
  size_t num_blocks = (inner_size - 1) / blockDim.x + 1;
  for (size_t bid = blockIdx.x; bid < num_blocks; bid += gridDim.x) {
    size_t inner_idx = threadIdx.x + bid * blockDim.x;
    bool inner_valid = inner_idx < inner_size;
    for (size_t inid = blockIdx.y; inid < outer_size; inid += gridDim.y) {
      size_t beg_pos = indices_pos_ptr[inid];
      size_t end_pos = indices_pos_ptr[inid + 1];
      for (size_t pos = beg_pos; pos < end_pos; pos += 1) {
        double reduce_result = 0;
        S index = inner_valid ? indices_ptr[pos] : outer_size;
        if (index >= 0 && index < outer_size) {
          reduce_result = static_cast<double>(__half2float(grad_ptr[index * inner_size + inner_idx]));
        }
        if (threadIdx.y == 0 && inner_valid) {
          half *out_pos = y_ptr + segment_ids_ptr[pos]* inner_size + inner_idx;
          MsAtomicAdd(out_pos, __float2half(static_cast<float>(reduce_result)));
        }
      }
    }
  }
}

template <typename R, typename S>
__global__ void SparseSegmentSqrtNGradKernel(const R *grad_ptr, const S *indices_ptr, const S *segment_ids_ptr,
                                             const size_t *indices_pos_ptr, size_t outer_size, size_t inner_size,
                                             size_t output_dim0, R *y_ptr) {
  size_t num_blocks = (inner_size - 1) / blockDim.x + 1;
  for (size_t bid = blockIdx.x; bid < num_blocks; bid += gridDim.x) {
    size_t inner_idx = threadIdx.x + bid * blockDim.x;
    bool inner_valid = inner_idx < inner_size;
    for (size_t inid = blockIdx.y; inid < outer_size; inid += gridDim.y) {
      size_t beg_pos = indices_pos_ptr[inid];
      size_t end_pos = indices_pos_ptr[inid + 1];
      double sqrt_segment_len = sqrt(static_cast<double>(end_pos - beg_pos));
      for (size_t pos = beg_pos; pos < end_pos; pos += 1) {
        double reduce_result = 0;
        S index = inner_valid ? indices_ptr[pos] : outer_size;
        if (index >= 0 && index < outer_size) {
          reduce_result = static_cast<double>(grad_ptr[index * inner_size + inner_idx]);
        }
        if (threadIdx.y == 0 && inner_valid) {
          R *out_pos = y_ptr + segment_ids_ptr[pos] * inner_size + inner_idx;
          MsAtomicAdd(out_pos, static_cast<R>(reduce_result / sqrt_segment_len));
        }
      }
    }
  }
}

template <typename S>
__global__ void SparseSegmentSqrtNGradKernel(const half *grad_ptr, const S *indices_ptr, const S *segment_ids_ptr,
                                             const size_t *indices_pos_ptr, size_t outer_size, size_t inner_size,
                                             size_t output_dim0, half *y_ptr) {
  size_t num_blocks = (inner_size - 1) / blockDim.x + 1;
  for (size_t bid = blockIdx.x; bid < num_blocks; bid += gridDim.x) {
    size_t inner_idx = threadIdx.x + bid * blockDim.x;
    bool inner_valid = inner_idx < inner_size;
    for (size_t inid = blockIdx.y; inid < outer_size; inid += gridDim.y) {
      size_t beg_pos = indices_pos_ptr[inid];
      size_t end_pos = indices_pos_ptr[inid + 1];
      double sqrt_segment_len = sqrt(static_cast<double>(end_pos - beg_pos));
      for (size_t pos = beg_pos; pos < end_pos; pos += 1) {
        double reduce_result = 0;
        S index = inner_valid ? indices_ptr[pos] : outer_size;
        if (index >= 0 && index < outer_size) {
          reduce_result = static_cast<double>(__half2float(grad_ptr[index * inner_size + inner_idx]));
        }
        if (threadIdx.y == 0 && inner_valid) {
          half *out_pos = y_ptr + segment_ids_ptr[pos] * inner_size + inner_idx;
          MsAtomicAdd(out_pos, __float2half(static_cast<float>(reduce_result / sqrt_segment_len)));
        }
      }
    }
  }
}

inline int Log2Floor_M(uint32_t n) {
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

inline int Log2Floor64_M(uint64_t n) {
  // Scan n first high 32 then low 32 bits.
  const uint32_t high_32_bit = static_cast<uint32_t>(n >> 32);
  if (high_32_bit == 0) {
    return Log2Floor_M(static_cast<uint32_t>(n));
  } else {
    return 32 + Log2Floor_M(high_32_bit);
  }
}

inline int Log2Ceil64_M(uint64_t n) {
  int floor = Log2Floor64_M(n);
  if (n == (n & ~(n - 1)))
    return floor;
  else
    return floor + 1;
}


template <typename R, typename S>
bool CalSparseSegmentGradCombination(const std::string kernel_type, const R *grad_ptr, const S *indices_ptr,
                                     const S *segment_ids_ptr, size_t *indices_pos_ptr, size_t outer_size,
                                     size_t inner_size, size_t idx_seg_size, size_t output_dim0, R *y_ptr,
                                     uint32_t device_id, cudaStream_t cuda_stream) {
  // Get start position of each segment and set to indices_pos_ptr.
  // The last element of indices_pos_ptr must equal to idx_seg_size.
  SparseSegmentPosKernel<<<CUDA_BLOCKS(device_id, idx_seg_size + 1), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    indices_ptr, indices_pos_ptr, idx_seg_size, outer_size);
  const unsigned int max_grid_x = (1u << 31) - 1;
  const unsigned int max_grid_y = (1u << 16) - 1;
  unsigned int block_x = 32;
  unsigned int block_y = 1;
  unsigned int grid_x = std::min(static_cast<unsigned int>(UP_DIV(inner_size, block_x)), max_grid_x);
  unsigned int grid_y = std::min(static_cast<unsigned int>(output_dim0), max_grid_y);
  dim3 block(block_x, block_y);
  dim3 grid(grid_x, grid_y);
  unsigned int shared_memory_size = block_x * block_y * sizeof(R);
  if (kernel_type == "SparseSegmentSumGrad") {
    SparseSegmentSumGradKernel<<<grid, block, shared_memory_size, cuda_stream>>>(grad_ptr, indices_ptr,
                                                                                 segment_ids_ptr, indices_pos_ptr,
                                                                                 outer_size, inner_size, output_dim0,
                                                                                 y_ptr);
  } else if (kernel_type == "SparseSegmentSqrtNGrad") {
    SparseSegmentSqrtNGradKernel<<<grid, block, shared_memory_size, cuda_stream>>>(grad_ptr, indices_ptr,
                                                                                   segment_ids_ptr, indices_pos_ptr,
                                                                                   outer_size, inner_size,
                                                                                   output_dim0, y_ptr);
  }
  return true;
}

#define ADD_SPARSE_SEGMENT_GRAD(R, S) \
  template CUDA_LIB_EXPORT bool CalSparseSegmentGradCombination<R, S>(const std::string kernel_type, \
                                                                      const R *grad_ptr, const S *indices_ptr, \
                                                                      const S *segment_ids_ptr, \
                                                                      size_t *indices_pos_ptr, size_t outer_size, \
                                                                      size_t inner_size, size_t idx_seg_size, \
                                                                      size_t output_dim0, R *y_ptr, \
                                                                      uint32_t device_id, cudaStream_t cuda_stream);

ADD_SPARSE_SEGMENT_GRAD(half, int32_t)
ADD_SPARSE_SEGMENT_GRAD(half, int64_t)

ADD_SPARSE_SEGMENT_GRAD(float, int32_t)
ADD_SPARSE_SEGMENT_GRAD(float, int64_t)

ADD_SPARSE_SEGMENT_GRAD(double, int32_t)
ADD_SPARSE_SEGMENT_GRAD(double, int64_t)
