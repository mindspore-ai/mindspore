/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/transpose_impl.cuh"

template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T>
__global__ void TransposeKernel(const T *__restrict__ input, const size_t size, const TransposeInfoDevice info,
                                const int ndims, T *__restrict__ output) {
  const int32_t *in_strides = info.transpose_info_device;
  const int32_t *out_strides = info.transpose_info_device + stride_ndims;
  const int32_t *perm = info.transpose_info_device + stride_ndims * 2;
  for (int output_pos = blockDim.x * blockIdx.x + threadIdx.x; output_pos < size;
       output_pos += blockDim.x * gridDim.x) {
    int32_t input_pos = 0;
    int32_t temp = output_pos;
    for (int i = 0; i < ndims; ++i) {
      const int32_t ratio = temp / out_strides[i];
      temp -= ratio * out_strides[i];
      input_pos += ratio * in_strides[perm[i]];
    }
    output[output_pos] = input[input_pos];
  }
}

template <typename T>
bool TransposeUsingTile(const T *input, const std::vector<int64_t> &shape, const std::vector<int32_t> &perm, T *output,
                        cudaStream_t cuda_stream) {
  int dims = shape.size();
  if (dims < 2 || dims > 3) {
    return false;
  }
  switch (dims) {
    case 2:
      if (perm[0] == 1 && perm[1] == 0) {
        Swap3DTensorLast2Dim(input, (int64_t)1, shape[0], shape[1], output, cuda_stream);
        return true;
      }
      break;
    case 3:
      if (perm == std::vector<int32_t>{0, 2, 1}) {
        Swap3DTensorLast2Dim(input, shape[0], shape[1], shape[2], output, cuda_stream);
        return true;
      } else if (perm == std::vector<int32_t>{2, 1, 0}) {
        Swap3DTensorDim0and2(input, shape[0], shape[1], shape[2], output, cuda_stream);
        return true;
      } else {
        // Do not support other 3D Transpose.
        return false;
      }
      break;
    default:
      return false;
  }
  return false;
}

// Optimize nchw2nhwc && nhwc2nchw with tiling and shared memory.
// Firstly, combined 2 dims hw together, treat input and output as 3D tensor.
// Secondly, determine whether a matrix is a large matrix or a narrow matrix,
// which determines the chosen TileSize.
// Reason: tiling and shared memory can avoid uncoalesced global memory access.
// There are two stages of this kernel, load-to-shm and write-to-output.
// load-to-shm: Threads in a thread block work together to load input data tile to shared mem.
// write-to-output: Threads in a thread block work together to write shared mem to output tile.
// because of the shared mem usage, The access to both input and output memory can be coalesced.

// SimpleTransposeKernel for small matrix

__forceinline__ __device__ int TensorIdxToOneDimIdx(int ndims, const int *idx, const int *dims) {
  int flat_idx = idx[0];
  for (int i = 1; i < ndims; i++) {
    flat_idx = flat_idx * dims[i] + idx[i];
  }
  return flat_idx;
}

__forceinline__ __device__ void OneDimIdxToTensorIdx(int ndims, int idx, const int *dims, int *out_tensor_idx) {
  for (int i = ndims - 1; i >= 0; i--) {
    int new_idx = idx / dims[i];
    out_tensor_idx[i] = idx - dims[i] * new_idx;
    idx = new_idx;
  }
}

template <typename T, int perm0, int perm1, int perm2>
__global__ void Transpose3DTensorSimple(const T *__restrict__ input, const size_t size, const int64_t dim0,
                                        const int64_t dim1, const int64_t dim2, T *__restrict__ output) {
  int output_shape[3]{0, 0, 0};
  output_shape[perm0] = dim0;
  output_shape[perm1] = dim1;
  output_shape[perm2] = dim2;
  for (int output_pos = blockIdx.x * blockDim.x + threadIdx.x; output_pos < size;
       output_pos += gridDim.x * blockDim.x) {
    int output_tensor_index[3]{0, 0, 0};
    OneDimIdxToTensorIdx(3, output_pos, output_shape, output_tensor_index);
    int input_tensor_index[3]{0, 0, 0};
    int input_shape[3]{static_cast<int>(dim0), static_cast<int>(dim1), static_cast<int>(dim2)};
    input_tensor_index[0] = output_tensor_index[perm0];
    input_tensor_index[1] = output_tensor_index[perm1];
    input_tensor_index[2] = output_tensor_index[perm2];
    int input_pos = TensorIdxToOneDimIdx(3, input_tensor_index, input_shape);
    output[output_pos] = input[input_pos];
  }
}

template <typename T>
__global__ void Swap3DTensorLast2DimKernel(const T *input, int NumThreads, int TileHeight, int TileWidth,
                                           int input_dims_0, int input_dims_1, int input_dims_2, T *output) {
  extern __shared__ unsigned char sdata_uchar[];
  // shm_tile[TileHeight][TileWidth + 1]: to avoid bank conflict in write-to-output period
  T *shm_tile = reinterpret_cast<T *>(sdata_uchar);
  int NumRowsPerLoadLoop = NumThreads / TileWidth;  // the number of shm rows that all threads can load into shm once
  int NumColsPerWriteLoop =
    NumThreads / TileHeight;  // the number of shm cols that all threads can write into output once
  int load_thread_num_align = NumRowsPerLoadLoop * TileWidth;     // use align num threads in load-to-shm period
  int write_thread_num_align = NumColsPerWriteLoop * TileHeight;  // use align num threads in write-to-output period
  int tid = threadIdx.x;
  int input_dims[3] = {input_dims_0, input_dims_1, input_dims_2};
  int output_dims[3] = {input_dims[0], input_dims[2], input_dims[1]};
  int input_dims_in_tiles[3] = {input_dims[0], (input_dims[1] + TileHeight - 1) / TileHeight,
                                (input_dims[2] + TileWidth - 1) / TileWidth};
  int input_tile_idx[3];
  OneDimIdxToTensorIdx(3, blockIdx.x, input_dims_in_tiles, input_tile_idx);
  int input_tile_origin[3] = {input_tile_idx[0], input_tile_idx[1] * TileHeight, input_tile_idx[2] * TileWidth};
  int input_block_start_idx = TensorIdxToOneDimIdx(3, input_tile_origin, input_dims);  // input idx of this thread block
  bool full_tile = true;
  int tile_width = TileWidth;
  // Only the last row or column may not have the full size
  // boundary process
  if (input_tile_idx[2] == input_dims_in_tiles[2] - 1) {
    tile_width = input_dims[2] - (input_dims_in_tiles[2] - 1) * TileWidth;
    full_tile &= false;
  }
  int tile_height = TileHeight;
  if (input_tile_idx[1] == input_dims_in_tiles[1] - 1) {
    tile_height = input_dims[1] - (input_dims_in_tiles[1] - 1) * TileHeight;
    full_tile &= false;
  }
  // load-to-shm: each block load input data into shared mem(loop)
  if (tid < load_thread_num_align) {
    // Map task blocks to thread blocks.
    // organize threads to n*TileWidth
    int shm_row_id = tid / TileWidth;  // shem_row_id, also the block row_id of input
    int shm_col_id = tid % TileWidth;  // shem_col_id, also the block col_id of input
    int input_idx = input_block_start_idx + shm_row_id * input_dims[2] + shm_col_id;  // the input idx of this thread
    int input_step = NumRowsPerLoadLoop * input_dims[2];
    if (full_tile) {  // thread blocks responses for inner tiles
#pragma unroll
      for (int row_id = shm_row_id; row_id < (TileHeight);
           row_id += NumRowsPerLoadLoop) {  // move to the next pass, loop
        // shm_tile[row_id][shm_col_id]
        shm_tile[row_id * (TileWidth + 1) + shm_col_id] =
          input[input_idx];       // each thread load one input data into shared mem
        input_idx += input_step;  // calculate the next input idx this thread should load
      }
    } else {                      // boundary process: thread blocks responses for edge tiles
      if (shm_col_id < tile_width) {
        for (int row_id = shm_row_id; row_id < (tile_height); row_id += NumRowsPerLoadLoop) {
          // shm_tile[row_id][shm_col_id]
          shm_tile[row_id * (TileWidth + 1) + shm_col_id] = input[input_idx];
          input_idx += input_step;
        }
      }
    }
  }
  __syncthreads();
  // load-to-shm: end

  // write-to-output: each block write shared mem into output(loop)
  int output_tile_idx[3] = {input_tile_idx[0], input_tile_idx[2], input_tile_idx[1]};
  int output_tile_origin[3] = {output_tile_idx[0], output_tile_idx[1] * TileWidth, output_tile_idx[2] * TileHeight};
  int output_block_start_idx = TensorIdxToOneDimIdx(3, output_tile_origin, output_dims);
  if (tid < write_thread_num_align) {
    // organize threads to TileHeight*n1
    int shm_col_id = tid / TileHeight;  // shm_col_id, also the block row_id of output
    int shm_row_id = tid % TileHeight;  // shm_row_id, also the block col_id of output
    int output_idx = output_block_start_idx + shm_col_id * output_dims[2] + shm_row_id;
    int output_step = NumColsPerWriteLoop * output_dims[2];
    if (full_tile) {
#pragma unroll
      for (int col_id = shm_col_id; col_id < (TileWidth);
           col_id += NumColsPerWriteLoop) {  // move to the next pass, loop
        // shm_tile[shm_row_id][col_id]
        output[output_idx] = shm_tile[shm_row_id * (TileWidth + 1) + col_id];  // avoid bank conflict
        output_idx += output_step;
      }
    } else {
      if (shm_row_id < tile_height) {
        for (int col_id = shm_col_id; col_id < (tile_width); col_id += NumColsPerWriteLoop) {
          // shm_tile[shm_row_id][col_id];
          output[output_idx] = shm_tile[shm_row_id * (TileWidth + 1) + col_id];
          output_idx += output_step;
        }
      }
    }
  }
}

template <typename T, int perm0, int perm1, int perm2>
__global__ void Transpose3DTensorSimpleVector(const T *__restrict__ input, size_t size, const int64_t dim0,
                                              const int64_t dim1, const int64_t dim2, T *__restrict__ output) {
  int output_shape[3]{0, 0, 0};
  output_shape[perm0] = dim0;
  output_shape[perm1] = dim1;
  output_shape[perm2] = dim2;

  const int stride = blockDim.x * gridDim.x * kUnroll;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  T vec[kUnroll];
  int output_pos;
  for (output_pos = tid * kUnroll; output_pos + kUnroll - 1 < size; output_pos += stride) {
#pragma unroll
    for (int i = 0; i < kUnroll; ++i) {
      int outpos_pos_i = output_pos + i;
      int output_tensor_index[3]{0, 0, 0};
      OneDimIdxToTensorIdx(3, outpos_pos_i, output_shape, output_tensor_index);
      int input_tensor_index[3]{0, 0, 0};
      int input_shape[3]{static_cast<int>(dim0), static_cast<int>(dim1), static_cast<int>(dim2)};
      input_tensor_index[0] = output_tensor_index[perm0];
      input_tensor_index[1] = output_tensor_index[perm1];
      input_tensor_index[2] = output_tensor_index[perm2];
      int input_pos_i = TensorIdxToOneDimIdx(3, input_tensor_index, input_shape);
      vec[i] = input[input_pos_i];
    }
    float2 *out = reinterpret_cast<float2 *>(output + output_pos);
    *out = *reinterpret_cast<float2 *>(vec);
  }

  for (; output_pos < size; ++output_pos) {
    int output_tensor_index[3]{0, 0, 0};
    OneDimIdxToTensorIdx(3, output_pos, output_shape, output_tensor_index);
    int input_tensor_index[3]{0, 0, 0};
    int input_shape[3]{static_cast<int>(dim0), static_cast<int>(dim1), static_cast<int>(dim2)};
    input_tensor_index[0] = output_tensor_index[perm0];
    input_tensor_index[1] = output_tensor_index[perm1];
    input_tensor_index[2] = output_tensor_index[perm2];
    int input_pos = TensorIdxToOneDimIdx(3, input_tensor_index, input_shape);
    output[output_pos] = input[input_pos];
  }
}

template <typename T>
void Swap3DTensorLast2Dim(const T *input, const int64_t dim0, const int64_t dim1, const int64_t dim2, T *output,
                          cudaStream_t cuda_stream) {
  static const int kMinDimensionToUseTiles = 16;
  static const int kMinDimensionToUseRectTiles = 96;
  auto short_side = std::min(dim1, dim2);
  auto long_side = std::max(dim1, dim2);
  // large matrix
  // Both dims are greater than 16 && cuda blocks have enough shared mem.
  constexpr int kTileSizeLargeMat = 32;
  constexpr int kNumThreadsLargeMat = 256;
  auto ShmemReqLargeMat = kTileSizeLargeMat * (kTileSizeLargeMat + 1) * sizeof(T);
  bool is_large_matrix = short_side >= kMinDimensionToUseTiles && ShmemReqLargeMat <= SHARED_MEM_PER_BLOCK;
  // narrow matrix
  // one dim less than 16 && one dim greater than 96(narrow)
  constexpr int kTileSizeNarrowMatLongSide = 128;
  const int kTileSizeNarrowMatShortSide = short_side;
  constexpr int kNumThreadsNarrowMat = kTileSizeNarrowMatLongSide;
  auto ShmemReqNarrowMat = kTileSizeNarrowMatLongSide * (kTileSizeNarrowMatShortSide + 1) * sizeof(T);
  bool is_narrow_matrix = short_side < kMinDimensionToUseTiles && long_side >= kMinDimensionToUseRectTiles &&
                          ShmemReqNarrowMat <= SHARED_MEM_PER_BLOCK;
  if (is_large_matrix) {
    int64_t input_dims_in_tiles[3]{dim0, (dim1 + kTileSizeLargeMat - 1) / kTileSizeLargeMat,
                                   (dim2 + kTileSizeLargeMat - 1) / kTileSizeLargeMat};
    int TotalNumTiles = input_dims_in_tiles[0] * input_dims_in_tiles[1] * input_dims_in_tiles[2];
    Swap3DTensorLast2DimKernel<T><<<TotalNumTiles, kNumThreadsLargeMat, ShmemReqLargeMat, cuda_stream>>>(
      input, kNumThreadsLargeMat, kTileSizeLargeMat, kTileSizeLargeMat, dim0, dim1, dim2, output);
  } else if (is_narrow_matrix) {
    int64_t input_dims_in_tiles[3]{dim0, 1, (long_side + kTileSizeNarrowMatLongSide - 1) / kTileSizeNarrowMatLongSide};
    int TotalNumTiles = input_dims_in_tiles[0] * input_dims_in_tiles[1] * input_dims_in_tiles[2];
    int TileHeight, TileWidth;
    if (long_side == dim1) {
      TileHeight = kTileSizeNarrowMatLongSide;
      TileWidth = short_side;
    } else {
      TileHeight = short_side;
      TileWidth = kTileSizeNarrowMatLongSide;
    }
    Swap3DTensorLast2DimKernel<T><<<TotalNumTiles, kNumThreadsNarrowMat, ShmemReqNarrowMat, cuda_stream>>>(
      input, kNumThreadsNarrowMat, TileHeight, TileWidth, dim0, dim1, dim2, output);
  } else {
    size_t size = static_cast<size_t>(dim0 * dim1 * dim2);
    Transpose3DTensorSimple<T, 0, 2, 1>
      <<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(input, size, dim0, dim1, dim2, output);
  }
  return;
}

template <typename T>
void Swap3DTensorDim0and2(const T *input, const int64_t dim0, const int64_t dim1, const int64_t dim2, T *output,
                          cudaStream_t cuda_stream) {
  size_t size = dim0 * dim1 * dim2;
  auto out_ptr = reinterpret_cast<uintptr_t>(output);
  bool aligned = (out_ptr % 16 == 0);  // Is aligned with 16 bits(2 bytes)?
  bool use_vector{false}, is_custom{false};
  if ((dim0 <= 128 && dim2 <= 128) || dim0 * dim1 <= 128 || dim1 * dim2 <= 8) {
    use_vector = is_custom = true;
  } else if (dim1 * dim2 <= 16384) {
    use_vector = true;
  }
  if (sizeof(T) == 2 && aligned && use_vector) {
    int grid_size;
    if (is_custom) {
      grid_size = (size + GET_THREADS - 1) / GET_THREADS;
    } else {
      grid_size = GET_BLOCKS(size);
    }
    Transpose3DTensorSimpleVector<T, 2, 1, 0>
      <<<grid_size, GET_THREADS / kUnroll, 0, cuda_stream>>>(input, size, dim0, dim1, dim2, output);
  } else {
    Transpose3DTensorSimple<T, 2, 1, 0>
      <<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(input, size, dim0, dim1, dim2, output);
  }

  return;
}

template <typename T, bool need_simplify>
cudaError_t CalTranspose(const size_t size, const T *input, const TransposeInfo &info, T *output,
                         cudaStream_t cuda_stream) {
  std::vector<int64_t> new_shape{0};
  std::vector<int32_t> new_perm{0};

  if (need_simplify) {
    SimplifyTranspose(info.input_shape, info.perm, &new_shape, &new_perm);
  } else {
    new_shape = info.input_shape;
    new_perm = info.perm;
  }

  if (TransposeUsingTile(input, new_shape, new_perm, output, cuda_stream)) {
    return GetCudaStatus();
  }

  TransposeInfoDevice transpose_info_device;
  int32_t input_stride[kDimSize];
  int32_t output_stride[kDimSize];
  ComputeInputStride(new_shape, input_stride);
  ComputeOutputStride(new_shape, new_perm, output_stride);

  for (size_t i = 0; i < new_shape.size(); ++i) {
    transpose_info_device.transpose_info_device[i] = input_stride[i];
    transpose_info_device.transpose_info_device[i + stride_ndims] = output_stride[i];
    transpose_info_device.transpose_info_device[i + stride_ndims * 2] = new_perm[i];
  }
  TransposeKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(input, size, transpose_info_device,
                                                                     new_shape.size(), output);
  return GetCudaStatus();
}

#define REGISTER_CALTRANSPOSE(T, NEED_SIMPLIFY)                        \
  template CUDA_LIB_EXPORT cudaError_t CalTranspose<T, NEED_SIMPLIFY>( \
    const size_t size, const T *input, const TransposeInfo &info, T *output, cudaStream_t cuda_stream)

#define REGISTER_BOTH_CALTRANSPOSE(T) \
  REGISTER_CALTRANSPOSE(T, true);     \
  REGISTER_CALTRANSPOSE(T, false)

REGISTER_BOTH_CALTRANSPOSE(bool);
REGISTER_BOTH_CALTRANSPOSE(int8_t);
REGISTER_BOTH_CALTRANSPOSE(int16_t);
REGISTER_BOTH_CALTRANSPOSE(int32_t);
REGISTER_BOTH_CALTRANSPOSE(int64_t);
REGISTER_BOTH_CALTRANSPOSE(uint8_t);
REGISTER_BOTH_CALTRANSPOSE(uint16_t);
REGISTER_BOTH_CALTRANSPOSE(uint32_t);
REGISTER_BOTH_CALTRANSPOSE(uint64_t);
REGISTER_BOTH_CALTRANSPOSE(half);
REGISTER_BOTH_CALTRANSPOSE(float);
REGISTER_BOTH_CALTRANSPOSE(double);
REGISTER_BOTH_CALTRANSPOSE(Complex<double>);
REGISTER_BOTH_CALTRANSPOSE(Complex<float>);
