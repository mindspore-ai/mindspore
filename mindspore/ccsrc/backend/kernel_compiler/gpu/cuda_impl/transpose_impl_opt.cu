/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <cstdint>
#include <vector>
#include <limits>
#include <utility>
#include <algorithm>
#include "transpose_impl_opt.cuh"
#include "runtime/device/gpu/cuda_common.h"

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
template <typename T>
__global__ void SimpleTransposeKernel(const size_t size, const T *input, const size_t *input_shape,
                                      const size_t *input_axis, const size_t shape_size, T *output) {
  size_t pos_size;
  size_t temp_pos;
  size_t newpos;
  size_t newpos_size;
  size_t pos_array[4];
  // for example 4-D: pos = posArray[0] * input_shape[1] * input_shape[2] * input_shape[3] +
  //                        posArray[1] * input_shape[2] * input_shape[3] +
  //                        posArray[2] * input_shape[3] +
  //                        posArray[3]
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    temp_pos = pos;
    pos_size = size / input_shape[0];    // C * H * W
    pos_array[0] = temp_pos / pos_size;  // i / (CHW)
    for (size_t i = 1; i < shape_size; i++) {
      temp_pos -= pos_array[i - 1] * pos_size;
      pos_size = pos_size / input_shape[i];
      pos_array[i] = temp_pos / pos_size;
    }
    newpos = pos_array[input_axis[shape_size - 1]];
    newpos_size = 1;
    for (int64_t j = shape_size - 2; j >= 0; j--) {
      newpos_size *= input_shape[input_axis[j + 1]];
      newpos += pos_array[input_axis[j]] * newpos_size;
    }
    output[newpos] = *(input + pos);
  }
  return;
}

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

template <typename T>
__global__ void Swap3DTensorLast2DimKernel_shared(const T *input, int NumThreads, int TileHeight, int TileWidth,
                                                  int input_dims_0, int input_dims_1, int input_dims_2, T *output) {
  extern __shared__ unsigned char sdata_uchar[];
  // shm_tile[TileHeight][TileWidth + 1]: to avoid bank conflict in write-to-output period
  T *shm_tile = reinterpret_cast<T*>(sdata_uchar);
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
    } else {  // boundary process: thread blocks responses for edge tiles
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

template <typename T>
void Swap3DTensorLast2Dim(const size_t size, const size_t shape_size, int *combined_dims, const T *d_input,
                          const size_t *input_shape, const size_t *input_axis, const size_t *d_input_shape,
                          const size_t *d_input_axis, T *d_output, cudaStream_t cuda_stream) {
  static const int kMinDimensionToUseTiles = 16;
  static const int kMinDimensionToUseRectTiles = 96;
  auto short_side = std::min(combined_dims[1], combined_dims[2]);
  auto long_side = std::max(combined_dims[1], combined_dims[2]);
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
    int input_dims_in_tiles[3] = {combined_dims[0], (combined_dims[1] + kTileSizeLargeMat - 1) / kTileSizeLargeMat,
                                  (combined_dims[2] + kTileSizeLargeMat - 1) / kTileSizeLargeMat};
    int TotalNumTiles = input_dims_in_tiles[0] * input_dims_in_tiles[1] * input_dims_in_tiles[2];
    Swap3DTensorLast2DimKernel_shared<T><<<TotalNumTiles, kNumThreadsLargeMat, ShmemReqLargeMat, cuda_stream>>>(
      d_input, kNumThreadsLargeMat, kTileSizeLargeMat, kTileSizeLargeMat, combined_dims[0], combined_dims[1],
      combined_dims[2], d_output);
  } else if (is_narrow_matrix) {
    int input_dims_in_tiles[3] = {combined_dims[0], 1,
                                  (long_side + kTileSizeNarrowMatLongSide - 1) / kTileSizeNarrowMatLongSide};
    int TotalNumTiles = input_dims_in_tiles[0] * input_dims_in_tiles[1] * input_dims_in_tiles[2];
    int TileHeight, TileWidth;
    if (long_side == combined_dims[1]) {
      TileHeight = kTileSizeNarrowMatLongSide;
      TileWidth = short_side;
    } else {
      TileHeight = short_side;
      TileWidth = kTileSizeNarrowMatLongSide;
    }
    Swap3DTensorLast2DimKernel_shared<T><<<TotalNumTiles, kNumThreadsNarrowMat, ShmemReqNarrowMat, cuda_stream>>>(
      d_input, kNumThreadsNarrowMat, TileHeight, TileWidth, combined_dims[0], combined_dims[1], combined_dims[2],
      d_output);
  } else {
    SimpleTransposeKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, d_input, d_input_shape, d_input_axis,
                                                                             shape_size, d_output);
  }
  return;
}
// specific for NHWC -> NCHW
template <typename T>
void CalNHWC2NCHWInterface(const size_t size, const size_t shape_size, const T *d_input, const size_t *input_shape,
                           const size_t *input_axis, const size_t *d_input_shape, const size_t *d_input_axis,
                           T *d_output, cudaStream_t cuda_stream) {
  int combined_dims[3];
  combined_dims[0] = input_shape[0];  // N
  combined_dims[1] = input_shape[1];  // HW
  for (unsigned int i = 2; i < shape_size - 1; i++) {
    combined_dims[1] *= input_shape[i];
  }
  combined_dims[2] = input_shape[shape_size - 1];  // C
  Swap3DTensorLast2Dim(size, shape_size, combined_dims, d_input, input_shape, input_axis, d_input_shape, d_input_axis,
                       d_output, cuda_stream);
}
// specific for NCHW -> NHWC
template <typename T>
void CalNCHW2NHWCInterface(const size_t size, const size_t shape_size, const T *d_input, const size_t *input_shape,
                           const size_t *input_axis, const size_t *d_input_shape, const size_t *d_input_axis,
                           T *d_output, cudaStream_t cuda_stream) {
  int combined_dims[3];
  combined_dims[0] = input_shape[0];  // N
  combined_dims[1] = input_shape[1];  // C
  combined_dims[2] = input_shape[2];  // HW
  for (unsigned int i = 3; i < shape_size; ++i) {
    combined_dims[2] *= input_shape[i];
  }
  Swap3DTensorLast2Dim(size, shape_size, combined_dims, d_input, input_shape, input_axis, d_input_shape, d_input_axis,
                       d_output, cuda_stream);
}

template void CalNHWC2NCHWInterface<double>(const size_t size, const size_t shape_size, const double *d_input,
                                           const size_t *input_shape, const size_t *input_axis,
                                           const size_t *d_input_shape, const size_t *d_input_axis, double *d_output,
                                           cudaStream_t cuda_stream);
template void CalNHWC2NCHWInterface<float>(const size_t size, const size_t shape_size, const float *d_input,
                                           const size_t *input_shape, const size_t *input_axis,
                                           const size_t *d_input_shape, const size_t *d_input_axis, float *d_output,
                                           cudaStream_t cuda_stream);
template void CalNHWC2NCHWInterface<half>(const size_t size, const size_t shape_size, const half *d_input,
                                          const size_t *input_shape, const size_t *input_axis,
                                          const size_t *d_input_shape, const size_t *d_input_axis, half *d_output,
                                          cudaStream_t cuda_stream);
template void CalNHWC2NCHWInterface<int>(const size_t size, const size_t shape_size, const int *d_input,
                                         const size_t *input_shape, const size_t *input_axis,
                                         const size_t *d_input_shape, const size_t *d_input_axis, int *d_output,
                                         cudaStream_t cuda_stream);
template void CalNHWC2NCHWInterface<int64_t>(const size_t size, const size_t shape_size, const int64_t *d_input,
                                             const size_t *input_shape, const size_t *input_axis,
                                             const size_t *d_input_shape, const size_t *d_input_axis, int64_t *d_output,
                                             cudaStream_t cuda_stream);

template void CalNCHW2NHWCInterface<double>(const size_t size, const size_t shape_size, const double *d_input,
                                           const size_t *input_shape, const size_t *input_axis,
                                           const size_t *d_input_shape, const size_t *d_input_axis, double *d_output,
                                           cudaStream_t cuda_stream);
template void CalNCHW2NHWCInterface<float>(const size_t size, const size_t shape_size, const float *d_input,
                                           const size_t *input_shape, const size_t *input_axis,
                                           const size_t *d_input_shape, const size_t *d_input_axis, float *d_output,
                                           cudaStream_t cuda_stream);
template void CalNCHW2NHWCInterface<half>(const size_t size, const size_t shape_size, const half *d_input,
                                          const size_t *input_shape, const size_t *input_axis,
                                          const size_t *d_input_shape, const size_t *d_input_axis, half *d_output,
                                          cudaStream_t cuda_stream);
template void CalNCHW2NHWCInterface<int>(const size_t size, const size_t shape_size, const int *d_input,
                                         const size_t *input_shape, const size_t *input_axis,
                                         const size_t *d_input_shape, const size_t *d_input_axis, int *d_output,
                                         cudaStream_t cuda_stream);
template void CalNCHW2NHWCInterface<int64_t>(const size_t size, const size_t shape_size, const int64_t *d_input,
                                             const size_t *input_shape, const size_t *input_axis,
                                             const size_t *d_input_shape, const size_t *d_input_axis, int64_t *d_output,
                                             cudaStream_t cuda_stream);
