/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include <complex>
#include <cstdio>
#include <cstdint>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/bias_add_grad_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "include/cuda_fp16.h"
#include "include/cuda_runtime.h"

const int kWarpSize = 32;
// tuning param, for those nhw >= kLargeSize, launch more blocks to solve
const int kLargeSize = 500000;  // tuning param for BiasAddGradNHWC
const int kNumBlocks = 8;       // tuning param for BiasAddGradNHWC

// For NHWC bias add grad, combine dy's NHW together, matrix column reduce.
// This is a simple implementation, can be further optimized  when C is small.
// Firstly, Each warp sums several rows, each thread's partial_sum is the sum of
// a part of one cloumn.
// Secondly, in order to sum up all values in one column, which is to sum up the partial_sum
// in different warps but with the same lane_id, each warp store their partial_sums
// to one row of shared mem, and read partial_sums from one col of shared mem.
// Then each warp do warp reduce to sum up 32 partial_sums, and write final result to db
// For larger NHW, one block is not enough to sum up all rows, needs to launch more blocks.
template <typename T>
__global__ void BiasAddGradNHWC(const T *dy, T *db, const size_t m, const size_t n, const size_t rows_per_block,
                                size_t rows_per_warp) {
  __shared__ T shared_d[kWarpSize][kWarpSize + 1];  // avoid bank conflict
  int shm_row_id = (threadIdx.x >> 5);
  int shm_col_id = (threadIdx.x % 32);
  int block_start_row = blockIdx.x * rows_per_block;
  int block_end_row = block_start_row + rows_per_block;
  block_end_row = block_end_row < m ? block_end_row : m;
  int warp_start_row = blockIdx.x * rows_per_block + shm_row_id * rows_per_warp;
  int warp_end_row = warp_start_row + rows_per_warp;
  int real_rows_per_warp = warp_end_row < block_end_row ? rows_per_warp : block_end_row - warp_start_row;
  // boundary process
  // Only the last row or column may not have the full size
  bool full_tile = true;
  int tile_width_real = 32;
  if (blockIdx.y == blockDim.y - 1) {
    tile_width_real = n - (blockDim.y - 1) * 32;
    full_tile = (tile_width_real == 32);
  }
  int read_offset = warp_start_row * n + (blockIdx.y << 5) + shm_col_id;
  T partial_sum = ZeroImpl<T>();
  if (full_tile) {
    for (int i = 0; i < real_rows_per_warp; i++) {
      partial_sum += dy[read_offset];
      read_offset += n;
    }
  } else {
    if (shm_col_id < tile_width_real) {
      for (int i = 0; i < real_rows_per_warp; i++) {
        partial_sum += dy[read_offset];
        read_offset += n;
      }
    }
  }
  shared_d[shm_row_id][shm_col_id] = partial_sum;
  __syncthreads();
  partial_sum = shared_d[shm_col_id][shm_row_id];
  __syncthreads();
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    partial_sum += shfl_down_sync(0xffffffff, partial_sum, offset);
  }
  if (shm_col_id == 0) {
    if (full_tile) {
      MsAtomicAdd(db + (blockIdx.y << 5) + shm_row_id, partial_sum);
    } else {
      if (shm_row_id < tile_width_real) {
        MsAtomicAdd(db + (blockIdx.y << 5) + shm_row_id, partial_sum);
      }
    }
  }
}

template <typename T>
__global__ void BiasAddGradNCHW(const size_t size, const int batch, const int bias_size, const int h, const int w,
                                const int bg_size, const T *dy, T *db) {
  __shared__ T shared_d[32];
  for (int i = threadIdx.x; i < 32; i += blockDim.x) {
    shared_d[i] = ZeroImpl<T>();
  }
  __syncthreads();
  T sum = ZeroImpl<T>();
  int lane_id = threadIdx.x % 32;
  int thread_id = threadIdx.x;
  int img_size = h * w;
  // N*H*W -> count / bg_size  equals the amount of work one block should reduce
  int count = batch * img_size;
  int bg_offset = blockIdx.x % bias_size;
  int bg_id = blockIdx.x / bias_size;
  for (int i = bg_id * blockDim.x + threadIdx.x;  // thread start
       i < count; i += blockDim.x * bg_size) {
    int img_offset = i % img_size;
    int img_id = i / img_size;
    sum += *(dy + (img_id * bias_size + bg_offset) * img_size + img_offset);
  }
  MsAtomicAdd(shared_d + lane_id, sum);
  __syncthreads();
  if (thread_id < 32) {
    T data = shared_d[thread_id];
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
      data += shfl_xor_sync(0xffffffff, data, offset);
    }
    if (thread_id == 0) {
      MsAtomicAdd(db + bg_offset, data);
    }
  }
}

template <typename T>
__global__ void FillDb(T *db, const size_t bias_size) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < bias_size; pos += blockDim.x * gridDim.x) {
    db[pos] = ZeroImpl<T>();
  }
}

template <typename T>
void CalBiasAddGradNCHW(const size_t size, const size_t bias_size, const int height, const int width, const T *dy,
                        T *db, cudaStream_t cuda_stream) {
  int batch_size = size / bias_size / height / width;
  int block_num = GET_BLOCKS(size);
  int thread_num = GET_THREADS;
  // how many blocks to solve one bias's reduce work(N * H * W)
  int block_group_size = (block_num + bias_size - 1) / bias_size;
  block_num = block_group_size * bias_size;
  if (thread_num < kWarpSize) {
    thread_num = kWarpSize;
  }
  FillDb<<<GET_BLOCKS(bias_size), GET_THREADS, 0, cuda_stream>>>(db, bias_size);
  BiasAddGradNCHW<<<block_num, thread_num, 0, cuda_stream>>>(size, batch_size, bias_size, height, width,
                                                             block_group_size, dy, db);
  return;
}

template <typename T>
void CalBiasAddGradNHWC(const size_t size, const size_t bias_size, const T *dy, T *db, cudaStream_t cuda_stream) {
  FillDb<<<GET_BLOCKS(bias_size), GET_THREADS, 0, cuda_stream>>>(db, bias_size);
  size_t rows = size / bias_size;
  int block_num_x = rows <= kLargeSize ? 1 : kNumBlocks;
  int block_num_y = (bias_size + kWarpSize - 1) / kWarpSize;
  dim3 grid_size(block_num_x, block_num_y, 1);
  dim3 block_size(kWarpSize * kWarpSize);
  size_t rows_per_block = (rows + block_num_x - 1) / block_num_x;
  size_t rows_per_warp = (rows_per_block + kWarpSize - 1) / kWarpSize;
  BiasAddGradNHWC<<<grid_size, block_size, 0, cuda_stream>>>(dy, db, rows, bias_size, rows_per_block, rows_per_warp);
  return;
}

template CUDA_LIB_EXPORT void CalBiasAddGradNCHW<half>(const size_t size, const size_t bias_size, const int height,
                                                       const int width, const half *dy, half *db,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalBiasAddGradNCHW<float>(const size_t size, const size_t bias_size, const int height,
                                                        const int width, const float *dy, float *db,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalBiasAddGradNCHW<double>(const size_t size, const size_t bias_size, const int height,
                                                         const int width, const double *dy, double *db,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalBiasAddGradNCHW<int8_t>(const size_t size, const size_t bias_size, const int height,
                                                         const int width, const int8_t *dy, int8_t *db,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalBiasAddGradNCHW<int16_t>(const size_t size, const size_t bias_size, const int height,
                                                          const int width, const int16_t *dy, int16_t *db,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalBiasAddGradNCHW<int>(const size_t size, const size_t bias_size, const int height,
                                                      const int width, const int *dy, int *db,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalBiasAddGradNCHW<int64_t>(const size_t size, const size_t bias_size, const int height,
                                                          const int width, const int64_t *dy, int64_t *db,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalBiasAddGradNCHW<uint8_t>(const size_t size, const size_t bias_size, const int height,
                                                          const int width, const uint8_t *dy, uint8_t *db,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalBiasAddGradNCHW<uint16_t>(const size_t size, const size_t bias_size, const int height,
                                                           const int width, const uint16_t *dy, uint16_t *db,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalBiasAddGradNCHW<uint32_t>(const size_t size, const size_t bias_size, const int height,
                                                           const int width, const uint32_t *dy, uint32_t *db,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalBiasAddGradNCHW<uint64_t>(const size_t size, const size_t bias_size, const int height,
                                                           const int width, const uint64_t *dy, uint64_t *db,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalBiasAddGradNCHW<Complex<float>>(const size_t size, const size_t bias_size,
                                                                 const int height, const int width,
                                                                 const Complex<float> *dy, Complex<float> *db,
                                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalBiasAddGradNCHW<Complex<double>>(const size_t size, const size_t bias_size,
                                                                  const int height, const int width,
                                                                  const Complex<double> *dy, Complex<double> *db,
                                                                  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalBiasAddGradNHWC<half>(const size_t size, const size_t bias_size, const half *dy,
                                                       half *db, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalBiasAddGradNHWC<float>(const size_t size, const size_t bias_size, const float *dy,
                                                        float *db, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalBiasAddGradNHWC<double>(const size_t size, const size_t bias_size, const double *dy,
                                                         double *db, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalBiasAddGradNHWC<int8_t>(const size_t size, const size_t bias_size, const int8_t *dy,
                                                         int8_t *db, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalBiasAddGradNHWC<int16_t>(const size_t size, const size_t bias_size, const int16_t *dy,
                                                          int16_t *db, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalBiasAddGradNHWC<int>(const size_t size, const size_t bias_size, const int *dy, int *db,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalBiasAddGradNHWC<int64_t>(const size_t size, const size_t bias_size, const int64_t *dy,
                                                          int64_t *db, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalBiasAddGradNHWC<uint8_t>(const size_t size, const size_t bias_size, const uint8_t *dy,
                                                          uint8_t *db, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalBiasAddGradNHWC<uint16_t>(const size_t size, const size_t bias_size,
                                                           const uint16_t *dy, uint16_t *db, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalBiasAddGradNHWC<uint32_t>(const size_t size, const size_t bias_size,
                                                           const uint32_t *dy, uint32_t *db, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalBiasAddGradNHWC<uint64_t>(const size_t size, const size_t bias_size,
                                                           const uint64_t *dy, uint64_t *db, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalBiasAddGradNHWC<Complex<float>>(const size_t size, const size_t bias_size,
                                                                 const Complex<float> *dy, Complex<float> *db,
                                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalBiasAddGradNHWC<Complex<double>>(const size_t size, const size_t bias_size,
                                                                  const Complex<double> *dy, Complex<double> *db,
                                                                  cudaStream_t cuda_stream);
