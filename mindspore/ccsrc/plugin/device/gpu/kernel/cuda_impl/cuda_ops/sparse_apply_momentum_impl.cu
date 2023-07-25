/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <cub/cub.cuh>
#include <vector>
#include <iostream>
#include <typeinfo>
#include <algorithm>
#include "sparse_apply_momentum_impl.cuh"
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename S>
__global__ void SumOfRows(S *indices_sort, size_t indices_num, int32_t *thready_pos) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < indices_num; idx += blockDim.x * gridDim.x) {
    if (idx == 0 || idx == indices_num - 1 || indices_sort[idx] != indices_sort[idx - 1]) {
      thready_pos[idx] = static_cast<S>(idx);
    } else {
      thready_pos[idx] = static_cast<int32_t>(-1);
    }
  }
}

template <typename T, typename S>
__global__ void SparseApplyMomentumKernel(const size_t inner_size, T *var, T *accum, const T *lr, const T *grad,
                                          int32_t *rows_index, S *indices_sort, int32_t *thready_pos_shrink,
                                          int32_t shrink_num, const T *momentum) {
  for (size_t pos_x = blockIdx.x * blockDim.x + threadIdx.x; pos_x < inner_size; pos_x += gridDim.x * blockDim.x) {
    for (size_t pos_y = blockIdx.y * blockDim.y + threadIdx.y; pos_y < shrink_num - 1;
         pos_y += gridDim.y * blockDim.y) {
      int32_t start_row = thready_pos_shrink[pos_y];
      int32_t end_row = thready_pos_shrink[pos_y + 1];
      S update_pos = indices_sort[start_row] * inner_size + pos_x;
      for (int idx = start_row; idx < end_row; ++idx) {
        int grad_pos = rows_index[idx] * inner_size + pos_x;
        accum[update_pos] = static_cast<T>(*momentum) * accum[update_pos] + grad[grad_pos];
        var[update_pos] = var[update_pos] - static_cast<T>(*lr) * grad[grad_pos] +
                          static_cast<T>(*lr) * static_cast<T>(*momentum) * accum[update_pos];
      }
    }
  }
}

template <typename T, typename S>
__global__ void SparseApplyMomentumKernel_(const size_t inner_size, T *var, T *accum, const T *lr, const T *grad,
                                           int32_t *rows_index, S *indices_sort, int32_t *thready_pos_shrink,
                                           int32_t shrink_num, const T *momentum) {
  for (size_t pos_x = blockIdx.x * blockDim.x + threadIdx.x; pos_x < inner_size; pos_x += gridDim.x * blockDim.x) {
    for (size_t pos_y = blockIdx.y * blockDim.y + threadIdx.y; pos_y < shrink_num - 1;
         pos_y += gridDim.y * blockDim.y) {
      int32_t start_row = thready_pos_shrink[pos_y];
      int32_t end_row = thready_pos_shrink[pos_y + 1];
      S update_pos = indices_sort[start_row] * inner_size + pos_x;
      for (int idx = start_row; idx < end_row; ++idx) {
        int grad_pos = rows_index[idx] * inner_size + pos_x;
        accum[update_pos] = static_cast<T>(*momentum) * accum[update_pos] + grad[grad_pos];
        var[update_pos] = var[update_pos] - static_cast<T>(*lr) * accum[update_pos];
      }
    }
  }
}

struct GreaterThan {
  __host__ __device__ __forceinline__ bool operator()(const int32_t &val) const { return (val > -1); }
};

template <typename T, typename S>
cudaError_t CalSparseApplyMomentum(const size_t size, const size_t indices_size, T *var, T *accum, const T *lr,
                                   const T *grad, const S *indices, const T *momentum, const bool use_nesterov,
                                   S *indices_sort, int32_t *rows_index, int32_t *thready_pos,
                                   int32_t *thready_pos_shrink, int32_t *shrink_num, T *var_out,
                                   const uint32_t &device_id, cudaStream_t cuda_stream) {
  auto policy = thrust::cuda::par.on(cuda_stream);
  thrust::sequence(policy, thrust::device_pointer_cast(rows_index),
                   thrust::device_pointer_cast(rows_index) + indices_size);
  thrust::copy(thrust::device_pointer_cast(indices), thrust::device_pointer_cast(indices) + indices_size,
               thrust::device_pointer_cast(indices_sort));
  thrust::stable_sort_by_key(policy, thrust::device_pointer_cast(indices_sort),
                             thrust::device_pointer_cast(indices_sort) + indices_size,
                             thrust::device_pointer_cast(rows_index));

  const int inner_size = static_cast<int>(size / indices_size);
  SumOfRows<<<CUDA_BLOCKS(device_id, indices_size + 1), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    indices_sort, indices_size + 1, thready_pos);

  GreaterThan greater;
  void *s_temp_storage = nullptr;
  size_t s_temp_storage_bytes = 0;
  (void)cub::DeviceSelect::If(nullptr, s_temp_storage_bytes, static_cast<float *>(nullptr),
                              static_cast<float *>(nullptr), static_cast<int *>(nullptr), indices_size + 1, greater,
                              cuda_stream);
  (void)cudaMalloc(&s_temp_storage, s_temp_storage_bytes);
  (void)cub::DeviceSelect::If(s_temp_storage, s_temp_storage_bytes, thready_pos, thready_pos_shrink, shrink_num,
                              indices_size + 1, greater, cuda_stream);
  cudaFree(s_temp_storage);
  cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream));
  int32_t h_shrink_num = 0;
  cudaMemcpy(&h_shrink_num, shrink_num, sizeof(int32_t), cudaMemcpyDeviceToHost);

  std::vector<int> thready_pos_shrink_h(indices_size + 1);
  cudaMemcpy(thready_pos_shrink_h.data(), thready_pos_shrink, h_shrink_num * sizeof(int32_t), cudaMemcpyDeviceToHost);

  int32_t thread_y = h_shrink_num - 1 > 128 ? 128 : (h_shrink_num - 1);
  int pow_num = static_cast<int>(log(thread_y * 1.0) / log(2.0)) + 1;
  thread_y = static_cast<int>(pow(2.0, pow_num));
  int32_t thread_x = 512 / thread_y > inner_size ? inner_size : (512 / thread_y);
  cudaDeviceProp prop;
  (void)cudaGetDeviceProperties(&prop, device_id);
  int max_blocks = prop.multiProcessorCount;
  int block_y = (h_shrink_num - 1) / thread_y > 8 ? 8 : ((h_shrink_num - 1) / thread_y);
  block_y = block_y == 0 ? 1 : block_y;
  int32_t need_block_x = (inner_size - 1) / thread_x + 1;
  int block_x = need_block_x > (max_blocks / block_y) ? (max_blocks / block_y) : need_block_x;
  dim3 block_dim(thread_x, thread_y);
  dim3 grid_dim(block_x, block_y);
  if (use_nesterov) {
    SparseApplyMomentumKernel<<<grid_dim, block_dim, 0, cuda_stream>>>(
      inner_size, var, accum, lr, grad, rows_index, indices_sort, thready_pos_shrink, h_shrink_num, momentum);
  } else {
    SparseApplyMomentumKernel_<<<grid_dim, block_dim, 0, cuda_stream>>>(
      inner_size, var, accum, lr, grad, rows_index, indices_sort, thready_pos_shrink, h_shrink_num, momentum);
  }
  cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream));
  cudaMemcpy(var_out, var, size * sizeof(T), cudaMemcpyDeviceToDevice);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyMomentum<int8_t, int32_t>(
  const size_t size, const size_t indices_size, int8_t *var, int8_t *accum, const int8_t *lr, const int8_t *grad,
  const int32_t *indices, const int8_t *momentum, const bool use_nesterov, int32_t *indices_sort, int32_t *rows_index,
  int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num, int8_t *var_out, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyMomentum<int16_t, int32_t>(
  const size_t size, const size_t indices_size, int16_t *var, int16_t *accum, const int16_t *lr, const int16_t *grad,
  const int32_t *indices, const int16_t *momentum, const bool use_nesterov, int32_t *indices_sort, int32_t *rows_index,
  int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num, int16_t *var_out, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyMomentum<int32_t, int32_t>(
  const size_t size, const size_t indices_size, int32_t *var, int32_t *accum, const int32_t *lr, const int32_t *grad,
  const int32_t *indices, const int32_t *momentum, const bool use_nesterov, int32_t *indices_sort, int32_t *rows_index,
  int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num, int32_t *var_out, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyMomentum<int64_t, int32_t>(
  const size_t size, const size_t indices_size, int64_t *var, int64_t *accum, const int64_t *lr, const int64_t *grad,
  const int32_t *indices, const int64_t *momentum, const bool use_nesterov, int32_t *indices_sort, int32_t *rows_index,
  int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num, int64_t *var_out, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyMomentum<uint8_t, int32_t>(
  const size_t size, const size_t indices_size, uint8_t *var, uint8_t *accum, const uint8_t *lr, const uint8_t *grad,
  const int32_t *indices, const uint8_t *momentum, const bool use_nesterov, int32_t *indices_sort, int32_t *rows_index,
  int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num, uint8_t *var_out, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyMomentum<uint16_t, int32_t>(
  const size_t size, const size_t indices_size, uint16_t *var, uint16_t *accum, const uint16_t *lr,
  const uint16_t *grad, const int32_t *indices, const uint16_t *momentum, const bool use_nesterov,
  int32_t *indices_sort, int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  uint16_t *var_out, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyMomentum<uint32_t, int32_t>(
  const size_t size, const size_t indices_size, uint32_t *var, uint32_t *accum, const uint32_t *lr,
  const uint32_t *grad, const int32_t *indices, const uint32_t *momentum, const bool use_nesterov,
  int32_t *indices_sort, int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  uint32_t *var_out, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyMomentum<uint64_t, int32_t>(
  const size_t size, const size_t indices_size, uint64_t *var, uint64_t *accum, const uint64_t *lr,
  const uint64_t *grad, const int32_t *indices, const uint64_t *momentum, const bool use_nesterov,
  int32_t *indices_sort, int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  uint64_t *var_out, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyMomentum<int8_t, int64_t>(
  const size_t size, const size_t indices_size, int8_t *var, int8_t *accum, const int8_t *lr, const int8_t *grad,
  const int64_t *indices, const int8_t *momentum, const bool use_nesterov, int64_t *indices_sort, int32_t *rows_index,
  int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num, int8_t *var_out, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyMomentum<int16_t, int64_t>(
  const size_t size, const size_t indices_size, int16_t *var, int16_t *accum, const int16_t *lr, const int16_t *grad,
  const int64_t *indices, const int16_t *momentum, const bool use_nesterov, int64_t *indices_sort, int32_t *rows_index,
  int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num, int16_t *var_out, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyMomentum<int32_t, int64_t>(
  const size_t size, const size_t indices_size, int32_t *var, int32_t *accum, const int32_t *lr, const int32_t *grad,
  const int64_t *indices, const int32_t *momentum, const bool use_nesterov, int64_t *indices_sort, int32_t *rows_index,
  int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num, int32_t *var_out, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyMomentum<int64_t, int64_t>(
  const size_t size, const size_t indices_size, int64_t *var, int64_t *accum, const int64_t *lr, const int64_t *grad,
  const int64_t *indices, const int64_t *momentum, const bool use_nesterov, int64_t *indices_sort, int32_t *rows_index,
  int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num, int64_t *var_out, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyMomentum<uint8_t, int64_t>(
  const size_t size, const size_t indices_size, uint8_t *var, uint8_t *accum, const uint8_t *lr, const uint8_t *grad,
  const int64_t *indices, const uint8_t *momentum, const bool use_nesterov, int64_t *indices_sort, int32_t *rows_index,
  int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num, uint8_t *var_out, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyMomentum<uint16_t, int64_t>(
  const size_t size, const size_t indices_size, uint16_t *var, uint16_t *accum, const uint16_t *lr,
  const uint16_t *grad, const int64_t *indices, const uint16_t *momentum, const bool use_nesterov,
  int64_t *indices_sort, int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  uint16_t *var_out, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyMomentum<uint32_t, int64_t>(
  const size_t size, const size_t indices_size, uint32_t *var, uint32_t *accum, const uint32_t *lr,
  const uint32_t *grad, const int64_t *indices, const uint32_t *momentum, const bool use_nesterov,
  int64_t *indices_sort, int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  uint32_t *var_out, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyMomentum<uint64_t, int64_t>(
  const size_t size, const size_t indices_size, uint64_t *var, uint64_t *accum, const uint64_t *lr,
  const uint64_t *grad, const int64_t *indices, const uint64_t *momentum, const bool use_nesterov,
  int64_t *indices_sort, int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  uint64_t *var_out, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyMomentum<half, int32_t>(
  const size_t size, const size_t indices_size, half *var, half *accum, const half *lr, const half *grad,
  const int32_t *indices, const half *momentum, const bool use_nesterov, int32_t *indices_sort, int32_t *rows_index,
  int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num, half *var_out, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyMomentum<float, int32_t>(
  const size_t size, const size_t indices_size, float *var, float *accum, const float *lr, const float *grad,
  const int32_t *indices, const float *momentum, const bool use_nesterov, int32_t *indices_sort, int32_t *rows_index,
  int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num, float *var_out, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyMomentum<double, int32_t>(
  const size_t size, const size_t indices_size, double *var, double *accum, const double *lr, const double *grad,
  const int32_t *indices, const double *momentum, const bool use_nesterov, int32_t *indices_sort, int32_t *rows_index,
  int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num, double *var_out, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyMomentum<half, int64_t>(
  const size_t size, const size_t indices_size, half *var, half *accum, const half *lr, const half *grad,
  const int64_t *indices, const half *momentum, const bool use_nesterov, int64_t *indices_sort, int32_t *rows_index,
  int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num, half *var_out, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyMomentum<float, int64_t>(
  const size_t size, const size_t indices_size, float *var, float *accum, const float *lr, const float *grad,
  const int64_t *indices, const float *momentum, const bool use_nesterov, int64_t *indices_sort, int32_t *rows_index,
  int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num, float *var_out, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyMomentum<double, int64_t>(
  const size_t size, const size_t indices_size, double *var, double *accum, const double *lr, const double *grad,
  const int64_t *indices, const double *momentum, const bool use_nesterov, int64_t *indices_sort, int32_t *rows_index,
  int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num, double *var_out, const uint32_t &device_id,
  cudaStream_t cuda_stream);
