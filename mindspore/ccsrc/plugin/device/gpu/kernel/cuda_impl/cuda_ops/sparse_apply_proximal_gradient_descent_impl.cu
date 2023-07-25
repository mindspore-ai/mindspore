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

#include "sparse_apply_proximal_gradient_descent_impl.cuh"
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <cub/cub.cuh>
#include <iostream>
#include <algorithm>
#include <vector>
#include <typeinfo>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__device__ __forceinline__ T MaxFunc(T x, T y) {
  return max(x, y);
}

template <>
__device__ __forceinline__ half MaxFunc(half x, half y) {
  return max(__half2float(x), __half2float(y));
}

template <typename T>
__device__ __forceinline__ T AbsFunc(T x) {
  return abs(x);
}

template <>
__device__ __forceinline__ half AbsFunc(half x) {
  return __float2half(abs(__half2float(x)));
}

template <typename T>
__device__ __forceinline__ T SgnFunc(T x) {
  return static_cast<T>(x != 0 ? (x > 0 ? 1 : -1) : 0);
}

template <>
__device__ __forceinline__ half SgnFunc(half x) {
  return __float2half(__half2float(x) != 0 ? (__half2float(x) > 0 ? 1 : -1) : 0);
}

template <typename T, typename S>
__global__ void SparseApplyProximalGradientDescentKernel_1(const size_t inner_size, T *var, const T *alpha, const T *l1,
                                                           const T *l2, const T *grad, int32_t *rows_index,
                                                           S *indices_sort, int32_t *thready_pos_shrink,
                                                           int32_t shrink_num) {
  for (size_t pos_x = blockIdx.x * blockDim.x + threadIdx.x; pos_x < inner_size; pos_x += gridDim.x * blockDim.x) {
    for (size_t pos_y = blockIdx.y * blockDim.y + threadIdx.y; pos_y < shrink_num - 1;
         pos_y += gridDim.y * blockDim.y) {
      int32_t start_row = thready_pos_shrink[pos_y];
      int32_t end_row = thready_pos_shrink[pos_y + 1];
      S update_pos = indices_sort[start_row] * inner_size + pos_x;
      for (int idx = start_row; idx < end_row; ++idx) {
        int grad_pos = rows_index[idx] * inner_size + pos_x;
        T prox_v = var[update_pos] - alpha[0] * grad[grad_pos];
        var[update_pos] =
          SgnFunc(prox_v) *
          MaxFunc(static_cast<T>(static_cast<T>(AbsFunc(prox_v)) - alpha[0] * l1[0]), static_cast<T>(0.0)) /
          (static_cast<T>(1) + l2[0] * alpha[0]);
      }
    }
  }
}

template <typename S>
__global__ void SparseApplyProximalGradientDescentKernel_1(const size_t inner_size, uint8_t *var, const uint8_t *alpha,
                                                           const uint8_t *l1, const uint8_t *l2, const uint8_t *grad,
                                                           int32_t *rows_index, S *indices_sort,
                                                           int32_t *thready_pos_shrink, int32_t shrink_num) {
  for (size_t pos_x = blockIdx.x * blockDim.x + threadIdx.x; pos_x < inner_size; pos_x += gridDim.x * blockDim.x) {
    for (size_t pos_y = blockIdx.y * blockDim.y + threadIdx.y; pos_y < shrink_num - 1;
         pos_y += gridDim.y * blockDim.y) {
      int32_t start_row = thready_pos_shrink[pos_y];
      int32_t end_row = thready_pos_shrink[pos_y + 1];
      S update_pos = indices_sort[start_row] * inner_size + pos_x;
      for (int idx = start_row; idx < end_row; ++idx) {
        int grad_pos = rows_index[idx] * inner_size + pos_x;
        int prox_v =
          static_cast<int32_t>(var[update_pos]) - static_cast<int32_t>(alpha[0]) * static_cast<int32_t>(grad[grad_pos]);
        var[update_pos] = static_cast<uint8_t>(
          SgnFunc(prox_v) * MaxFunc(prox_v - static_cast<int32_t>(alpha[0] * l1[0]), static_cast<int32_t>(0)) /
          (static_cast<int32_t>(1) + static_cast<int32_t>(l2[0] * alpha[0])));
      }
    }
  }
}

template <typename S>
__global__ void SparseApplyProximalGradientDescentKernel_1(const size_t inner_size, uint16_t *var,
                                                           const uint16_t *alpha, const uint16_t *l1,
                                                           const uint16_t *l2, const uint16_t *grad,
                                                           int32_t *rows_index, S *indices_sort,
                                                           int32_t *thready_pos_shrink, int32_t shrink_num) {
  for (size_t pos_x = blockIdx.x * blockDim.x + threadIdx.x; pos_x < inner_size; pos_x += gridDim.x * blockDim.x) {
    for (size_t pos_y = blockIdx.y * blockDim.y + threadIdx.y; pos_y < shrink_num - 1;
         pos_y += gridDim.y * blockDim.y) {
      int32_t start_row = thready_pos_shrink[pos_y];
      int32_t end_row = thready_pos_shrink[pos_y + 1];
      S update_pos = indices_sort[start_row] * inner_size + pos_x;
      for (int idx = start_row; idx < end_row; ++idx) {
        int grad_pos = rows_index[idx] * inner_size + pos_x;
        int prox_v =
          static_cast<int32_t>(var[update_pos]) - static_cast<int32_t>(alpha[0]) * static_cast<int32_t>(grad[grad_pos]);
        var[update_pos] = static_cast<uint16_t>(
          SgnFunc(prox_v) * MaxFunc(prox_v - static_cast<int32_t>(alpha[0] * l1[0]), static_cast<int32_t>(0)) /
          (static_cast<int32_t>(1) + static_cast<int32_t>(l2[0] * alpha[0])));
      }
    }
  }
}

template <typename S>
__global__ void SparseApplyProximalGradientDescentKernel_1(const size_t inner_size, uint32_t *var,
                                                           const uint32_t *alpha, const uint32_t *l1,
                                                           const uint32_t *l2, const uint32_t *grad,
                                                           int32_t *rows_index, S *indices_sort,
                                                           int32_t *thready_pos_shrink, int32_t shrink_num) {
  for (size_t pos_x = blockIdx.x * blockDim.x + threadIdx.x; pos_x < inner_size; pos_x += gridDim.x * blockDim.x) {
    for (size_t pos_y = blockIdx.y * blockDim.y + threadIdx.y; pos_y < shrink_num - 1;
         pos_y += gridDim.y * blockDim.y) {
      int32_t start_row = thready_pos_shrink[pos_y];
      int32_t end_row = thready_pos_shrink[pos_y + 1];
      S update_pos = indices_sort[start_row] * inner_size + pos_x;
      for (int idx = start_row; idx < end_row; ++idx) {
        int grad_pos = rows_index[idx] * inner_size + pos_x;
        int prox_v =
          static_cast<int32_t>(var[update_pos]) - static_cast<int32_t>(alpha[0]) * static_cast<int32_t>(grad[grad_pos]);
        var[update_pos] = static_cast<uint32_t>(
          SgnFunc(prox_v) * MaxFunc(prox_v - static_cast<int32_t>(alpha[0] * l1[0]), static_cast<int32_t>(0)) /
          (static_cast<int32_t>(1) + static_cast<int32_t>(l2[0] * alpha[0])));
      }
    }
  }
}

template <typename S>
__global__ void SparseApplyProximalGradientDescentKernel_1(const size_t inner_size, uint64_t *var,
                                                           const uint64_t *alpha, const uint64_t *l1,
                                                           const uint64_t *l2, const uint64_t *grad,
                                                           int32_t *rows_index, S *indices_sort,
                                                           int32_t *thready_pos_shrink, int32_t shrink_num) {
  for (size_t pos_x = blockIdx.x * blockDim.x + threadIdx.x; pos_x < inner_size; pos_x += gridDim.x * blockDim.x) {
    for (size_t pos_y = blockIdx.y * blockDim.y + threadIdx.y; pos_y < shrink_num - 1;
         pos_y += gridDim.y * blockDim.y) {
      int32_t start_row = thready_pos_shrink[pos_y];
      int32_t end_row = thready_pos_shrink[pos_y + 1];
      S update_pos = indices_sort[start_row] * inner_size + pos_x;
      for (int idx = start_row; idx < end_row; ++idx) {
        int grad_pos = rows_index[idx] * inner_size + pos_x;
        int prox_v =
          static_cast<int32_t>(var[update_pos]) - static_cast<int32_t>(alpha[0]) * static_cast<int32_t>(grad[grad_pos]);
        var[update_pos] = static_cast<uint64_t>(
          SgnFunc(prox_v) * MaxFunc(prox_v - static_cast<int32_t>(alpha[0] * l1[0]), static_cast<int32_t>(0)) /
          (static_cast<int32_t>(1) + static_cast<int32_t>(l2[0] * alpha[0])));
      }
    }
  }
}

template <typename T, typename S>
__global__ void SparseApplyProximalGradientDescentKernel_2(const size_t inner_size, T *var, const T *alpha, const T *l1,
                                                           const T *l2, const T *grad, int32_t *rows_index,
                                                           S *indices_sort, int32_t *thready_pos_shrink,
                                                           int32_t shrink_num) {
  for (size_t pos_x = blockIdx.x * blockDim.x + threadIdx.x; pos_x < inner_size; pos_x += gridDim.x * blockDim.x) {
    for (size_t pos_y = blockIdx.y * blockDim.y + threadIdx.y; pos_y < shrink_num - 1;
         pos_y += gridDim.y * blockDim.y) {
      int32_t start_row = thready_pos_shrink[pos_y];
      int32_t end_row = thready_pos_shrink[pos_y + 1];
      S update_pos = indices_sort[start_row] * inner_size + pos_x;
      for (int idx = start_row; idx < end_row; ++idx) {
        int grad_pos = rows_index[idx] * inner_size + pos_x;
        T prox_v = var[update_pos] - alpha[0] * grad[grad_pos];
        var[update_pos] = prox_v / (static_cast<T>(1) + l2[0] * alpha[0]);
      }
    }
  }
}

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

struct GreaterThan {
  __host__ __device__ __forceinline__ bool operator()(const int32_t &val) const { return (val > -1); }
};

template <typename T, typename S>
cudaError_t CalSparseApplyProximalGradientDescent(const size_t size, const size_t indices_size, T *var, const T *alpha,
                                                  const T *l1, const T *l2, const T *grad, const S *indices, T *var_out,
                                                  S *indices_sort, int32_t *rows_index, int32_t *thready_pos,
                                                  int32_t *thready_pos_shrink, int32_t *shrink_num,
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

  SparseApplyProximalGradientDescentKernel_1<<<grid_dim, block_dim, 0, cuda_stream>>>(
    inner_size, var, alpha, l1, l2, grad, rows_index, indices_sort, thready_pos_shrink, h_shrink_num);
  cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream));
  cudaMemcpy(var_out, var, size * sizeof(T), cudaMemcpyDeviceToDevice);
  return GetCudaStatus();
}

template <typename T, typename S>
cudaError_t CalSparseApplyProximalGradientDescent_2(const size_t size, const size_t indices_size, T *var,
                                                    const T *alpha, const T *l1, const T *l2, const T *grad,
                                                    const S *indices, T *var_out, S *indices_sort, int32_t *rows_index,
                                                    int32_t *thready_pos, int32_t *thready_pos_shrink,
                                                    int32_t *shrink_num, const uint32_t &device_id,
                                                    cudaStream_t cuda_stream) {
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

  SparseApplyProximalGradientDescentKernel_2<<<grid_dim, block_dim, 0, cuda_stream>>>(
    inner_size, var, alpha, l1, l2, grad, rows_index, indices_sort, thready_pos_shrink, h_shrink_num);
  cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream));
  cudaMemcpy(var_out, var, size * sizeof(T), cudaMemcpyDeviceToDevice);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent<float, int32_t>(
  const size_t size, const size_t indices_size, float *var, const float *alpha, const float *l1, const float *l2,
  const float *grad, const int32_t *indices, float *var_out, int32_t *indices_sort, int32_t *rows_index,
  int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent<double, int32_t>(
  const size_t size, const size_t indices_size, double *var, const double *alpha, const double *l1, const double *l2,
  const double *grad, const int32_t *indices, double *var_out, int32_t *indices_sort, int32_t *rows_index,
  int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent<half, int32_t>(
  const size_t size, const size_t indices_size, half *var, const half *alpha, const half *l1, const half *l2,
  const half *grad, const int32_t *indices, half *var_out, int32_t *indices_sort, int32_t *rows_index,
  int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent<int8_t, int32_t>(
  const size_t size, const size_t indices_size, int8_t *var, const int8_t *alpha, const int8_t *l1, const int8_t *l2,
  const int8_t *grad, const int32_t *indices, int8_t *var_out, int32_t *indices_sort, int32_t *rows_index,
  int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent<int16_t, int32_t>(
  const size_t size, const size_t indices_size, int16_t *var, const int16_t *alpha, const int16_t *l1,
  const int16_t *l2, const int16_t *grad, const int32_t *indices, int16_t *var_out, int32_t *indices_sort,
  int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent<int32_t, int32_t>(
  const size_t size, const size_t indices_size, int32_t *var, const int32_t *alpha, const int32_t *l1,
  const int32_t *l2, const int32_t *grad, const int32_t *indices, int32_t *var_out, int32_t *indices_sort,
  int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent<int64_t, int32_t>(
  const size_t size, const size_t indices_size, int64_t *var, const int64_t *alpha, const int64_t *l1,
  const int64_t *l2, const int64_t *grad, const int32_t *indices, int64_t *var_out, int32_t *indices_sort,
  int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent<uint8_t, int32_t>(
  const size_t size, const size_t indices_size, uint8_t *var, const uint8_t *alpha, const uint8_t *l1,
  const uint8_t *l2, const uint8_t *grad, const int32_t *indices, uint8_t *var_out, int32_t *indices_sort,
  int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent<uint16_t, int32_t>(
  const size_t size, const size_t indices_size, uint16_t *var, const uint16_t *alpha, const uint16_t *l1,
  const uint16_t *l2, const uint16_t *grad, const int32_t *indices, uint16_t *var_out, int32_t *indices_sort,
  int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent<uint32_t, int32_t>(
  const size_t size, const size_t indices_size, uint32_t *var, const uint32_t *alpha, const uint32_t *l1,
  const uint32_t *l2, const uint32_t *grad, const int32_t *indices, uint32_t *var_out, int32_t *indices_sort,
  int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent<uint64_t, int32_t>(
  const size_t size, const size_t indices_size, uint64_t *var, const uint64_t *alpha, const uint64_t *l1,
  const uint64_t *l2, const uint64_t *grad, const int32_t *indices, uint64_t *var_out, int32_t *indices_sort,
  int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent<float, int64_t>(
  const size_t size, const size_t indices_size, float *var, const float *alpha, const float *l1, const float *l2,
  const float *grad, const int64_t *indices, float *var_out, int64_t *indices_sort, int32_t *rows_index,
  int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent<double, int64_t>(
  const size_t size, const size_t indices_size, double *var, const double *alpha, const double *l1, const double *l2,
  const double *grad, const int64_t *indices, double *var_out, int64_t *indices_sort, int32_t *rows_index,
  int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent<half, int64_t>(
  const size_t size, const size_t indices_size, half *var, const half *alpha, const half *l1, const half *l2,
  const half *grad, const int64_t *indices, half *var_out, int64_t *indices_sort, int32_t *rows_index,
  int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent<int8_t, int64_t>(
  const size_t size, const size_t indices_size, int8_t *var, const int8_t *alpha, const int8_t *l1, const int8_t *l2,
  const int8_t *grad, const int64_t *indices, int8_t *var_out, int64_t *indices_sort, int32_t *rows_index,
  int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent<int16_t, int64_t>(
  const size_t size, const size_t indices_size, int16_t *var, const int16_t *alpha, const int16_t *l1,
  const int16_t *l2, const int16_t *grad, const int64_t *indices, int16_t *var_out, int64_t *indices_sort,
  int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent<int32_t, int64_t>(
  const size_t size, const size_t indices_size, int32_t *var, const int32_t *alpha, const int32_t *l1,
  const int32_t *l2, const int32_t *grad, const int64_t *indices, int32_t *var_out, int64_t *indices_sort,
  int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent<int64_t, int64_t>(
  const size_t size, const size_t indices_size, int64_t *var, const int64_t *alpha, const int64_t *l1,
  const int64_t *l2, const int64_t *grad, const int64_t *indices, int64_t *var_out, int64_t *indices_sort,
  int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent<uint8_t, int64_t>(
  const size_t size, const size_t indices_size, uint8_t *var, const uint8_t *alpha, const uint8_t *l1,
  const uint8_t *l2, const uint8_t *grad, const int64_t *indices, uint8_t *var_out, int64_t *indices_sort,
  int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent<uint16_t, int64_t>(
  const size_t size, const size_t indices_size, uint16_t *var, const uint16_t *alpha, const uint16_t *l1,
  const uint16_t *l2, const uint16_t *grad, const int64_t *indices, uint16_t *var_out, int64_t *indices_sort,
  int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent<uint32_t, int64_t>(
  const size_t size, const size_t indices_size, uint32_t *var, const uint32_t *alpha, const uint32_t *l1,
  const uint32_t *l2, const uint32_t *grad, const int64_t *indices, uint32_t *var_out, int64_t *indices_sort,
  int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent<uint64_t, int64_t>(
  const size_t size, const size_t indices_size, uint64_t *var, const uint64_t *alpha, const uint64_t *l1,
  const uint64_t *l2, const uint64_t *grad, const int64_t *indices, uint64_t *var_out, int64_t *indices_sort,
  int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent_2<float, int32_t>(
  const size_t size, const size_t indices_size, float *var, const float *alpha, const float *l1, const float *l2,
  const float *grad, const int32_t *indices, float *var_out, int32_t *indices_sort, int32_t *rows_index,
  int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent_2<double, int32_t>(
  const size_t size, const size_t indices_size, double *var, const double *alpha, const double *l1, const double *l2,
  const double *grad, const int32_t *indices, double *var_out, int32_t *indices_sort, int32_t *rows_index,
  int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent_2<half, int32_t>(
  const size_t size, const size_t indices_size, half *var, const half *alpha, const half *l1, const half *l2,
  const half *grad, const int32_t *indices, half *var_out, int32_t *indices_sort, int32_t *rows_index,
  int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent_2<int8_t, int32_t>(
  const size_t size, const size_t indices_size, int8_t *var, const int8_t *alpha, const int8_t *l1, const int8_t *l2,
  const int8_t *grad, const int32_t *indices, int8_t *var_out, int32_t *indices_sort, int32_t *rows_index,
  int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent_2<int16_t, int32_t>(
  const size_t size, const size_t indices_size, int16_t *var, const int16_t *alpha, const int16_t *l1,
  const int16_t *l2, const int16_t *grad, const int32_t *indices, int16_t *var_out, int32_t *indices_sort,
  int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent_2<int32_t, int32_t>(
  const size_t size, const size_t indices_size, int32_t *var, const int32_t *alpha, const int32_t *l1,
  const int32_t *l2, const int32_t *grad, const int32_t *indices, int32_t *var_out, int32_t *indices_sort,
  int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent_2<int64_t, int32_t>(
  const size_t size, const size_t indices_size, int64_t *var, const int64_t *alpha, const int64_t *l1,
  const int64_t *l2, const int64_t *grad, const int32_t *indices, int64_t *var_out, int32_t *indices_sort,
  int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent_2<uint8_t, int32_t>(
  const size_t size, const size_t indices_size, uint8_t *var, const uint8_t *alpha, const uint8_t *l1,
  const uint8_t *l2, const uint8_t *grad, const int32_t *indices, uint8_t *var_out, int32_t *indices_sort,
  int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent_2<uint16_t, int32_t>(
  const size_t size, const size_t indices_size, uint16_t *var, const uint16_t *alpha, const uint16_t *l1,
  const uint16_t *l2, const uint16_t *grad, const int32_t *indices, uint16_t *var_out, int32_t *indices_sort,
  int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent_2<uint32_t, int32_t>(
  const size_t size, const size_t indices_size, uint32_t *var, const uint32_t *alpha, const uint32_t *l1,
  const uint32_t *l2, const uint32_t *grad, const int32_t *indices, uint32_t *var_out, int32_t *indices_sort,
  int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent_2<uint64_t, int32_t>(
  const size_t size, const size_t indices_size, uint64_t *var, const uint64_t *alpha, const uint64_t *l1,
  const uint64_t *l2, const uint64_t *grad, const int32_t *indices, uint64_t *var_out, int32_t *indices_sort,
  int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent_2<float, int64_t>(
  const size_t size, const size_t indices_size, float *var, const float *alpha, const float *l1, const float *l2,
  const float *grad, const int64_t *indices, float *var_out, int64_t *indices_sort, int32_t *rows_index,
  int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent_2<double, int64_t>(
  const size_t size, const size_t indices_size, double *var, const double *alpha, const double *l1, const double *l2,
  const double *grad, const int64_t *indices, double *var_out, int64_t *indices_sort, int32_t *rows_index,
  int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent_2<half, int64_t>(
  const size_t size, const size_t indices_size, half *var, const half *alpha, const half *l1, const half *l2,
  const half *grad, const int64_t *indices, half *var_out, int64_t *indices_sort, int32_t *rows_index,
  int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent_2<int8_t, int64_t>(
  const size_t size, const size_t indices_size, int8_t *var, const int8_t *alpha, const int8_t *l1, const int8_t *l2,
  const int8_t *grad, const int64_t *indices, int8_t *var_out, int64_t *indices_sort, int32_t *rows_index,
  int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent_2<int16_t, int64_t>(
  const size_t size, const size_t indices_size, int16_t *var, const int16_t *alpha, const int16_t *l1,
  const int16_t *l2, const int16_t *grad, const int64_t *indices, int16_t *var_out, int64_t *indices_sort,
  int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent_2<int32_t, int64_t>(
  const size_t size, const size_t indices_size, int32_t *var, const int32_t *alpha, const int32_t *l1,
  const int32_t *l2, const int32_t *grad, const int64_t *indices, int32_t *var_out, int64_t *indices_sort,
  int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent_2<int64_t, int64_t>(
  const size_t size, const size_t indices_size, int64_t *var, const int64_t *alpha, const int64_t *l1,
  const int64_t *l2, const int64_t *grad, const int64_t *indices, int64_t *var_out, int64_t *indices_sort,
  int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent_2<uint8_t, int64_t>(
  const size_t size, const size_t indices_size, uint8_t *var, const uint8_t *alpha, const uint8_t *l1,
  const uint8_t *l2, const uint8_t *grad, const int64_t *indices, uint8_t *var_out, int64_t *indices_sort,
  int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent_2<uint16_t, int64_t>(
  const size_t size, const size_t indices_size, uint16_t *var, const uint16_t *alpha, const uint16_t *l1,
  const uint16_t *l2, const uint16_t *grad, const int64_t *indices, uint16_t *var_out, int64_t *indices_sort,
  int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent_2<uint32_t, int64_t>(
  const size_t size, const size_t indices_size, uint32_t *var, const uint32_t *alpha, const uint32_t *l1,
  const uint32_t *l2, const uint32_t *grad, const int64_t *indices, uint32_t *var_out, int64_t *indices_sort,
  int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyProximalGradientDescent_2<uint64_t, int64_t>(
  const size_t size, const size_t indices_size, uint64_t *var, const uint64_t *alpha, const uint64_t *l1,
  const uint64_t *l2, const uint64_t *grad, const int64_t *indices, uint64_t *var_out, int64_t *indices_sort,
  int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink, int32_t *shrink_num,
  const uint32_t &device_id, cudaStream_t cuda_stream);
