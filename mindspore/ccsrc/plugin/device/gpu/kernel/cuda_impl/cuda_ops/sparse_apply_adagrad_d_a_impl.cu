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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sparse_apply_adagrad_d_a_impl.cuh"
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <cub/cub.cuh>
#include <iostream>
#include <algorithm>
#include <vector>
#include <typeinfo>
#include "include/cuda_fp16.h"

template <typename T>
__device__ __forceinline__ T AbsFunc(T x) {
  return abs(x);
}

template <>
__device__ __forceinline__ half AbsFunc(half x) {
  return abs(__half2float(x));
}

template <typename T>
__device__ __forceinline__ T MaxFunc(T x, T y) {
  return max(x, y);
}

template <>
__device__ __forceinline__ half MaxFunc(half x, half y) {
  return max(__half2float(x), __half2float(y));
}

template <typename T>
__device__ __forceinline__ T Sign(T num) {
  if (num > static_cast<T>(0.0)) {
    return static_cast<T>(1.0);
  } else if (num == static_cast<T>(0.0)) {
    return static_cast<T>(0.0);
  } else {
    return static_cast<T>(-1.0);
  }
}

template <typename T, typename S, typename S2>
__global__ void SparseApplyAdagradDAKernel(const size_t inner_size, T *var, T *accum, T *squared_accum, const T *grad,
                                           const T *lr, const T *l1, const T *l2, const S2 *global_step,
                                           int32_t *rows_index, S *indices_sort, int32_t *thready_pos_shrink,
                                           int32_t shrink_num) {
  T zero = static_cast<T>(0.0);
  T minus_one = static_cast<T>(-1);
  T global_step_scalar = static_cast<T>(static_cast<double>(global_step[0]));
  T gs_lr = global_step_scalar * lr[0];
  T l1_scalar = l1[0];
  T l2_scalar = l2[0];

  for (size_t pos_x = blockIdx.x * blockDim.x + threadIdx.x; pos_x < inner_size; pos_x += gridDim.x * blockDim.x) {
    for (size_t pos_y = blockIdx.y * blockDim.y + threadIdx.y; pos_y < shrink_num - 1;
         pos_y += gridDim.y * blockDim.y) {
      int32_t start_row = thready_pos_shrink[pos_y];
      int32_t end_row = thready_pos_shrink[pos_y + 1];
      S update_pos = indices_sort[start_row] * inner_size + pos_x;
      for (int idx = start_row; idx < end_row; ++idx) {
        int grad_pos = rows_index[idx] * inner_size + pos_x;
        accum[update_pos] += grad[grad_pos];
        squared_accum[update_pos] += grad[grad_pos] * grad[grad_pos];
        if (squared_accum[update_pos] <= zero) {
          var[update_pos] = NAN;
          continue;
        }
        if (gs_lr == zero) {
          var[update_pos] = 0;
          continue;
        }
        if (l1_scalar > zero) {
          var[update_pos] =
            minus_one * static_cast<T>(Sign(static_cast<double>(accum[update_pos]))) *
            static_cast<T>(MaxFunc(
              static_cast<double>(
                (static_cast<T>(AbsFunc(static_cast<double>(accum[update_pos]))) / global_step_scalar) - l1_scalar),
              static_cast<double>(0.0))) /
            (l2_scalar + static_cast<T>(sqrt(static_cast<double>(squared_accum[update_pos]))) / gs_lr);
        } else {
          var[update_pos] = minus_one * (accum[update_pos] / global_step_scalar) /
                            (l2_scalar + static_cast<T>(sqrt(static_cast<double>(squared_accum[update_pos]))) / gs_lr);
        }
      }
    }
  }
}

template <>
__global__ void SparseApplyAdagradDAKernel(const size_t inner_size, half *var, half *accum, half *squared_accum,
                                           const half *grad, const half *lr, const half *l1, const half *l2,
                                           const int32_t *global_step, int32_t *rows_index, int32_t *indices_sort,
                                           int32_t *thready_pos_shrink, int32_t shrink_num) {
  float zero = static_cast<float>(0.0);
  float minus_one = static_cast<float>(-1);
  float global_step_scalar = static_cast<float>(global_step[0]);
  float gs_lr = global_step_scalar * __half2float(lr[0]);
  float l1_scalar = __half2float(l1[0]);
  float l2_scalar = __half2float(l2[0]);

  for (size_t pos_x = blockIdx.x * blockDim.x + threadIdx.x; pos_x < inner_size; pos_x += gridDim.x * blockDim.x) {
    for (size_t pos_y = blockIdx.y * blockDim.y + threadIdx.y; pos_y < shrink_num - 1;
         pos_y += gridDim.y * blockDim.y) {
      int32_t start_row = thready_pos_shrink[pos_y];
      int32_t end_row = thready_pos_shrink[pos_y + 1];
      int32_t update_pos = indices_sort[start_row] * inner_size + pos_x;
      for (int idx = start_row; idx < end_row; ++idx) {
        int grad_pos = rows_index[idx] * inner_size + pos_x;
        accum[update_pos] += grad[grad_pos];
        squared_accum[update_pos] += grad[grad_pos] * grad[grad_pos];
        if (squared_accum[update_pos] <= __float2half(zero)) {
          var[update_pos] = NAN;
          continue;
        }
        if (gs_lr == zero) {
          var[update_pos] = zero;
          continue;
        }
        if (l1_scalar > zero) {
          var[update_pos] =
            __float2half(minus_one * (Sign(__half2float(accum[update_pos]))) *
                         (MaxFunc((((AbsFunc(__half2float(accum[update_pos]))) / global_step_scalar) - l1_scalar),
                                  static_cast<float>(0.0))) /
                         (l2_scalar + (sqrt(__half2float(squared_accum[update_pos]))) / gs_lr));
        } else {
          var[update_pos] = __float2half(minus_one * (__half2float(accum[update_pos]) / global_step_scalar) /
                                         (l2_scalar + (sqrt(__half2float(squared_accum[update_pos]))) / gs_lr));
        }
      }
    }
  }
}

template <>
__global__ void SparseApplyAdagradDAKernel(const size_t inner_size, half *var, half *accum, half *squared_accum,
                                           const half *grad, const half *lr, const half *l1, const half *l2,
                                           const int64_t *global_step, int32_t *rows_index, int32_t *indices_sort,
                                           int32_t *thready_pos_shrink, int32_t shrink_num) {
  float zero = static_cast<float>(0.0);
  float minus_one = static_cast<float>(-1);
  float global_step_scalar = static_cast<float>(global_step[0]);
  float gs_lr = global_step_scalar * __half2float(lr[0]);
  float l1_scalar = __half2float(l1[0]);
  float l2_scalar = __half2float(l2[0]);

  for (size_t pos_x = blockIdx.x * blockDim.x + threadIdx.x; pos_x < inner_size; pos_x += gridDim.x * blockDim.x) {
    for (size_t pos_y = blockIdx.y * blockDim.y + threadIdx.y; pos_y < shrink_num - 1;
         pos_y += gridDim.y * blockDim.y) {
      int32_t start_row = thready_pos_shrink[pos_y];
      int32_t end_row = thready_pos_shrink[pos_y + 1];
      int32_t update_pos = indices_sort[start_row] * inner_size + pos_x;
      for (int idx = start_row; idx < end_row; ++idx) {
        int grad_pos = rows_index[idx] * inner_size + pos_x;
        accum[update_pos] += grad[grad_pos];
        squared_accum[update_pos] += grad[grad_pos] * grad[grad_pos];
        if (squared_accum[update_pos] <= __float2half(zero)) {
          var[update_pos] = NAN;
          continue;
        }
        if (gs_lr == zero) {
          var[update_pos] = 0;
          continue;
        }
        if (l1_scalar > zero) {
          var[update_pos] =
            __float2half(minus_one * (Sign(__half2float(accum[update_pos]))) *
                         (MaxFunc((((AbsFunc(__half2float(accum[update_pos]))) / global_step_scalar) - l1_scalar),
                                  static_cast<float>(0.0))) /
                         (l2_scalar + (sqrt(__half2float(squared_accum[update_pos]))) / gs_lr));
        } else {
          var[update_pos] = __float2half(minus_one * (__half2float(accum[update_pos]) / global_step_scalar) /
                                         (l2_scalar + (sqrt(__half2float(squared_accum[update_pos]))) / gs_lr));
        }
      }
    }
  }
}

template <>
__global__ void SparseApplyAdagradDAKernel(const size_t inner_size, half *var, half *accum, half *squared_accum,
                                           const half *grad, const half *lr, const half *l1, const half *l2,
                                           const int64_t *global_step, int32_t *rows_index, int64_t *indices_sort,
                                           int32_t *thready_pos_shrink, int32_t shrink_num) {
  float zero = static_cast<float>(0.0);
  float minus_one = static_cast<float>(-1);
  float global_step_scalar = static_cast<float>(global_step[0]);
  float gs_lr = global_step_scalar * __half2float(lr[0]);
  float l1_scalar = __half2float(l1[0]);
  float l2_scalar = __half2float(l2[0]);

  for (size_t pos_x = blockIdx.x * blockDim.x + threadIdx.x; pos_x < inner_size; pos_x += gridDim.x * blockDim.x) {
    for (size_t pos_y = blockIdx.y * blockDim.y + threadIdx.y; pos_y < shrink_num - 1;
         pos_y += gridDim.y * blockDim.y) {
      int32_t start_row = thready_pos_shrink[pos_y];
      int32_t end_row = thready_pos_shrink[pos_y + 1];
      int64_t update_pos = indices_sort[start_row] * inner_size + pos_x;
      for (int idx = start_row; idx < end_row; ++idx) {
        int grad_pos = rows_index[idx] * inner_size + pos_x;
        accum[update_pos] += grad[grad_pos];
        squared_accum[update_pos] += grad[grad_pos] * grad[grad_pos];
        if (squared_accum[update_pos] <= __float2half(zero)) {
          var[update_pos] = NAN;
          continue;
        }
        if (gs_lr == zero) {
          var[update_pos] = 0;
          continue;
        }
        if (l1_scalar > zero) {
          var[update_pos] =
            __float2half(minus_one * (Sign(__half2float(accum[update_pos]))) *
                         (MaxFunc((((AbsFunc(__half2float(accum[update_pos]))) / global_step_scalar) - l1_scalar),
                                  static_cast<float>(0.0))) /
                         (l2_scalar + (sqrt(__half2float(squared_accum[update_pos]))) / gs_lr));
        } else {
          var[update_pos] = __float2half(minus_one * (__half2float(accum[update_pos]) / global_step_scalar) /
                                         (l2_scalar + (sqrt(__half2float(squared_accum[update_pos]))) / gs_lr));
        }
      }
    }
  }
}

template <>
__global__ void SparseApplyAdagradDAKernel(const size_t inner_size, half *var, half *accum, half *squared_accum,
                                           const half *grad, const half *lr, const half *l1, const half *l2,
                                           const int32_t *global_step, int32_t *rows_index, int64_t *indices_sort,
                                           int32_t *thready_pos_shrink, int32_t shrink_num) {
  float zero = static_cast<float>(0.0);
  float minus_one = static_cast<float>(-1);
  float global_step_scalar = static_cast<float>(global_step[0]);
  float gs_lr = global_step_scalar * __half2float(lr[0]);
  float l1_scalar = __half2float(l1[0]);
  float l2_scalar = __half2float(l2[0]);

  for (size_t pos_x = blockIdx.x * blockDim.x + threadIdx.x; pos_x < inner_size; pos_x += gridDim.x * blockDim.x) {
    for (size_t pos_y = blockIdx.y * blockDim.y + threadIdx.y; pos_y < shrink_num - 1;
         pos_y += gridDim.y * blockDim.y) {
      int32_t start_row = thready_pos_shrink[pos_y];
      int32_t end_row = thready_pos_shrink[pos_y + 1];
      int64_t update_pos = indices_sort[start_row] * inner_size + pos_x;
      for (int idx = start_row; idx < end_row; ++idx) {
        int grad_pos = rows_index[idx] * inner_size + pos_x;
        accum[update_pos] += grad[grad_pos];
        squared_accum[update_pos] += grad[grad_pos] * grad[grad_pos];
        if (squared_accum[update_pos] <= __float2half(zero)) {
          var[update_pos] = NAN;
          continue;
        }
        if (gs_lr == zero) {
          var[update_pos] = 0;
          continue;
        }
        if (l1_scalar > zero) {
          var[update_pos] =
            __float2half(minus_one * (Sign(__half2float(accum[update_pos]))) *
                         (MaxFunc((((AbsFunc(__half2float(accum[update_pos]))) / global_step_scalar) - l1_scalar),
                                  static_cast<float>(0.0))) /
                         (l2_scalar + (sqrt(__half2float(squared_accum[update_pos]))) / gs_lr));
        } else {
          var[update_pos] = __float2half(minus_one * (__half2float(accum[update_pos]) / global_step_scalar) /
                                         (l2_scalar + (sqrt(__half2float(squared_accum[update_pos]))) / gs_lr));
        }
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

template <typename T, typename S, typename S2>
cudaError_t CalSparseApplyAdagradDA(const size_t batch_size, size_t indices_size, const size_t size, T *var, T *accum,
                                    T *squared_accum, const T *grad, const S *indices, const T *lr, const T *l1,
                                    const T *l2, const S2 *global_step, T *output_var, S *indices_sort,
                                    int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink,
                                    int32_t *shrink_num, const uint32_t &device_id, cudaStream_t cuda_stream) {
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

  SparseApplyAdagradDAKernel<<<grid_dim, block_dim, 0, cuda_stream>>>(inner_size, var, accum, squared_accum, grad, lr,
                                                                      l1, l2, global_step, rows_index, indices_sort,
                                                                      thready_pos_shrink, h_shrink_num);

  cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream));
  cudaMemcpy(output_var, var, size * sizeof(T), cudaMemcpyDeviceToDevice);
  return GetCudaStatus();
}

template <typename T, typename S, typename S1>
CUDA_LIB_EXPORT cudaError_t CalSparseApplyAdagradDA(const size_t batch_size, size_t indices_size, const size_t size,
                                                    T *var, T *accum, T *squared_accum, const T *grad, const S *indices,
                                                    const T *lr, const T *l1, const T *l2, const S1 *global_step,
                                                    T *output_var, S *indices_sort, int32_t *rows_index,
                                                    int32_t *thready_pos, int32_t *thready_pos_shrink,
                                                    int32_t *shrink_num, const uint32_t &device_id,
                                                    cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseApplyAdagradDA<int8_t, int32_t, int64_t>(
  const size_t batch_size, size_t indices_size, const size_t size, int8_t *var, int8_t *accum, int8_t *squared_accum,
  const int8_t *grad, const int32_t *indices, const int8_t *lr, const int8_t *l1, const int8_t *l2,
  const int64_t *global_step, int8_t *output_var, int32_t *indices_sort, int32_t *rows_index, int32_t *thready_pos,
  int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyAdagradDA<int8_t, int64_t, int64_t>(
  const size_t batch_size, size_t indices_size, const size_t size, int8_t *var, int8_t *accum, int8_t *squared_accum,
  const int8_t *grad, const int64_t *indices, const int8_t *lr, const int8_t *l1, const int8_t *l2,
  const int64_t *global_step, int8_t *output_var, int64_t *indices_sort, int32_t *rows_index, int32_t *thready_pos,
  int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyAdagradDA<int8_t, int32_t, int32_t>(
  const size_t batch_size, size_t indices_size, const size_t size, int8_t *var, int8_t *accum, int8_t *squared_accum,
  const int8_t *grad, const int32_t *indices, const int8_t *lr, const int8_t *l1, const int8_t *l2,
  const int32_t *global_step, int8_t *output_var, int32_t *indices_sort, int32_t *rows_index, int32_t *thready_pos,
  int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyAdagradDA<int8_t, int64_t, int32_t>(
  const size_t batch_size, size_t indices_size, const size_t size, int8_t *var, int8_t *accum, int8_t *squared_accum,
  const int8_t *grad, const int64_t *indices, const int8_t *lr, const int8_t *l1, const int8_t *l2,
  const int32_t *global_step, int8_t *output_var, int64_t *indices_sort, int32_t *rows_index, int32_t *thready_pos,
  int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyAdagradDA<int16_t, int32_t, int64_t>(
  const size_t batch_size, size_t indices_size, const size_t size, int16_t *var, int16_t *accum, int16_t *squared_accum,
  const int16_t *grad, const int32_t *indices, const int16_t *lr, const int16_t *l1, const int16_t *l2,
  const int64_t *global_step, int16_t *output_var, int32_t *indices_sort, int32_t *rows_index, int32_t *thready_pos,
  int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyAdagradDA<int16_t, int64_t, int64_t>(
  const size_t batch_size, size_t indices_size, const size_t size, int16_t *var, int16_t *accum, int16_t *squared_accum,
  const int16_t *grad, const int64_t *indices, const int16_t *lr, const int16_t *l1, const int16_t *l2,
  const int64_t *global_step, int16_t *output_var, int64_t *indices_sort, int32_t *rows_index, int32_t *thready_pos,
  int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyAdagradDA<int16_t, int32_t, int32_t>(
  const size_t batch_size, size_t indices_size, const size_t size, int16_t *var, int16_t *accum, int16_t *squared_accum,
  const int16_t *grad, const int32_t *indices, const int16_t *lr, const int16_t *l1, const int16_t *l2,
  const int32_t *global_step, int16_t *output_var, int32_t *indices_sort, int32_t *rows_index, int32_t *thready_pos,
  int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyAdagradDA<int16_t, int64_t, int32_t>(
  const size_t batch_size, size_t indices_size, const size_t size, int16_t *var, int16_t *accum, int16_t *squared_accum,
  const int16_t *grad, const int64_t *indices, const int16_t *lr, const int16_t *l1, const int16_t *l2,
  const int32_t *global_step, int16_t *output_var, int64_t *indices_sort, int32_t *rows_index, int32_t *thready_pos,
  int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyAdagradDA<int32_t, int32_t, int64_t>(
  const size_t batch_size, size_t indices_size, const size_t size, int32_t *var, int32_t *accum, int32_t *squared_accum,
  const int32_t *grad, const int32_t *indices, const int32_t *lr, const int32_t *l1, const int32_t *l2,
  const int64_t *global_step, int32_t *output_var, int32_t *indices_sort, int32_t *rows_index, int32_t *thready_pos,
  int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyAdagradDA<int32_t, int64_t, int64_t>(
  const size_t batch_size, size_t indices_size, const size_t size, int32_t *var, int32_t *accum, int32_t *squared_accum,
  const int32_t *grad, const int64_t *indices, const int32_t *lr, const int32_t *l1, const int32_t *l2,
  const int64_t *global_step, int32_t *output_var, int64_t *indices_sort, int32_t *rows_index, int32_t *thready_pos,
  int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyAdagradDA<int32_t, int32_t, int32_t>(
  const size_t batch_size, size_t indices_size, const size_t size, int32_t *var, int32_t *accum, int32_t *squared_accum,
  const int32_t *grad, const int32_t *indices, const int32_t *lr, const int32_t *l1, const int32_t *l2,
  const int32_t *global_step, int32_t *output_var, int32_t *indices_sort, int32_t *rows_index, int32_t *thready_pos,
  int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyAdagradDA<int32_t, int64_t, int32_t>(
  const size_t batch_size, size_t indices_size, const size_t size, int32_t *var, int32_t *accum, int32_t *squared_accum,
  const int32_t *grad, const int64_t *indices, const int32_t *lr, const int32_t *l1, const int32_t *l2,
  const int32_t *global_step, int32_t *output_var, int64_t *indices_sort, int32_t *rows_index, int32_t *thready_pos,
  int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyAdagradDA<int64_t, int32_t, int64_t>(
  const size_t batch_size, size_t indices_size, const size_t size, int64_t *var, int64_t *accum, int64_t *squared_accum,
  const int64_t *grad, const int32_t *indices, const int64_t *lr, const int64_t *l1, const int64_t *l2,
  const int64_t *global_step, int64_t *output_var, int32_t *indices_sort, int32_t *rows_index, int32_t *thready_pos,
  int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyAdagradDA<int64_t, int64_t, int64_t>(
  const size_t batch_size, size_t indices_size, const size_t size, int64_t *var, int64_t *accum, int64_t *squared_accum,
  const int64_t *grad, const int64_t *indices, const int64_t *lr, const int64_t *l1, const int64_t *l2,
  const int64_t *global_step, int64_t *output_var, int64_t *indices_sort, int32_t *rows_index, int32_t *thready_pos,
  int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyAdagradDA<int64_t, int32_t, int32_t>(
  const size_t batch_size, size_t indices_size, const size_t size, int64_t *var, int64_t *accum, int64_t *squared_accum,
  const int64_t *grad, const int32_t *indices, const int64_t *lr, const int64_t *l1, const int64_t *l2,
  const int32_t *global_step, int64_t *output_var, int32_t *indices_sort, int32_t *rows_index, int32_t *thready_pos,
  int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyAdagradDA<int64_t, int64_t, int32_t>(
  const size_t batch_size, size_t indices_size, const size_t size, int64_t *var, int64_t *accum, int64_t *squared_accum,
  const int64_t *grad, const int64_t *indices, const int64_t *lr, const int64_t *l1, const int64_t *l2,
  const int32_t *global_step, int64_t *output_var, int64_t *indices_sort, int32_t *rows_index, int32_t *thready_pos,
  int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyAdagradDA<double, int32_t, int64_t>(
  const size_t batch_size, size_t indices_size, const size_t size, double *var, double *accum, double *squared_accum,
  const double *grad, const int32_t *indices, const double *lr, const double *l1, const double *l2,
  const int64_t *global_step, double *output_var, int32_t *indices_sort, int32_t *rows_index, int32_t *thready_pos,
  int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyAdagradDA<double, int64_t, int64_t>(
  const size_t batch_size, size_t indices_size, const size_t size, double *var, double *accum, double *squared_accum,
  const double *grad, const int64_t *indices, const double *lr, const double *l1, const double *l2,
  const int64_t *global_step, double *output_var, int64_t *indices_sort, int32_t *rows_index, int32_t *thready_pos,
  int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyAdagradDA<double, int32_t, int32_t>(
  const size_t batch_size, size_t indices_size, const size_t size, double *var, double *accum, double *squared_accum,
  const double *grad, const int32_t *indices, const double *lr, const double *l1, const double *l2,
  const int32_t *global_step, double *output_var, int32_t *indices_sort, int32_t *rows_index, int32_t *thready_pos,
  int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyAdagradDA<double, int64_t, int32_t>(
  const size_t batch_size, size_t indices_size, const size_t size, double *var, double *accum, double *squared_accum,
  const double *grad, const int64_t *indices, const double *lr, const double *l1, const double *l2,
  const int32_t *global_step, double *output_var, int64_t *indices_sort, int32_t *rows_index, int32_t *thready_pos,
  int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyAdagradDA<float, int32_t, int64_t>(
  const size_t batch_size, size_t indices_size, const size_t size, float *var, float *accum, float *squared_accum,
  const float *grad, const int32_t *indices, const float *lr, const float *l1, const float *l2,
  const int64_t *global_step, float *output_var, int32_t *indices_sort, int32_t *rows_index, int32_t *thready_pos,
  int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyAdagradDA<float, int64_t, int64_t>(
  const size_t batch_size, size_t indices_size, const size_t size, float *var, float *accum, float *squared_accum,
  const float *grad, const int64_t *indices, const float *lr, const float *l1, const float *l2,
  const int64_t *global_step, float *output_var, int64_t *indices_sort, int32_t *rows_index, int32_t *thready_pos,
  int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyAdagradDA<float, int32_t, int32_t>(
  const size_t batch_size, size_t indices_size, const size_t size, float *var, float *accum, float *squared_accum,
  const float *grad, const int32_t *indices, const float *lr, const float *l1, const float *l2,
  const int32_t *global_step, float *output_var, int32_t *indices_sort, int32_t *rows_index, int32_t *thready_pos,
  int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyAdagradDA<float, int64_t, int32_t>(
  const size_t batch_size, size_t indices_size, const size_t size, float *var, float *accum, float *squared_accum,
  const float *grad, const int64_t *indices, const float *lr, const float *l1, const float *l2,
  const int32_t *global_step, float *output_var, int64_t *indices_sort, int32_t *rows_index, int32_t *thready_pos,
  int32_t *thready_pos_shrink, int32_t *shrink_num, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyAdagradDA<half, int32_t, int64_t>(
  const size_t batch_size, size_t indices_size, const size_t size, half *var, half *accum, half *squared_accum,
  const half *grad, const int32_t *indices, const half *lr, const half *l1, const half *l2, const int64_t *global_step,
  half *output_var, int32_t *indices_sort, int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink,
  int32_t *shrink_num, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyAdagradDA<half, int64_t, int64_t>(
  const size_t batch_size, size_t indices_size, const size_t size, half *var, half *accum, half *squared_accum,
  const half *grad, const int64_t *indices, const half *lr, const half *l1, const half *l2, const int64_t *global_step,
  half *output_var, int64_t *indices_sort, int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink,
  int32_t *shrink_num, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyAdagradDA<half, int32_t, int32_t>(
  const size_t batch_size, size_t indices_size, const size_t size, half *var, half *accum, half *squared_accum,
  const half *grad, const int32_t *indices, const half *lr, const half *l1, const half *l2, const int32_t *global_step,
  half *output_var, int32_t *indices_sort, int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink,
  int32_t *shrink_num, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSparseApplyAdagradDA<half, int64_t, int32_t>(
  const size_t batch_size, size_t indices_size, const size_t size, half *var, half *accum, half *squared_accum,
  const half *grad, const int64_t *indices, const half *lr, const half *l1, const half *l2, const int32_t *global_step,
  half *output_var, int64_t *indices_sort, int32_t *rows_index, int32_t *thready_pos, int32_t *thready_pos_shrink,
  int32_t *shrink_num, const uint32_t &device_id, cudaStream_t cuda_stream);
