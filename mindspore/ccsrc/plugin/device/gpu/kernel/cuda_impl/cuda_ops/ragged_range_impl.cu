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

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/ragged_range_impl.cuh"

template <typename T, typename TSPLITS>
__device__ TSPLITS RangeSize(const T start, const T limit, const T delta) {
  int64_t range_size;
  if (((delta > (T)0) && (limit < start)) || ((delta < (T)0) && (limit > start))) {
    range_size = 0;
  } else if (std::is_integral<T>::value) {
    range_size = (std::abs(static_cast<int64_t>(limit) - static_cast<int64_t>(start)) +
                  std::abs(static_cast<int64_t>(delta)) - 1) /
                 std::abs(static_cast<int64_t>(delta));
  } else {
    range_size =
      std::ceil(std::abs((static_cast<double>(limit) - static_cast<double>(start)) / static_cast<double>(delta)));
  }
  return range_size;
}

template <typename T, typename TSPLITS>
__global__ void CalRangeSizes(TSPLITS *range_sizes_addr, T *starts_addr, T *limits_addr, T *deltas_addr,
                              const size_t nrows, bool broadcast_starts, bool broadcast_limits, bool broadcast_deltas) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nrows; pos += blockDim.x * gridDim.x) {
    T start = broadcast_starts ? starts_addr[0] : starts_addr[pos];
    T limit = broadcast_limits ? limits_addr[0] : limits_addr[pos];
    T delta = broadcast_deltas ? deltas_addr[0] : deltas_addr[pos];
    range_sizes_addr[pos] = RangeSize<T, TSPLITS>(start, limit, delta);
  }
}

template <typename TSPLITS>
__global__ void SetNestedSplitsStartingZero(TSPLITS *rt_nested_splits_addr) {
  rt_nested_splits_addr[0] = 0;
}

template <typename T, typename TSPLITS>
__global__ void RaggedRange(T *starts_addr, T *deltas_addr, T *output, TSPLITS *rt_nested_splits_addr,
                            bool broadcast_starts, bool broadcast_deltas, const size_t nrows) {
  for (int pos_y = blockIdx.y * blockDim.y + threadIdx.y; pos_y < nrows; pos_y += blockDim.y * gridDim.y) {
    TSPLITS size = rt_nested_splits_addr[pos_y + 1] - rt_nested_splits_addr[pos_y];
    T start = broadcast_starts ? starts_addr[0] : starts_addr[pos_y];
    T delta = broadcast_deltas ? deltas_addr[0] : deltas_addr[pos_y];
    TSPLITS start_index = rt_nested_splits_addr[pos_y];
    for (int pos_x = threadIdx.x; pos_x < size; pos_x += blockDim.x) {
      output[start_index + pos_x] = pos_x * delta + start;
    }
  }
}

template <typename T, typename TSPLITS>
void CalRaggedRange(T *starts_addr, T *limits_addr, T *deltas_addr, TSPLITS *rt_nested_splits_addr,
                    T *rt_dense_values_addr, TSPLITS *range_sizes_addr, const size_t nrows, bool broadcast_starts,
                    bool broadcast_limits, bool broadcast_deltas, const uint32_t &device_id, cudaStream_t cuda_stream) {
  cudaDeviceProp prop;
  (void)cudaGetDeviceProperties(&prop, device_id);
  int max_blocks = prop.multiProcessorCount;
  int block_num = std::min(static_cast<int>(((static_cast<int64_t>(nrows) - 1) / 256) + 1), max_blocks);
  CalRangeSizes<<<block_num, 256, 0, cuda_stream>>>(range_sizes_addr, starts_addr, limits_addr, deltas_addr, nrows,
                                                    broadcast_starts, broadcast_limits, broadcast_deltas);

  SetNestedSplitsStartingZero<<<1, 1, 0, cuda_stream>>>(rt_nested_splits_addr);
  if (nrows == 0) {
    return;
  }
  size_t temp_storage_bytes = 0;
  (void)cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, range_sizes_addr, rt_nested_splits_addr + 1, nrows,
                                      cuda_stream);
  void *d_temp_storage = nullptr;
  (void)cudaMalloc(&d_temp_storage, temp_storage_bytes);
  (void)cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, range_sizes_addr, rt_nested_splits_addr + 1,
                                      nrows, cuda_stream);
  (void)cudaFree(d_temp_storage);
  (void)cudaStreamSynchronize(cuda_stream);

  TSPLITS avg_rt_nested_splits = 0;
  (void)cudaMemcpy(&avg_rt_nested_splits, rt_nested_splits_addr + nrows, sizeof(TSPLITS), cudaMemcpyDeviceToHost);
  avg_rt_nested_splits /= nrows;
  size_t thread_x_num, thread_y_num;
  if (avg_rt_nested_splits == 0) {
    thread_x_num = 32;
  } else if (avg_rt_nested_splits > 128) {
    thread_x_num = 128;
  } else {
    thread_x_num = avg_rt_nested_splits;
  }
  thread_y_num = 256 / thread_x_num;
  dim3 thread_num(thread_x_num, thread_y_num);
  block_num = std::min(
    static_cast<int>(((avg_rt_nested_splits * static_cast<int64_t>(nrows) - 1) / (thread_num.x * thread_num.y)) + 1),
    max_blocks);
  dim3 block(1, block_num);
  RaggedRange<<<block, thread_num, 0, cuda_stream>>>(starts_addr, deltas_addr, rt_dense_values_addr,
                                                     rt_nested_splits_addr, broadcast_starts, broadcast_deltas, nrows);
}

template CUDA_LIB_EXPORT void CalRaggedRange<int32_t, int32_t>(int32_t *starts_addr, int32_t *limits_addr,
                                                               int32_t *deltas_addr, int32_t *rt_nested_splits_addr,
                                                               int32_t *rt_dense_values_addr, int32_t *range_sizes_addr,
                                                               const size_t nrows, bool broadcast_starts,
                                                               bool broadcast_limits, bool broadcast_deltas,
                                                               const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalRaggedRange<int32_t, int64_t>(int32_t *starts_addr, int32_t *limits_addr,
                                                               int32_t *deltas_addr, int64_t *rt_nested_splits_addr,
                                                               int32_t *rt_dense_values_addr, int64_t *range_sizes_addr,
                                                               const size_t nrows, bool broadcast_starts,
                                                               bool broadcast_limits, bool broadcast_deltas,
                                                               const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalRaggedRange<int64_t, int32_t>(int64_t *starts_addr, int64_t *limits_addr,
                                                               int64_t *deltas_addr, int32_t *rt_nested_splits_addr,
                                                               int64_t *rt_dense_values_addr, int32_t *range_sizes_addr,
                                                               const size_t nrows, bool broadcast_starts,
                                                               bool broadcast_limits, bool broadcast_deltas,
                                                               const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalRaggedRange<int64_t, int64_t>(int64_t *starts_addr, int64_t *limits_addr,
                                                               int64_t *deltas_addr, int64_t *rt_nested_splits_addr,
                                                               int64_t *rt_dense_values_addr, int64_t *range_sizes_addr,
                                                               const size_t nrows, bool broadcast_starts,
                                                               bool broadcast_limits, bool broadcast_deltas,
                                                               const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalRaggedRange<float, int32_t>(float *starts_addr, float *limits_addr, float *deltas_addr,
                                                             int32_t *rt_nested_splits_addr,
                                                             float *rt_dense_values_addr, int32_t *range_sizes_addr,
                                                             const size_t nrows, bool broadcast_starts,
                                                             bool broadcast_limits, bool broadcast_deltas,
                                                             const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalRaggedRange<float, int64_t>(float *starts_addr, float *limits_addr, float *deltas_addr,
                                                             int64_t *rt_nested_splits_addr,
                                                             float *rt_dense_values_addr, int64_t *range_sizes_addr,
                                                             const size_t nrows, bool broadcast_starts,
                                                             bool broadcast_limits, bool broadcast_deltas,
                                                             const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalRaggedRange<double, int32_t>(double *starts_addr, double *limits_addr,
                                                              double *deltas_addr, int32_t *rt_nested_splits_addr,
                                                              double *rt_dense_values_addr, int32_t *range_sizes_addr,
                                                              const size_t nrows, bool broadcast_starts,
                                                              bool broadcast_limits, bool broadcast_deltas,
                                                              const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalRaggedRange<double, int64_t>(double *starts_addr, double *limits_addr,
                                                              double *deltas_addr, int64_t *rt_nested_splits_addr,
                                                              double *rt_dense_values_addr, int64_t *range_sizes_addr,
                                                              const size_t nrows, bool broadcast_starts,
                                                              bool broadcast_limits, bool broadcast_deltas,
                                                              const uint32_t &device_id, cudaStream_t cuda_stream);
