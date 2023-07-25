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

#include <cub/cub.cuh>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sparse_reorder_impl.cuh"
#include "include/cuda_runtime.h"
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

__global__ void IndicesFlattenKernel(const int num_elems, const int num_dims, const int64_t *indices,
                                     const int64_t *shape, int64_t *flat_indices, int *check_flag) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < num_elems; pos += blockDim.x * gridDim.x) {
    int64_t output_idx = indices[pos * num_dims + num_dims - 1];
    int64_t strides = 1;
    for (int i = num_dims - 2; i >= 0; i--) {
      strides *= shape[i + 1];
      if (indices[pos * num_dims + i + 1] < 0 || indices[pos * num_dims + i + 1] >= shape[i + 1]) {
        *check_flag = 1;
        return;
      }
      output_idx += indices[pos * num_dims + i] * strides;
    }
    flat_indices[pos] = output_idx;
  }
}

template <typename T>
__global__ void PermuteIndicesAndValuesKernel(const int num_elems, const int num_dims, const int64_t *indices,
                                              const T *values, const int64_t *shape, int64_t *permutation_data,
                                              int64_t *y_indices, T *y_values) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < num_elems * num_dims; pos += blockDim.x * gridDim.x) {
    size_t ele_pos = pos / num_dims;
    size_t dim_pos = pos - ele_pos * num_dims;
    y_indices[pos] = indices[permutation_data[ele_pos] * num_dims + dim_pos];
    if (pos % num_dims == 0) {
      y_values[ele_pos] = values[permutation_data[ele_pos]];
    }
  }
}

// namespace str
template <typename T>
CUDA_LIB_EXPORT cudaError_t SparseReorder(const int num_elems, const int num_dims, const int64_t *indices,
                                          const T *values, const int64_t *shape, int64_t *y_indices, T *y_values,
                                          int64_t *flat_indices, int64_t *permutation_data, int32_t *check_flag,
                                          const uint32_t &device_id, cudaStream_t cuda_stream) {
  if (num_dims < 1) {
    return cudaErrorNotReady;
  }
  int thread_num = num_elems > 128 ? 128 : num_elems;
  cudaMemset(check_flag, 0, sizeof(int32_t));
  IndicesFlattenKernel<<<CUDA_BLOCKS_CAL(device_id, num_elems, thread_num), thread_num, 0, cuda_stream>>>(
    num_elems, num_dims, indices, shape, flat_indices, check_flag);
  cudaStreamSynchronize(cuda_stream);
  int32_t check_flag_host = 0;
  cudaMemcpy(&check_flag_host, check_flag, sizeof(int32_t), cudaMemcpyDeviceToHost);
  if (check_flag_host == 1) {
    return cudaErrorNotReady;
  }
  auto policy = thrust::cuda::par.on(cuda_stream);
  thrust::sequence(policy, thrust::device_pointer_cast(permutation_data),
                   thrust::device_pointer_cast(permutation_data) + num_elems);
  thrust::stable_sort_by_key(policy, thrust::device_pointer_cast(flat_indices),
                             thrust::device_pointer_cast(flat_indices) + num_elems,
                             thrust::device_pointer_cast(permutation_data));
  thread_num = num_elems * num_dims > 256 ? 256 : num_elems * num_dims;
  PermuteIndicesAndValuesKernel<<<CUDA_BLOCKS_CAL(device_id, num_elems * num_dims, thread_num), thread_num, 0,
                                  cuda_stream>>>(num_elems, num_dims, indices, values, shape, permutation_data,
                                                 y_indices, y_values);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t SparseReorder<bool>(const int num_elems, const int num_dims,
                                                         const int64_t *indices, const bool *values,
                                                         const int64_t *shape, int64_t *y_indices, bool *y_values,
                                                         int64_t *flat_indices, int64_t *permutation_data,
                                                         int32_t *check_flag, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t SparseReorder<int8_t>(const int num_elems, const int num_dims,
                                                           const int64_t *indices, const int8_t *values,
                                                           const int64_t *shape, int64_t *y_indices, int8_t *y_values,
                                                           int64_t *flat_indices, int64_t *permutation_data,
                                                           int32_t *check_flag, const uint32_t &device_id,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t SparseReorder<int16_t>(const int num_elems, const int num_dims,
                                                            const int64_t *indices, const int16_t *values,
                                                            const int64_t *shape, int64_t *y_indices, int16_t *y_values,
                                                            int64_t *flat_indices, int64_t *permutation_data,
                                                            int32_t *check_flag, const uint32_t &device_id,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t SparseReorder<int32_t>(const int num_elems, const int num_dims,
                                                            const int64_t *indices, const int32_t *values,
                                                            const int64_t *shape, int64_t *y_indices, int32_t *y_values,
                                                            int64_t *flat_indices, int64_t *permutation_data,
                                                            int32_t *check_flag, const uint32_t &device_id,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t SparseReorder<int64_t>(const int num_elems, const int num_dims,
                                                            const int64_t *indices, const int64_t *values,
                                                            const int64_t *shape, int64_t *y_indices, int64_t *y_values,
                                                            int64_t *flat_indices, int64_t *permutation_data,
                                                            int32_t *check_flag, const uint32_t &device_id,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t SparseReorder<uint8_t>(const int num_elems, const int num_dims,
                                                            const int64_t *indices, const uint8_t *values,
                                                            const int64_t *shape, int64_t *y_indices, uint8_t *y_values,
                                                            int64_t *flat_indices, int64_t *permutation_data,
                                                            int32_t *check_flag, const uint32_t &device_id,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t SparseReorder<uint16_t>(const int num_elems, const int num_dims,
                                                             const int64_t *indices, const uint16_t *values,
                                                             const int64_t *shape, int64_t *y_indices,
                                                             uint16_t *y_values, int64_t *flat_indices,
                                                             int64_t *permutation_data, int32_t *check_flag,
                                                             const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t SparseReorder<half>(const int num_elems, const int num_dims,
                                                         const int64_t *indices, const half *values,
                                                         const int64_t *shape, int64_t *y_indices, half *y_values,
                                                         int64_t *flat_indices, int64_t *permutation_data,
                                                         int32_t *check_flag, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t SparseReorder<float>(const int num_elems, const int num_dims,
                                                          const int64_t *indices, const float *values,
                                                          const int64_t *shape, int64_t *y_indices, float *y_values,
                                                          int64_t *flat_indices, int64_t *permutation_data,
                                                          int32_t *check_flag, const uint32_t &device_id,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t SparseReorder<double>(const int num_elems, const int num_dims,
                                                           const int64_t *indices, const double *values,
                                                           const int64_t *shape, int64_t *y_indices, double *y_values,
                                                           int64_t *flat_indices, int64_t *permutation_data,
                                                           int32_t *check_flag, const uint32_t &device_id,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t SparseReorder<cuFloatComplex>(const int num_elems, const int num_dims,
                                                                   const int64_t *indices, const cuFloatComplex *values,
                                                                   const int64_t *shape, int64_t *y_indices,
                                                                   cuFloatComplex *y_values, int64_t *flat_indices,
                                                                   int64_t *permutation_data, int32_t *check_flag,
                                                                   const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t SparseReorder<cuDoubleComplex>(
  const int num_elems, const int num_dims, const int64_t *indices, const cuDoubleComplex *values, const int64_t *shape,
  int64_t *y_indices, cuDoubleComplex *y_values, int64_t *flat_indices, int64_t *permutation_data, int32_t *check_flag,
  const uint32_t &device_id, cudaStream_t cuda_stream);
