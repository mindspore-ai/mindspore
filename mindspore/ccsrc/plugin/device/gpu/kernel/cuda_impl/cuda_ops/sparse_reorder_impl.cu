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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sparse_reorder_impl.cuh"
#include "include/cuda_runtime.h"
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T>
__global__ void IndicesFlattenKernel(const int num_elems, const int num_dims, const int64_t *indices,
                                     const int64_t *shape, int64_t *flat_indices, int64_t *y_indices, T *y_values) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < num_elems; pos += blockDim.x * gridDim.x) {
    CUDA_KERNEL_ASSERT(num_dims >= 1);
    int64_t output_idx = indices[pos * num_dims + num_dims - 1];
    int64_t strides = 1;
    for (int i = num_dims - 2; i >= 0; i--) {
      strides *= shape[i + 1];
      CUDA_KERNEL_ASSERT(0 <= indices[pos * num_dims + i + 1]);
      CUDA_KERNEL_ASSERT(indices[pos * num_dims + i + 1] < shape[i + 1]);
      output_idx += indices[pos * num_dims + i] * strides;
    }
    flat_indices[pos] = output_idx;
  }
}

__global__ void RangeInitKernel(const int num_elems, int64_t *indices_in) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < num_elems; pos += blockDim.x * gridDim.x) {
    indices_in[pos] = pos;
  }
}

template <typename T>
__global__ void PermuteIndicesAndValuesKernel(const int num_elems, const int num_dims, const int64_t *indices,
                                              const T *values, const int64_t *shape, int64_t *permutation_data,
                                              int64_t *y_indices, T *y_values) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < num_elems; pos += blockDim.x * gridDim.x) {
    for (int i = 0; i < num_dims; i++) {
      y_indices[pos * num_dims + i] = indices[permutation_data[pos] * num_dims + i];
    }
    y_values[pos] = values[permutation_data[pos]];
  }
}

// namespace str
template <typename T>
CUDA_LIB_EXPORT void SparseReorder(const int num_elems, const int num_dims, const int64_t *indices, const T *values,
                                   const int64_t *shape, int64_t *y_indices, T *y_values, int64_t *flat_indices,
                                   int64_t *permutation_data, int64_t *keys_out, int64_t *indices_in,
                                   const uint32_t &device_id, cudaStream_t cuda_stream) {
  IndicesFlattenKernel<<<CUDA_BLOCKS(device_id, num_elems), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    num_elems, num_dims, indices, shape, flat_indices, y_indices, y_values);
  size_t temp_storage_bytes = 0;
  RangeInitKernel<<<CUDA_BLOCKS(device_id, num_elems), CUDA_THREADS(device_id), 0, cuda_stream>>>(num_elems,
                                                                                                  indices_in);
  (void)cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, flat_indices, keys_out, indices_in,
                                        permutation_data, num_elems, /*begin_bit=*/0, /*end_bit=*/sizeof(int64_t) * 8,
                                        cuda_stream);
  void *d_temp_storage = nullptr;
  (void)cudaMalloc(&d_temp_storage, temp_storage_bytes);
  (void)cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, flat_indices, keys_out, indices_in,
                                        permutation_data, num_elems, /*begin_bit=*/0, /*end_bit=*/sizeof(int64_t) * 8,
                                        cuda_stream);
  PermuteIndicesAndValuesKernel<<<CUDA_BLOCKS(device_id, num_elems), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    num_elems, num_dims, indices, values, shape, permutation_data, y_indices, y_values);
  (void)cudaFree(d_temp_storage);
}

template CUDA_LIB_EXPORT void SparseReorder<bool>(const int num_elems, const int num_dims, const int64_t *indices,
                                                  const bool *values, const int64_t *shape, int64_t *y_indices,
                                                  bool *y_values, int64_t *flat_indices, int64_t *permutation_data,
                                                  int64_t *keys_out, int64_t *indices_in, const uint32_t &device_id,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseReorder<int8_t>(const int num_elems, const int num_dims, const int64_t *indices,
                                                    const int8_t *values, const int64_t *shape, int64_t *y_indices,
                                                    int8_t *y_values, int64_t *flat_indices, int64_t *permutation_data,
                                                    int64_t *keys_out, int64_t *indices_in, const uint32_t &device_id,
                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseReorder<int16_t>(const int num_elems, const int num_dims, const int64_t *indices,
                                                     const int16_t *values, const int64_t *shape, int64_t *y_indices,
                                                     int16_t *y_values, int64_t *flat_indices,
                                                     int64_t *permutation_data, int64_t *keys_out, int64_t *indices_in,
                                                     const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseReorder<int32_t>(const int num_elems, const int num_dims, const int64_t *indices,
                                                     const int32_t *values, const int64_t *shape, int64_t *y_indices,
                                                     int32_t *y_values, int64_t *flat_indices,
                                                     int64_t *permutation_data, int64_t *keys_out, int64_t *indices_in,
                                                     const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseReorder<int64_t>(const int num_elems, const int num_dims, const int64_t *indices,
                                                     const int64_t *values, const int64_t *shape, int64_t *y_indices,
                                                     int64_t *y_values, int64_t *flat_indices,
                                                     int64_t *permutation_data, int64_t *keys_out, int64_t *indices_in,
                                                     const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseReorder<uint8_t>(const int num_elems, const int num_dims, const int64_t *indices,
                                                     const uint8_t *values, const int64_t *shape, int64_t *y_indices,
                                                     uint8_t *y_values, int64_t *flat_indices,
                                                     int64_t *permutation_data, int64_t *keys_out, int64_t *indices_in,
                                                     const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseReorder<uint16_t>(const int num_elems, const int num_dims, const int64_t *indices,
                                                      const uint16_t *values, const int64_t *shape, int64_t *y_indices,
                                                      uint16_t *y_values, int64_t *flat_indices,
                                                      int64_t *permutation_data, int64_t *keys_out, int64_t *indices_in,
                                                      const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseReorder<half>(const int num_elems, const int num_dims, const int64_t *indices,
                                                  const half *values, const int64_t *shape, int64_t *y_indices,
                                                  half *y_values, int64_t *flat_indices, int64_t *permutation_data,
                                                  int64_t *keys_out, int64_t *indices_in, const uint32_t &device_id,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseReorder<float>(const int num_elems, const int num_dims, const int64_t *indices,
                                                   const float *values, const int64_t *shape, int64_t *y_indices,
                                                   float *y_values, int64_t *flat_indices, int64_t *permutation_data,
                                                   int64_t *keys_out, int64_t *indices_in, const uint32_t &device_id,
                                                   cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseReorder<double>(const int num_elems, const int num_dims, const int64_t *indices,
                                                    const double *values, const int64_t *shape, int64_t *y_indices,
                                                    double *y_values, int64_t *flat_indices, int64_t *permutation_data,
                                                    int64_t *keys_out, int64_t *indices_in, const uint32_t &device_id,
                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseReorder<cuFloatComplex>(
  const int num_elems, const int num_dims, const int64_t *indices, const cuFloatComplex *values, const int64_t *shape,
  int64_t *y_indices, cuFloatComplex *y_values, int64_t *flat_indices, int64_t *permutation_data, int64_t *keys_out,
  int64_t *indices_in, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseReorder<cuDoubleComplex>(
  const int num_elems, const int num_dims, const int64_t *indices, const cuDoubleComplex *values, const int64_t *shape,
  int64_t *y_indices, cuDoubleComplex *y_values, int64_t *flat_indices, int64_t *permutation_data, int64_t *keys_out,
  int64_t *indices_in, const uint32_t &device_id, cudaStream_t cuda_stream);
