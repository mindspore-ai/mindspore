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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/masked_select_impl.cuh"
#include <cub/cub.cuh>
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/broadcast_to_impl.cuh"

template <typename T>
using Complex = mindspore::utils::Complex<T>;

struct BoolToSize {
  typedef size_t index_type;

  __device__ index_type operator()(bool x) const { return x ? 1 : 0; }
};

__device__ __forceinline__ size_t Index(const size_t &index, const size_t &dim) { return dim == 1 ? 0 : index; }

template <typename T>
__global__ void MaskedSelectKernel(const T *input_ptr, const size_t *index_ptr, T *output_ptr, size_t index_size) {
  // e.g., [0, 0, 1, 2, 2], the fill_index is 0 and 1, the output_ptr[0] = input_ptr[2], output_ptr[1] = input_ptr[3]
  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < index_size; tid += blockDim.x * gridDim.x) {
    bool is_write = (tid != 0 && index_ptr[tid] != index_ptr[tid - 1]) || (tid == 0 && index_ptr[tid]);
    if (is_write) {
      size_t fill_index = index_ptr[tid] - 1;
      output_ptr[fill_index] = input_ptr[tid];
    }
  }
}

template <typename T>
cudaError_t MaskedSelect(T *input_ptr, bool *mask_ptr, size_t *index_ptr, const std::vector<int64_t> input_shape,
                         const std::vector<int64_t> mask_shape, const std::vector<int64_t> broadcast_shape,
                         T *input_broadcast_ptr, bool *mask_broadcast_ptr, T *output_ptr, size_t device_id,
                         cudaStream_t cuda_stream) {
  const T *last_input = nullptr;
  const bool *last_mask = nullptr;
  size_t dim_size = broadcast_shape.size();
  UnaryBroadcastStrideInfo input_strides = UnaryBroadcastCalStride(dim_size, input_shape, broadcast_shape);
  UnaryBroadcastStrideInfo mask_strides = UnaryBroadcastCalStride(dim_size, mask_shape, broadcast_shape);
  size_t output_num = 1;
  for (auto val : broadcast_shape) {
    output_num *= val;
  }
  size_t thread_num = output_num > 1024 ? 1024 : output_num;

  if (input_broadcast_ptr != nullptr) {
    BroadcastToCpyCuda<<<CUDA_BLOCKS_CAL(device_id, output_num, thread_num), thread_num, 0, cuda_stream>>>(
      dim_size, output_num, input_strides, input_ptr, input_broadcast_ptr);
    last_input = input_broadcast_ptr;
  } else {
    last_input = input_ptr;
  }

  if (mask_broadcast_ptr != nullptr) {
    BroadcastToCpyCuda<<<CUDA_BLOCKS_CAL(device_id, output_num, thread_num), thread_num, 0, cuda_stream>>>(
      dim_size, output_num, mask_strides, mask_ptr, mask_broadcast_ptr);
    last_mask = mask_broadcast_ptr;
  } else {
    last_mask = mask_ptr;
  }

  // using cub to calculate prefix sum of 01 transformed sequence
  BoolToSize op;
  cub::TransformInputIterator<size_t, BoolToSize, const bool *> iter(last_mask, op);
  size_t temp_storage_bytes = 0;
  (void)cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, iter, index_ptr, output_num, cuda_stream);
  void *d_temp_storage = nullptr;
  (void)cudaMalloc(&d_temp_storage, temp_storage_bytes);
  (void)cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, iter, index_ptr, output_num, cuda_stream);

  // Extract the first index to appear and transform into output index
  MaskedSelectKernel<<<GET_BLOCKS(output_num), GET_THREADS, 0, cuda_stream>>>(last_input, index_ptr, output_ptr,
                                                                              output_num);
  (void)cudaFree(d_temp_storage);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t MaskedSelect<uint8_t>(
  uint8_t *input_ptr, bool *mask_ptr, size_t *index_ptr, const std::vector<int64_t> input_shape,
  const std::vector<int64_t> mask_shape, const std::vector<int64_t> broadcast_shape, uint8_t *input_broadcast_ptr,
  bool *mask_broadcast_ptr, uint8_t *output_ptr, size_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MaskedSelect<uint16_t>(
  uint16_t *input_ptr, bool *mask_ptr, size_t *index_ptr, const std::vector<int64_t> input_shape,
  const std::vector<int64_t> mask_shape, const std::vector<int64_t> broadcast_shape, uint16_t *input_broadcast_ptr,
  bool *mask_broadcast_ptr, uint16_t *output_ptr, size_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MaskedSelect<uint32_t>(
  uint32_t *input_ptr, bool *mask_ptr, size_t *index_ptr, const std::vector<int64_t> input_shape,
  const std::vector<int64_t> mask_shape, const std::vector<int64_t> broadcast_shape, uint32_t *input_broadcast_ptr,
  bool *mask_broadcast_ptr, uint32_t *output_ptr, size_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MaskedSelect<uint64_t>(
  uint64_t *input_ptr, bool *mask_ptr, size_t *index_ptr, const std::vector<int64_t> input_shape,
  const std::vector<int64_t> mask_shape, const std::vector<int64_t> broadcast_shape, uint64_t *input_broadcast_ptr,
  bool *mask_broadcast_ptr, uint64_t *output_ptr, size_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MaskedSelect<int8_t>(
  int8_t *input_ptr, bool *mask_ptr, size_t *index_ptr, const std::vector<int64_t> input_shape,
  const std::vector<int64_t> mask_shape, const std::vector<int64_t> broadcast_shape, int8_t *input_broadcast_ptr,
  bool *mask_broadcast_ptr, int8_t *output_ptr, size_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MaskedSelect<int16_t>(
  int16_t *input_ptr, bool *mask_ptr, size_t *index_ptr, const std::vector<int64_t> input_shape,
  const std::vector<int64_t> mask_shape, const std::vector<int64_t> broadcast_shape, int16_t *input_broadcast_ptr,
  bool *mask_broadcast_ptr, int16_t *output_ptr, size_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MaskedSelect<int32_t>(
  int32_t *input_ptr, bool *mask_ptr, size_t *index_ptr, const std::vector<int64_t> input_shape,
  const std::vector<int64_t> mask_shape, const std::vector<int64_t> broadcast_shape, int32_t *input_broadcast_ptr,
  bool *mask_broadcast_ptr, int32_t *output_ptr, size_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MaskedSelect<int64_t>(
  int64_t *input_ptr, bool *mask_ptr, size_t *index_ptr, const std::vector<int64_t> input_shape,
  const std::vector<int64_t> mask_shape, const std::vector<int64_t> broadcast_shape, int64_t *input_broadcast_ptr,
  bool *mask_broadcast_ptr, int64_t *output_ptr, size_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MaskedSelect<half>(half *input_ptr, bool *mask_ptr, size_t *index_ptr,
                                                        const std::vector<int64_t> input_shape,
                                                        const std::vector<int64_t> mask_shape,
                                                        const std::vector<int64_t> broadcast_shape,
                                                        half *input_broadcast_ptr, bool *mask_broadcast_ptr,
                                                        half *output_ptr, size_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MaskedSelect<float>(float *input_ptr, bool *mask_ptr, size_t *index_ptr,
                                                         const std::vector<int64_t> input_shape,
                                                         const std::vector<int64_t> mask_shape,
                                                         const std::vector<int64_t> broadcast_shape,
                                                         float *input_broadcast_ptr, bool *mask_broadcast_ptr,
                                                         float *output_ptr, size_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MaskedSelect<double>(
  double *input_ptr, bool *mask_ptr, size_t *index_ptr, const std::vector<int64_t> input_shape,
  const std::vector<int64_t> mask_shape, const std::vector<int64_t> broadcast_shape, double *input_broadcast_ptr,
  bool *mask_broadcast_ptr, double *output_ptr, size_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MaskedSelect<bool>(bool *input_ptr, bool *mask_ptr, size_t *index_ptr,
                                                        const std::vector<int64_t> input_shape,
                                                        const std::vector<int64_t> mask_shape,
                                                        const std::vector<int64_t> broadcast_shape,
                                                        bool *input_broadcast_ptr, bool *mask_broadcast_ptr,
                                                        bool *output_ptr, size_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MaskedSelect<Complex<float>>(
  Complex<float> *input_ptr, bool *mask_ptr, size_t *index_ptr, const std::vector<int64_t> input_shape,
  const std::vector<int64_t> mask_shape, const std::vector<int64_t> broadcast_shape,
  Complex<float> *input_broadcast_ptr, bool *mask_broadcast_ptr, Complex<float> *output_ptr, size_t device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MaskedSelect<Complex<double>>(
  Complex<double> *input_ptr, bool *mask_ptr, size_t *index_ptr, const std::vector<int64_t> input_shape,
  const std::vector<int64_t> mask_shape, const std::vector<int64_t> broadcast_shape,
  Complex<double> *input_broadcast_ptr, bool *mask_broadcast_ptr, Complex<double> *output_ptr, size_t device_id,
  cudaStream_t cuda_stream);
