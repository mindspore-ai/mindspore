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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/masked_select_grad_impl.cuh"
#include <cub/cub.cuh>
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/broadcast_to_impl.cuh"

struct BoolToSize {
  typedef size_t index_type;

  __device__ index_type operator()(bool x) const { return x ? 1 : 0; }
};

__device__ __forceinline__ size_t Index(const size_t &index, const size_t &dim) { return dim == 1 ? 0 : index; }

// BroadcastTo
template <typename T>
__global__ void GetResultKernel(size_t dim_size, size_t output_num, UnaryBroadcastStrideInfo strides,
                                T *input_grad_addr, T *broadcasted_input_grad_addr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < output_num; pos += blockDim.x * gridDim.x) {
    int64_t cur_out_idx = 0;
    size_t cur_pos = pos;
    size_t inp_pos = 0;
    for (int idx = 0; idx < dim_size; ++idx) {
      cur_out_idx = cur_pos / strides.output_stride[idx];
      inp_pos += cur_out_idx * strides.input_stride[idx];
      cur_pos -= cur_out_idx * strides.output_stride[idx];
    }
    MsAtomicAdd(input_grad_addr + inp_pos, broadcasted_input_grad_addr[pos]);
  }
}

template <typename T>
__global__ void MaskedSelectGradKernel(T *broadcasted_input_grad_ptr, const size_t *index_ptr, T *output_grad_ptr,
                                       size_t index_size) {
  // e.g., [0, 0, 1, 2, 2], the fill_index is 0 and 1, the output_ptr[0] = input_ptr[2], output_ptr[1] = input_ptr[3]
  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < index_size; tid += blockDim.x * gridDim.x) {
    bool is_write = (tid != 0 && index_ptr[tid] != index_ptr[tid - 1]) || (tid == 0 && index_ptr[tid]);
    if (is_write) {
      size_t fill_index = index_ptr[tid] - 1;
      broadcasted_input_grad_ptr[tid] = output_grad_ptr[fill_index];
    }
  }
}

template <typename T>
cudaError_t MaskedSelectGrad(T *input_grad_ptr, bool *mask_ptr, size_t *index_ptr,
                             const std::vector<int64_t> input_shape, const std::vector<int64_t> mask_shape,
                             const std::vector<int64_t> broadcast_shape, T *input_broadcast_grad_ptr,
                             bool *mask_broadcast_ptr, T *output_grad_ptr, size_t device_id, cudaStream_t cuda_stream) {
  const bool *last_mask = nullptr;
  size_t dim_size = broadcast_shape.size();
  UnaryBroadcastStrideInfo input_strides = UnaryBroadcastCalStride(dim_size, input_shape, broadcast_shape);
  UnaryBroadcastStrideInfo mask_strides = UnaryBroadcastCalStride(dim_size, mask_shape, broadcast_shape);
  size_t output_num = 1;
  for (auto val : broadcast_shape) {
    output_num *= val;
  }

  // broadcast the mask_ptr
  if (mask_broadcast_ptr != nullptr) {
    size_t thread_num = output_num > 1024 ? 1024 : output_num;
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
  if (input_broadcast_grad_ptr != nullptr) {
    MaskedSelectGradKernel<<<GET_BLOCKS(output_num), GET_THREADS, 0, cuda_stream>>>(input_broadcast_grad_ptr, index_ptr,
                                                                                    output_grad_ptr, output_num);
    GetResultKernel<<<GET_BLOCKS(output_num), GET_THREADS, 0, cuda_stream>>>(dim_size, output_num, input_strides,
                                                                             input_grad_ptr, input_broadcast_grad_ptr);
  } else {
    MaskedSelectGradKernel<<<GET_BLOCKS(output_num), GET_THREADS, 0, cuda_stream>>>(input_grad_ptr, index_ptr,
                                                                                    output_grad_ptr, output_num);
  }
  (void)cudaFree(d_temp_storage);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t MaskedSelectGrad<uint8_t>(
  uint8_t *input_grad_ptr, bool *mask_ptr, size_t *index_ptr, const std::vector<int64_t> input_shape,
  const std::vector<int64_t> mask_shape, const std::vector<int64_t> broadcast_shape, uint8_t *input_broadcast_ptr,
  bool *mask_broadcast_ptr, uint8_t *output_ptr, size_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MaskedSelectGrad<uint16_t>(
  uint16_t *input_grad_ptr, bool *mask_ptr, size_t *index_ptr, const std::vector<int64_t> input_shape,
  const std::vector<int64_t> mask_shape, const std::vector<int64_t> broadcast_shape, uint16_t *input_broadcast_ptr,
  bool *mask_broadcast_ptr, uint16_t *output_ptr, size_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MaskedSelectGrad<uint32_t>(
  uint32_t *input_grad_ptr, bool *mask_ptr, size_t *index_ptr, const std::vector<int64_t> input_shape,
  const std::vector<int64_t> mask_shape, const std::vector<int64_t> broadcast_shape, uint32_t *input_broadcast_ptr,
  bool *mask_broadcast_ptr, uint32_t *output_ptr, size_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MaskedSelectGrad<uint64_t>(
  uint64_t *input_grad_ptr, bool *mask_ptr, size_t *index_ptr, const std::vector<int64_t> input_shape,
  const std::vector<int64_t> mask_shape, const std::vector<int64_t> broadcast_shape, uint64_t *input_broadcast_ptr,
  bool *mask_broadcast_ptr, uint64_t *output_ptr, size_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MaskedSelectGrad<int8_t>(
  int8_t *input_grad_ptr, bool *mask_ptr, size_t *index_ptr, const std::vector<int64_t> input_shape,
  const std::vector<int64_t> mask_shape, const std::vector<int64_t> broadcast_shape, int8_t *input_broadcast_ptr,
  bool *mask_broadcast_ptr, int8_t *output_ptr, size_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MaskedSelectGrad<int16_t>(
  int16_t *input_grad_ptr, bool *mask_ptr, size_t *index_ptr, const std::vector<int64_t> input_shape,
  const std::vector<int64_t> mask_shape, const std::vector<int64_t> broadcast_shape, int16_t *input_broadcast_ptr,
  bool *mask_broadcast_ptr, int16_t *output_ptr, size_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MaskedSelectGrad<int32_t>(
  int32_t *input_grad_ptr, bool *mask_ptr, size_t *index_ptr, const std::vector<int64_t> input_shape,
  const std::vector<int64_t> mask_shape, const std::vector<int64_t> broadcast_shape, int32_t *input_broadcast_ptr,
  bool *mask_broadcast_ptr, int32_t *output_ptr, size_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MaskedSelectGrad<int64_t>(
  int64_t *input_grad_ptr, bool *mask_ptr, size_t *index_ptr, const std::vector<int64_t> input_shape,
  const std::vector<int64_t> mask_shape, const std::vector<int64_t> broadcast_shape, int64_t *input_broadcast_ptr,
  bool *mask_broadcast_ptr, int64_t *output_ptr, size_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MaskedSelectGrad<half>(
  half *input_grad_ptr, bool *mask_ptr, size_t *index_ptr, const std::vector<int64_t> input_shape,
  const std::vector<int64_t> mask_shape, const std::vector<int64_t> broadcast_shape, half *input_broadcast_ptr,
  bool *mask_broadcast_ptr, half *output_ptr, size_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MaskedSelectGrad<float>(
  float *input_grad_ptr, bool *mask_ptr, size_t *index_ptr, const std::vector<int64_t> input_shape,
  const std::vector<int64_t> mask_shape, const std::vector<int64_t> broadcast_shape, float *input_broadcast_ptr,
  bool *mask_broadcast_ptr, float *output_ptr, size_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t MaskedSelectGrad<double>(
  double *input_grad_ptr, bool *mask_ptr, size_t *index_ptr, const std::vector<int64_t> input_shape,
  const std::vector<int64_t> mask_shape, const std::vector<int64_t> broadcast_shape, double *input_broadcast_ptr,
  bool *mask_broadcast_ptr, double *output_ptr, size_t device_id, cudaStream_t cuda_stream);
