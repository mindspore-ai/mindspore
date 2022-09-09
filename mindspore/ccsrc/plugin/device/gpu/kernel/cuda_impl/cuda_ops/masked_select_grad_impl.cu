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

struct BoolToSize {
  typedef size_t index_type;

  __device__ index_type operator()(bool x) const { return x ? 1 : 0; }
};

__device__ __forceinline__ size_t Index(const size_t &index, const size_t &dim) { return dim == 1 ? 0 : index; }


// BroadcastTo
template <typename T>
__global__ void BroadcastToKernel(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5, size_t i6,
                                  size_t o0, size_t o1, size_t o2, size_t o3, size_t o4, size_t o5, size_t o6,
                                  const T *input_addr, T *output_addr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < o0 * o1 * o2 * o3 * o4 * o5 * o6;
       pos += blockDim.x * gridDim.x) {
    size_t i = pos / (o1 * o2 * o3 * o4 * o5 * o6) % o0;
    size_t j = pos / (o2 * o3 * o4 * o5 * o6) % o1;
    size_t k = pos / (o3 * o4 * o5 * o6) % o2;
    size_t l = pos / (o4 * o5 * o6) % o3;
    size_t m = pos / (o5 * o6) % o4;
    size_t n = pos / o6 % o5;
    size_t o = pos % o6;

    size_t input_idx = Index(i, i0) * i1 * i2 * i3 * i4 * i5 * i6;
    input_idx += Index(j, i1) * i2 * i3 * i4 * i5 * i6;
    input_idx += Index(k, i2) * i3 * i4 * i5 * i6;
    input_idx += Index(l, i3) * i4 * i5 * i6;
    input_idx += Index(m, i4) * i5 * i6;
    input_idx += Index(n, i5) * i6;
    input_idx += Index(o, i6);

    output_addr[pos] = input_addr[input_idx];
  }
}

// BroadcastTo
template <typename T>
__global__ void GetResult(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5, size_t i6,
                                  size_t o0, size_t o1, size_t o2, size_t o3, size_t o4, size_t o5, size_t o6,
                                  T *input_grad_addr, T *broadcasted_input_grad_addr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < o0 * o1 * o2 * o3 * o4 * o5 * o6;
       pos += blockDim.x * gridDim.x) {
    size_t i = pos / (o1 * o2 * o3 * o4 * o5 * o6) % o0;
    size_t j = pos / (o2 * o3 * o4 * o5 * o6) % o1;
    size_t k = pos / (o3 * o4 * o5 * o6) % o2;
    size_t l = pos / (o4 * o5 * o6) % o3;
    size_t m = pos / (o5 * o6) % o4;
    size_t n = pos / o6 % o5;
    size_t o = pos % o6;

    size_t input_idx = Index(i, i0) * i1 * i2 * i3 * i4 * i5 * i6;
    input_idx += Index(j, i1) * i2 * i3 * i4 * i5 * i6;
    input_idx += Index(k, i2) * i3 * i4 * i5 * i6;
    input_idx += Index(l, i3) * i4 * i5 * i6;
    input_idx += Index(m, i4) * i5 * i6;
    input_idx += Index(n, i5) * i6;
    input_idx += Index(o, i6);

    MsAtomicAdd(input_grad_addr + input_idx, broadcasted_input_grad_addr[pos]);
  }
}

template <typename T>
__global__ void MaskedSelectGradKernel(T *broadcasted_input_grad_ptr,
                                       const size_t *index_ptr, T *output_grad_ptr, size_t index_size) {
  // e.g., [0, 0, 1, 2, 2], the fill_index is 0 and 1, the output_ptr[0] = input_ptr[2], output_ptr[1] = input_ptr[3]
  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < index_size; tid += blockDim.x * gridDim.x) {
    bool is_write = (tid != 0 && index_ptr[tid] != index_ptr[tid - 1]) || (tid == 0 && index_ptr[tid]);
    if (is_write) {
      size_t fill_index = index_ptr[tid] - 1;
      broadcasted_input_grad_ptr[tid] = output_grad_ptr[fill_index];
    }
  }
}

// the i is input shape, the j is mask shape, the o is broadcast shape
template <typename T>
void MaskedSelectGrad(T *input_grad_ptr, const bool *mask_ptr, size_t *index_ptr,
                      const std::vector<size_t> i, const std::vector<size_t> j, const std::vector<size_t> o,
                      T *input_broadcast_grad_ptr, bool *mask_broadcast_ptr,
                      T *output_grad_ptr, cudaStream_t cuda_stream) {
  size_t broadcast_size = o[0] * o[1] * o[2] * o[3] * o[4] * o[5] * o[6];
  const bool *last_mask = nullptr;

  if (mask_broadcast_ptr != nullptr) {
    BroadcastToKernel<<<GET_BLOCKS(broadcast_size), GET_THREADS, 0, cuda_stream>>>(
      j[0], j[1], j[2], j[3], j[4], j[5], j[6], o[0], o[1], o[2], o[3], o[4], o[5], o[6], mask_ptr,
      mask_broadcast_ptr);
    last_mask = mask_broadcast_ptr;
  } else {
    last_mask = mask_ptr;
  }

  // using cub to calculate prefix sum of 01 transformed sequence
  BoolToSize op;
  cub::TransformInputIterator<size_t, BoolToSize, const bool*> iter(last_mask, op);
  size_t temp_storage_bytes = 0;
  (void)cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, iter, index_ptr, broadcast_size, cuda_stream);
  void *d_temp_storage = nullptr;
  (void)cudaMalloc(&d_temp_storage, temp_storage_bytes);
  (void)cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, iter, index_ptr, broadcast_size, cuda_stream);

  // Extract the first index to appear and transform into output index
  if (input_broadcast_grad_ptr != nullptr) {
    MaskedSelectGradKernel<<<GET_BLOCKS(broadcast_size), GET_THREADS, 0, cuda_stream>>>(input_broadcast_grad_ptr,
                                                                                        index_ptr, output_grad_ptr,
                                                                                        broadcast_size);
    GetResult<<<GET_BLOCKS(broadcast_size), GET_THREADS, 0, cuda_stream>>>(
      i[0], i[1], i[2], i[3], i[4], i[5], i[6], o[0], o[1], o[2], o[3], o[4], o[5], o[6], input_grad_ptr,
      input_broadcast_grad_ptr);
  } else {
    MaskedSelectGradKernel<<<GET_BLOCKS(broadcast_size), GET_THREADS, 0, cuda_stream>>>(input_grad_ptr,
                                                                                        index_ptr, output_grad_ptr,
                                                                                        broadcast_size);
  }
  (void)cudaFree(d_temp_storage);
}

template CUDA_LIB_EXPORT void MaskedSelectGrad<uint8_t>(uint8_t *input_grad_ptr,
                                                    const bool *mask_ptr, size_t *index_ptr,
                                                    const std::vector<size_t> input_shape,
                                                    const std::vector<size_t> mask_shape,
                                                    const std::vector<size_t> broadcast_shape,
                                                    uint8_t *input_broadcast_ptr, bool *mask_broadcast_ptr,
                                                    uint8_t *output_ptr, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MaskedSelectGrad<uint16_t>(uint16_t *input_grad_ptr,
                                                     const bool *mask_ptr, size_t *index_ptr,
                                                     const std::vector<size_t> input_shape,
                                                     const std::vector<size_t> mask_shape,
                                                     const std::vector<size_t> broadcast_shape,
                                                     uint16_t *input_broadcast_ptr, bool *mask_broadcast_ptr,
                                                     uint16_t *output_ptr, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MaskedSelectGrad<uint32_t>(uint32_t *input_grad_ptr,
                                                     const bool *mask_ptr, size_t *index_ptr,
                                                     const std::vector<size_t> input_shape,
                                                     const std::vector<size_t> mask_shape,
                                                     const std::vector<size_t> broadcast_shape,
                                                     uint32_t *input_broadcast_ptr, bool *mask_broadcast_ptr,
                                                     uint32_t *output_ptr, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MaskedSelectGrad<uint64_t>(uint64_t *input_grad_ptr,
                                                     const bool *mask_ptr, size_t *index_ptr,
                                                     const std::vector<size_t> input_shape,
                                                     const std::vector<size_t> mask_shape,
                                                     const std::vector<size_t> broadcast_shape,
                                                     uint64_t *input_broadcast_ptr, bool *mask_broadcast_ptr,
                                                     uint64_t *output_ptr, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MaskedSelectGrad<int8_t>(int8_t *input_grad_ptr,
                                                   const bool *mask_ptr, size_t *index_ptr,
                                                   const std::vector<size_t> input_shape,
                                                   const std::vector<size_t> mask_shape,
                                                   const std::vector<size_t> broadcast_shape,
                                                   int8_t *input_broadcast_ptr, bool *mask_broadcast_ptr,
                                                   int8_t *output_ptr, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MaskedSelectGrad<int16_t>(int16_t *input_grad_ptr,
                                                    const bool *mask_ptr, size_t *index_ptr,
                                                    const std::vector<size_t> input_shape,
                                                    const std::vector<size_t> mask_shape,
                                                    const std::vector<size_t> broadcast_shape,
                                                    int16_t *input_broadcast_ptr, bool *mask_broadcast_ptr,
                                                    int16_t *output_ptr, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MaskedSelectGrad<int32_t>(int32_t *input_grad_ptr,
                                                    const bool *mask_ptr, size_t *index_ptr,
                                                    const std::vector<size_t> input_shape,
                                                    const std::vector<size_t> mask_shape,
                                                    const std::vector<size_t> broadcast_shape,
                                                    int32_t *input_broadcast_ptr, bool *mask_broadcast_ptr,
                                                    int32_t *output_ptr, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MaskedSelectGrad<int64_t>(int64_t *input_grad_ptr,
                                                    const bool *mask_ptr, size_t *index_ptr,
                                                    const std::vector<size_t> input_shape,
                                                    const std::vector<size_t> mask_shape,
                                                    const std::vector<size_t> broadcast_shape,
                                                    int64_t *input_broadcast_ptr, bool *mask_broadcast_ptr,
                                                    int64_t *output_ptr, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MaskedSelectGrad<half>(half *input_grad_ptr,
                                                 const bool *mask_ptr, size_t *index_ptr,
                                                 const std::vector<size_t> input_shape,
                                                 const std::vector<size_t> mask_shape,
                                                 const std::vector<size_t> broadcast_shape, half *input_broadcast_ptr,
                                                 bool *mask_broadcast_ptr, half *output_ptr, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MaskedSelectGrad<float>(float *input_grad_ptr,
                                                  const bool *mask_ptr, size_t *index_ptr,
                                                  const std::vector<size_t> input_shape,
                                                  const std::vector<size_t> mask_shape,
                                                  const std::vector<size_t> broadcast_shape, float *input_broadcast_ptr,
                                                  bool *mask_broadcast_ptr, float *output_ptr,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void MaskedSelectGrad<double>(double *input_grad_ptr,
                                                   const bool *mask_ptr, size_t *index_ptr,
                                                   const std::vector<size_t> input_shape,
                                                   const std::vector<size_t> mask_shape,
                                                   const std::vector<size_t> broadcast_shape,
                                                   double *input_broadcast_ptr, bool *mask_broadcast_ptr,
                                                   double *output_ptr, cudaStream_t cuda_stream);
