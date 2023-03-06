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

#include "inplace_op_impl.cuh"
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T>
struct SubFunc {
  __device__ __forceinline__ void operator()(T *lhs, const T &rhs) { MsAtomicSub(lhs, rhs); }
};

template <typename T>
struct AddFunc {
  __device__ __forceinline__ void operator()(T *lhs, const T &rhs) { MsAtomicAdd(lhs, rhs); }
};

template <typename T, typename S>
__global__ void InplaceUpdate(const size_t size, const T *input_v, T *output, const S *indices, S *indices_key,
                              size_t indices_len, const int64_t band_size) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    size_t row = pos / band_size;
    if (row == indices_len || indices[row] != indices[row + 1]) {
      S x_row = indices[row];
      S v_row = indices_key[row];
      int offset = pos % band_size;
      int x_offset = x_row * band_size;
      output[x_offset + offset] = input_v[v_row * band_size + offset];
    }
  }
  return;
}
template <typename T, typename S, typename Func>
__global__ void InplaceAddOrSub(const size_t size, const T *input_v, T *output, const S *indices,
                                const int64_t band_size) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int v_row = pos / band_size;
    S x_row = indices[v_row];
    int offset = pos % band_size;
    int x_offset = x_row * band_size;
    Func()(&output[x_offset + offset], input_v[pos]);
  }
  return;
}

template <typename T, typename S>
void CalInplaceOp(const size_t size_v, const T *input_v, T *output, S *indices, S *indices_key, const int64_t band_size,
                  const uint32_t &device_id, int op_type, cudaStream_t cuda_stream) {
  int thread_num = 256 > size_v ? size_v : 256;
  if (op_type == INPLACE_OP_TYPE_UPDATE) {
    auto policy = thrust::cuda::par.on(cuda_stream);
    size_t indices_element = size_v / band_size;
    thrust::sequence(policy, thrust::device_pointer_cast(indices_key),
                     thrust::device_pointer_cast(indices_key) + indices_element);
    thrust::stable_sort_by_key(policy, thrust::device_pointer_cast(indices),
                               thrust::device_pointer_cast(indices) + indices_element,
                               thrust::device_pointer_cast(indices_key));
    InplaceUpdate<<<CUDA_BLOCKS_CAL(device_id, size_v, thread_num), thread_num, 0, cuda_stream>>>(
      size_v, input_v, output, indices, indices_key, indices_element, band_size);
  } else if (op_type == INPLACE_OP_TYPE_ADD) {
    InplaceAddOrSub<T, S, AddFunc<T>><<<CUDA_BLOCKS_CAL(device_id, size_v, thread_num), thread_num, 0, cuda_stream>>>(
      size_v, input_v, output, indices, band_size);
  } else if (op_type == INPLACE_OP_TYPE_SUB) {
    InplaceAddOrSub<T, S, SubFunc<T>><<<CUDA_BLOCKS_CAL(device_id, size_v, thread_num), thread_num, 0, cuda_stream>>>(
      size_v, input_v, output, indices, band_size);
  }
  return;
}

template CUDA_LIB_EXPORT void CalInplaceOp<half>(const size_t size_v, const half *input_v, half *output,
                                                 int64_t *indices, int64_t *indices_key, const int64_t band_size,
                                                 const uint32_t &device_id, int op_type, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalInplaceOp<float>(const size_t size_v, const float *input_v, float *output,
                                                  int64_t *indices, int64_t *indices_key, const int64_t band_size,
                                                  const uint32_t &device_id, int op_type, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalInplaceOp<double>(const size_t size_v, const double *input_v, double *output,
                                                   int64_t *indices, int64_t *indices_key, const int64_t band_size,
                                                   const uint32_t &device_id, int op_type, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalInplaceOp<int>(const size_t size_v, const int *input_v, int *output, int64_t *indices,
                                                int64_t *indices_key, const int64_t band_size,
                                                const uint32_t &device_id, int op_type, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalInplaceOp<half>(const size_t size_v, const half *input_v, half *output,
                                                 int32_t *indices, int32_t *indices_key, const int64_t band_size,
                                                 const uint32_t &device_id, int op_type, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalInplaceOp<float>(const size_t size_v, const float *input_v, float *output,
                                                  int32_t *indices, int32_t *indices_key, const int64_t band_size,
                                                  const uint32_t &device_id, int op_type, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalInplaceOp<double>(const size_t size_v, const double *input_v, double *output,
                                                   int32_t *indices, int32_t *indices_key, const int64_t band_size,
                                                   const uint32_t &device_id, int op_type, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalInplaceOp<int>(const size_t size_v, const int *input_v, int *output, int32_t *indices,
                                                int32_t *indices_key, const int64_t band_size,
                                                const uint32_t &device_id, int op_type, cudaStream_t cuda_stream);
