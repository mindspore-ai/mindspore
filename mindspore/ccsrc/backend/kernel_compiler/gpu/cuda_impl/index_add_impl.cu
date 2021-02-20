/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/gpu/cuda_impl/index_add_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/util.cuh"
#include "runtime/device/gpu/cuda_common.h"
#include "include/cuda_fp16.h"
__global__ void InitErrorCode(IndexAddErrorCode *error_code) {
  *error_code = IndexAddErrorCode::kOk;
}

__global__ void ValidateIndexValues(const int *index, const size_t src_axis_size, const size_t dst_axis_size,
  IndexAddErrorCode *error_code) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < src_axis_size; pos += blockDim.x * gridDim.x) {
    const int idx_value = index[pos];
    if (idx_value < 0 || idx_value >= dst_axis_size) {
      *error_code = IndexAddErrorCode::kIndexOutOfRange;
      return;
    }
  }
  return;
}

template <typename T>
__global__ void IndexAddAtomic(T *dst, const int *index, const T *src, const size_t src_size, const size_t outer_size,
  const size_t src_axis_size, const size_t dst_axis_size, const size_t inner_size) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < src_size; pos += blockDim.x * gridDim.x) {
    const size_t src_axis_idx = (pos / inner_size) % src_axis_size;
    const size_t src_outer_idx = pos / (src_axis_size * inner_size);
    const size_t dst_axis_idx = static_cast<size_t>(index[src_axis_idx]);
    const size_t dst_inner_idx = pos % inner_size;
    const size_t dst_idx = src_outer_idx * (dst_axis_size * inner_size) + dst_axis_idx * inner_size + dst_inner_idx;
    MsAtomicAdd(&dst[dst_idx], src[pos]);
  }
  return;
}

template <typename T>
__global__ void IndexAdd(T *dst, const int *index, const T *src, const size_t src_size, const size_t outer_size,
  const size_t src_axis_size, const size_t dst_axis_size, const size_t inner_size) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < src_size; pos += blockDim.x * gridDim.x) {
    const size_t src_axis_idx = (pos / inner_size) % src_axis_size;
    const size_t src_outer_idx = pos / (src_axis_size * inner_size);
    const size_t dst_axis_idx = static_cast<size_t>(index[src_axis_idx]);
    const size_t dst_inner_idx = pos % inner_size;
    const size_t dst_idx = src_outer_idx * (dst_axis_size * inner_size) + dst_axis_idx * inner_size + dst_inner_idx;
    dst[dst_idx] += src[pos];
  }
  return;
}

void ValidateIndexAddInputValues(const int *index, const size_t src_axis_size, const size_t dst_axis_size,
  IndexAddErrorCode *error_code, cudaStream_t cuda_stream) {
  InitErrorCode<<<1, 1, 0, cuda_stream>>>(error_code);
  ValidateIndexValues<<<GET_BLOCKS(src_axis_size), GET_THREADS, 0, cuda_stream>>>(index, src_axis_size, dst_axis_size,
    error_code);
}

template <typename T>
void CalIndexAdd(T *dst, const int *index, const T *src, const size_t outer_size, const size_t src_axis_size,
  const size_t dst_axis_size, const size_t inner_size, const bool use_lock, cudaStream_t cuda_stream) {
  size_t src_size = outer_size * src_axis_size * inner_size;
  if (use_lock) {
    IndexAddAtomic<<<GET_BLOCKS(src_size), GET_THREADS, 0, cuda_stream>>>(dst, index, src, src_size, outer_size,
      src_axis_size, dst_axis_size, inner_size);
  } else {
    IndexAdd<<<GET_BLOCKS(src_size), GET_THREADS, 0, cuda_stream>>>(dst, index, src, src_size, outer_size,
      src_axis_size, dst_axis_size, inner_size);
  }
  return;
}

template void CalIndexAdd<double>(double *dst, const int *index, const double *src, const size_t outer_size,
  const size_t src_axis_size, const size_t dst_axis_size, const size_t inner_size, const bool use_lock,
  cudaStream_t cuda_stream);
template void CalIndexAdd<float>(float *dst, const int *index, const float *src, const size_t outer_size,
  const size_t src_axis_size, const size_t dst_axis_size, const size_t inner_size, const bool use_lock,
  cudaStream_t cuda_stream);
template void CalIndexAdd<half>(half *dst, const int *index, const half *src, const size_t outer_size,
  const size_t src_axis_size, const size_t dst_axis_size, const size_t inner_size, const bool use_lock,
  cudaStream_t cuda_stream);
template void CalIndexAdd<int>(int *dst, const int *index, const int *src, const size_t outer_size,
  const size_t src_axis_size, const size_t dst_axis_size, const size_t inner_size, const bool use_lock,
  cudaStream_t cuda_stream);
template void CalIndexAdd<int16_t>(int16_t *dst, const int *index, const int16_t *src, const size_t outer_size,
  const size_t src_axis_size, const size_t dst_axis_size, const size_t inner_size, const bool use_lock,
  cudaStream_t cuda_stream);
template void CalIndexAdd<int8_t>(int8_t *dst, const int *index, const int8_t *src, const size_t outer_size,
  const size_t src_axis_size, const size_t dst_axis_size, const size_t inner_size, const bool use_lock,
  cudaStream_t cuda_stream);
template void CalIndexAdd<uint8_t>(uint8_t *dst, const int *index, const uint8_t *src, const size_t outer_size,
  const size_t src_axis_size, const size_t dst_axis_size, const size_t inner_size, const bool use_lock,
  cudaStream_t cuda_stream);
