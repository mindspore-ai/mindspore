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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sparse_apply_adagrad_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__device__ __forceinline__ T RsqrtFunc(T x) {
  return __frsqrt_rn(x);
}

template <>
__device__ __forceinline__ half RsqrtFunc(half x) {
  return hrsqrt(x);
}

template <typename T, typename S>
__global__ void SparseApplyAdagradUpdate(const size_t size, const size_t indices_size, const float learning_rate,
                                         const bool update_slots, const T *gradient, const S *indices, T *variable,
                                         T *accumulation, T *variable_out, T *accumulation_out) {
  const int inner_size = static_cast<int>(size / indices_size);
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < static_cast<int>(size); pos += gridDim.x * blockDim.x) {
    const int index = pos / inner_size;
    const int inner_pos = pos % inner_size;
    const int grad_pos = pos;
    const int cur_pos = indices[index] * inner_size + inner_pos;
    if (update_slots) {
      accumulation[cur_pos] += gradient[grad_pos] * gradient[grad_pos];
    }
    const T scratch1 = static_cast<T>(learning_rate) * RsqrtFunc(accumulation[cur_pos]);
    variable[cur_pos] -= gradient[grad_pos] * scratch1;
    accumulation_out[cur_pos] = accumulation[cur_pos];
    variable_out[cur_pos] = variable[cur_pos];
  }
}

template <typename T, typename S>
void CalSparseApplyAdagrad(const size_t size, const size_t indices_size, const float learning_rate,
                           const bool update_slots, const T *gradient, const S *indices, T *variable, T *accumulation,
                           T *variable_out, T *accumulation_out, cudaStream_t cuda_stream) {
  SparseApplyAdagradUpdate<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(
    size, indices_size, learning_rate, update_slots, gradient, indices, variable, accumulation, variable_out,
    accumulation_out);
}

template CUDA_LIB_EXPORT void CalSparseApplyAdagrad<float, int>(const size_t size, const size_t indices_size,
                                                                const float learning_rate, const bool update_slots,
                                                                const float *gradient, const int *indices,
                                                                float *variable, float *accumulation,
                                                                float *variable_out, float *accumulation_out,
                                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSparseApplyAdagrad<half, int>(const size_t size, const size_t indices_size,
                                                               const float learning_rate, const bool update_slots,
                                                               const half *gradient, const int *indices, half *variable,
                                                               half *accumulation, half *variable_out,
                                                               half *accumulation_out, cudaStream_t cuda_stream);
