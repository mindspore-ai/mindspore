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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sparse_apply_r_m_s_prop_impl.cuh"
#include <iostream>
#include "include/cuda_fp16.h"

template <typename T>
__device__ __forceinline__ T RsqrtFunc(T x) {
  return rsqrtf(x);
}

template <typename T>
__device__ __forceinline__ float GetFloat(T x) {
  return x;
}

template <>
__device__ __forceinline__ float GetFloat(half x) {
  return __half2float(x);
}

template <typename T, typename S>
__global__ void SparseApplyRMSPropUpdate(const size_t size, const size_t indices_size, const float decay_rate,
                                         const float momentum, const float epsilon, const T *learning_rate,
                                         const T *gradient, const S *indices, T *variable, T *ms, T *mom) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < static_cast<int>(size); pos += gridDim.x * blockDim.x) {
    const size_t cur_pos = indices[pos / indices_size] * indices_size + pos % indices_size;
    const float grad_t = GetFloat(gradient[pos]);
    float msf = GetFloat(ms[cur_pos]);
    if (grad_t != 0) {
      msf = msf * decay_rate + grad_t * grad_t * (1.0f - decay_rate);
      ms[cur_pos] = static_cast<T>(msf);
    }
    mom[cur_pos] = static_cast<T>(GetFloat(mom[cur_pos]) * momentum  + RsqrtFunc(msf + epsilon) *
                                  GetFloat(learning_rate[0]) * grad_t);
    variable[cur_pos] -= mom[cur_pos];
  }
}

template <typename T, typename S>
void CalSparseApplyRMSProp(const size_t size, const size_t indices_size, const float decay_rate, const float momentum,
                           const float epsilon, const T *learning_rate, const T *gradient, const S *indices,
                           T *variable, T *ms, T *mom, cudaStream_t cuda_stream) {
  SparseApplyRMSPropUpdate<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(
    size, indices_size, decay_rate, momentum, epsilon, learning_rate, gradient, indices, variable, ms, mom);
}

#define SPARSE_RMS_PROP_CAL_TEMPLATE(var_type, indices_type)                                                   \
  template CUDA_LIB_EXPORT void CalSparseApplyRMSProp<var_type, indices_type>(                                 \
    const size_t size, const size_t indices_size, const float decay_rate, const float momentum,                \
    const float epsilon, const var_type *learning_rate, const var_type *gradient, const indices_type *indices, \
    var_type *variable, var_type *ms, var_type *mom, cudaStream_t cuda_stream);

SPARSE_RMS_PROP_CAL_TEMPLATE(float, int);
SPARSE_RMS_PROP_CAL_TEMPLATE(float, int64_t);
SPARSE_RMS_PROP_CAL_TEMPLATE(half, int64_t);
SPARSE_RMS_PROP_CAL_TEMPLATE(half, int);
