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

#include <algorithm>
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/apply_adagrad_d_a_impl.cuh"

template <typename T>
__device__ __forceinline__ T SqrtFunc(T input) {
  return sqrt(input);
}

template <>
__device__ __forceinline__ half SqrtFunc(half input) {
  return hsqrt(input);
}

template <typename T>
__device__ __forceinline__ T AbsFunc(T x) {
  return abs(x);
}

template <>
__device__ __forceinline__ half AbsFunc(half x) {
  return abs(__half2float(x));
}

template <typename T>
__device__ __forceinline__ T MaxFunc(T x, T y) {
  return max(x, y);
}

template <>
__device__ __forceinline__ half MaxFunc(half x, half y) {
  return max(__half2float(x), __half2float(y));
}

template <typename T>
__device__ __forceinline__ T Sign(T num) {
  if (num > static_cast<T>(0.0)) {
    return static_cast<T>(1.0);
  } else if (num == static_cast<T>(0.0)) {
    return static_cast<T>(0.0);
  } else {
    return static_cast<T>(-1.0);
  }
}

template <typename T, typename S>
__global__ void ApplyAdagradDAKernel(const size_t batch_size, const size_t size, T *var, T * accum, T *squared_accum,
                                     const T *grad, const T *lr, const T *l1, const T *l2, const S *global_step,
                                     T *output_var, T *output_accum, T *output_squared_accum) {
  T zero = static_cast<T>(0.0);
  T minus_one = static_cast<T>(-1);
  for (size_t b = 0; b < batch_size; b++) {
    for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += gridDim.x * blockDim.x) {
      output_accum[pos] = accum[pos] + grad[pos];
      output_squared_accum[pos] = squared_accum[pos] + grad[pos] * grad[pos];
      T tmp_val;
      if (lr[0] > zero) {
        T tmp_accum = AbsFunc(output_accum[pos]) - l1[0] * static_cast<T>(static_cast<double>(global_step[0]));
        tmp_val = Sign(output_accum[pos]) * MaxFunc(tmp_accum, zero);
      } else {
        tmp_val = output_accum[pos];
      }
      auto x_value = minus_one * lr[0] * tmp_val;
      auto y_value = l2[0] * static_cast<T>(static_cast<double>(global_step[0])) * lr[0] +
      SqrtFunc(output_squared_accum[pos]);
      output_var[pos] = x_value / y_value;
    }
    var = var + size;
    accum = accum + size;
    grad = grad + size;
  }
}

template <typename T, typename S>
void ApplyAdagradDA(const size_t batch_size, const size_t size, T *var, T * accum, T *squared_accum, const T *grad,
                    const T *lr, const T *l1, const T *l2, const S *global_step, T *output_var,
                    T *output_accum, T *output_squared_accum,
                    const uint32_t &device_id, cudaStream_t cuda_stream) {
  ApplyAdagradDAKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(batch_size, size,
    var, accum, squared_accum, grad, lr, l1, l2, global_step, output_var, output_accum, output_squared_accum);
}

template CUDA_LIB_EXPORT void ApplyAdagradDA<float, int32_t>(const size_t batch_size, const size_t size, float *var,
                                                             float * accum, float *squared_accum, const float *grad,
                                                             const float *lr, const float *l1, const float *l2,
                                                             const int32_t *global_step, float *output_var,
                                                             float *output_accum, float *output_squared_accum,
                                                             const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ApplyAdagradDA<float, int64_t>(const size_t batch_size, const size_t size, float *var,
                                                             float * accum, float *squared_accum, const float *grad,
                                                             const float *lr, const float *l1, const float *l2,
                                                             const int64_t *global_step, float *output_var,
                                                             float *output_accum, float *output_squared_accum,
                                                             const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ApplyAdagradDA<half, int32_t>(const size_t batch_size, const size_t size, half *var,
                                                            half * accum, half *squared_accum, const half *grad,
                                                            const half *lr, const half *l1, const half *l2,
                                                            const int32_t *global_step, half *output_var,
                                                             half *output_accum, half *output_squared_accum,
                                                             const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ApplyAdagradDA<half, int64_t>(const size_t batch_size, const size_t size, half *var,
                                                            half * accum, half *squared_accum, const half *grad,
                                                            const half *lr, const half *l1, const half *l2,
                                                            const int64_t *global_step, half *output_var,
                                                            half *output_accum, half *output_squared_accum,
                                                            const uint32_t &device_id, cudaStream_t cuda_stream);
