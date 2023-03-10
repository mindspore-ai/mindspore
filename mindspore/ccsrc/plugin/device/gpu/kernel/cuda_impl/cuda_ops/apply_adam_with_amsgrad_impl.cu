/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/apply_adam_with_amsgrad_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__device__ __forceinline__ T sqrtFunc(T x) {
  return sqrt(x);
}

template <>
__device__ __forceinline__ half sqrtFunc(half x) {
  return hsqrt(x);
}

template <typename T>
__device__ __forceinline__ T maxFunc(T x, T y) {
  return x > y ? x : y;
}

template <>
__device__ __forceinline__ half maxFunc(half x, half y) {
  return x > y ? x : y;
}

template <typename T>
__global__ void CalApplyAdamWithAmsgradKernel(const size_t size, const int64_t batch_size, T *var, T *m, T *v, T *vhat,
                                              T *beta1_power, T *beta2_power, const T *lr, const T *grad, const T beta1,
                                              const T beta2, const T epsilon, T *output_var, T *output_m, T *output_v,
                                              T *output_vhat) {
  auto all_elements = size * batch_size;
  const T one = static_cast<T>(1.0);
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < all_elements; pos += gridDim.x * blockDim.x) {
    auto batch = pos / size;
    auto new_learning_rate = lr[batch] * sqrtFunc(one - beta2_power[batch]) / (one - beta1_power[batch]);
    m[pos] += (grad[pos] - m[pos]) * (one - static_cast<T>(beta1));
    output_m[pos] = m[pos];
    v[pos] += (grad[pos] * grad[pos] - v[pos]) * (one - static_cast<T>(beta2));
    output_v[pos] = v[pos];
    vhat[pos] = maxFunc(vhat[pos], v[pos]);
    output_vhat[pos] = vhat[pos];
    var[pos] -= new_learning_rate * m[pos] / (sqrtFunc(vhat[pos]) + static_cast<T>(epsilon));
    output_var[pos] = var[pos];
  }
}

template <typename T>
cudaError_t CalApplyAdamWithAmsgrad(const size_t size, const int64_t batch_size, T *var, T *m, T *v, T *vhat,
                                    T *beta1_power, T *beta2_power, const T *lr, const T *grad, const T beta1,
                                    const T beta2, const T epsilon, T *output_var, T *output_m, T *output_v,
                                    T *output_vhat, const uint32_t &device_id, cudaStream_t stream_ptr) {
  CalApplyAdamWithAmsgradKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, stream_ptr>>>(
    size, batch_size, var, m, v, vhat, beta1_power, beta2_power, lr, grad, beta1, beta2, epsilon, output_var, output_m,
    output_v, output_vhat);
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template CUDA_LIB_EXPORT cudaError_t CalApplyAdamWithAmsgrad<double>(
  const size_t size, const int64_t batch_size, double *var, double *m, double *v, double *vhat, double *beta1_power,
  double *beta2_power, const double *lr, const double *grad, const double beta1, const double beta2,
  const double epsilon, double *output_var, double *output_m, double *output_v, double *output_vhat,
  const uint32_t &device_id, cudaStream_t stream_ptr);

template CUDA_LIB_EXPORT cudaError_t CalApplyAdamWithAmsgrad<float>(
  const size_t size, const int64_t batch_size, float *var, float *m, float *v, float *vhat, float *beta1_power,
  float *beta2_power, const float *lr, const float *grad, const float beta1, const float beta2, const float epsilon,
  float *output_var, float *output_m, float *output_v, float *output_vhat, const uint32_t &device_id,
  cudaStream_t stream_ptr);

template CUDA_LIB_EXPORT cudaError_t CalApplyAdamWithAmsgrad<half>(const size_t size, const int64_t batch_size,
                                                                   half *var, half *m, half *v, half *vhat,
                                                                   half *beta1_power, half *beta2_power, const half *lr,
                                                                   const half *grad, const half beta1, const half beta2,
                                                                   const half epsilon, half *output_var, half *output_m,
                                                                   half *output_v, half *output_vhat,
                                                                   const uint32_t &device_id, cudaStream_t stream_ptr);
