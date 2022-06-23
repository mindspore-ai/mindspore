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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/apply_adam_with_amsgrad_impl.cuh"
#include "include/cuda_fp16.h"


template <typename T>
__global__ void ApplyAdamWithAmsgradKernel(const size_t size,
                                   T *var,
                                   T *m,
                                   T *v,
                                   T *vhat,
                                   T *beta1_power,
                                   T *beta2_power,
                                   T *lr,
                                   T *grad,
                                   const float beta1,
                                   const float beta2,
                                   const float epsilon) {
  const T one = static_cast<T>(1.0);
  const T new_learning_rate = lr[0] * sqrt(one - beta2_power[0]) / (one - beta1_power[0]);

  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    m[i] += (grad[i] - m[i]) * (one - beta1);
    v[i] += (grad[i] * grad[i] - v[i]) * (one - beta2);
    vhat[i] = static_cast<T>(vhat[i] > v[i] ? vhat[i] : v[i]);
    var[i] -= new_learning_rate * m[i] / (sqrt(vhat[i]) + epsilon);
  }
}

template <>
__global__ void ApplyAdamWithAmsgradKernel(const size_t size,
                                   half *var,
                                   half *m,
                                   half *v,
                                   half *vhat,
                                   half *beta1_power,
                                   half *beta2_power,
                                   half *lr,
                                   half *grad,
                                   const float beta1,
                                   const float beta2,
                                   const float epsilon) {
  const float one = static_cast<float>(1.0);
  const float new_learning_rate = __half2float(lr[0]) *
                                  sqrt(one - __half2float(beta2_power[0])) /
                                  (one - __half2float(beta1_power[0]));

  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    m[i] += (__half2float(grad[i]) - __half2float(m[i])) * (one - beta1);
    v[i] += (__half2float(grad[i]) * __half2float(grad[i]) - __half2float(v[i])) * (one - beta2);
    vhat[i] = __float2half(__half2float(vhat[i]) > __half2float(v[i]) ? __half2float(vhat[i]) : __half2float(v[i]));
    var[i] -= __float2half(new_learning_rate) * m[i] /
              (__float2half(sqrt(__half2float(vhat[i]))) + __float2half(epsilon));
  }
}

template <typename T>
void ApplyAdamWithAmsgrad(const size_t size,
                  T *var,
                  T *m,
                  T *v,
                  T *vhat,
                  T *beta1_power,
                  T *beta2_power,
                  T *lr,
                  T *grad,
                  const float beta1,
                  const float beta2,
                  const float epsilon,
                  const uint32_t &device_id,
                  cudaStream_t cuda_stream) {
  ApplyAdamWithAmsgradKernel<<< CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
          size, var, m, v, vhat, beta1_power, beta2_power, lr, grad, beta1, beta2, epsilon);
}

template CUDA_LIB_EXPORT void ApplyAdamWithAmsgrad<double>(const size_t size,
                                                                   double *var,
                                                                   double *m,
                                                                   double *v,
                                                                   double *vhat,
                                                                   double *beta1_power,
                                                                   double *beta2_power,
                                                                   double *lr,
                                                                   double *grad,
                                                                   const float beta1,
                                                                   const float beta2,
                                                                   const float epsilon,
                                                                   const uint32_t &device_id,
                                                                   cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void ApplyAdamWithAmsgrad<float>(const size_t size,
                                                                   float *var,
                                                                   float *m,
                                                                   float *v,
                                                                   float *vhat,
                                                                   float *beta1_power,
                                                                   float *beta2_power,
                                                                   float *lr,
                                                                   float *grad,
                                                                   const float beta1,
                                                                   const float beta2,
                                                                   const float epsilon,
                                                                   const uint32_t &device_id,
                                                                   cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void ApplyAdamWithAmsgrad<half>(const size_t size,
                                                                   half *var,
                                                                   half *m,
                                                                   half *v,
                                                                   half *vhat,
                                                                   half *beta1_power,
                                                                   half *beta2_power,
                                                                   half *lr,
                                                                   half *grad,
                                                                   const float beta1,
                                                                   const float beta2,
                                                                   const float epsilon,
                                                                   const uint32_t &device_id,
                                                                   cudaStream_t cuda_stream);
