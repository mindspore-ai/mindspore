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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/lamb_impl.cuh"
#include "include/cuda_fp16.h"

const int32_t kSqareNum = 2;

template <typename T>
__global__ void ApplyLambEralyKernel(const size_t size, T *variable, T *m, T *v, const float *beta1, const float *beta2,
                                     const float *epsilon, const float *decay, const int32_t *global_step,
                                     const T *gradient, float *update, float *var_float, float *grad_float,
                                     float *g_hat_var) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    float next_m = (beta1[0] * m[i] + (1 - beta1[0]) * gradient[i]);
    float next_v = (beta2[0] * v[i] + (1 - beta2[0]) * pow(gradient[i], static_cast<T>(kSqareNum)));
    float next_mm = next_m / (1 - pow(beta1[0], static_cast<float>(global_step[0])));
    float next_vv = next_v / (1 - pow(beta2[0], static_cast<float>(global_step[0])));
    var_float[i] = variable[i];
    grad_float[i] = gradient[i];
    g_hat_var[i] = (next_mm / sqrt(next_vv + epsilon[0]) + decay[0] * variable[i]);
    update[i] = next_mm / (sqrt(next_vv) + epsilon[0]);
    update[i] += decay[0] * variable[i];
    m[i] = next_m;
    v[i] = next_v;
  }
}

template <>
__global__ void ApplyLambEralyKernel(const size_t size, half *variable, half *m, half *v, const float *beta1,
                                     const float *beta2, const float *epsilon, const float *decay,
                                     const int32_t *global_step, const half *gradient, float *update, float *var_float,
                                     float *grad_float, float *g_hat_var) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    float float_gradient = __half2float(gradient[i]);
    float float_var = __half2float(variable[i]);
    float float_decay = decay[0];

    float next_m = (beta1[0] * __half2float(m[i]) + (1 - beta1[0]) * float_gradient);
    float tmp_gradient = pow(float_gradient, static_cast<float>(kSqareNum));
    float next_v = beta2[0] * __half2float(v[i]) + (1 - beta2[0]) * tmp_gradient;
    float next_mm = next_m / (1 - pow(beta1[0], static_cast<float>(global_step[0])));
    float next_vv = next_v / (1 - pow(beta2[0], static_cast<float>(global_step[0])));
    var_float[i] = float_var;
    grad_float[i] = float_gradient;
    g_hat_var[i] = next_mm / sqrt(next_vv + epsilon[0]) + float_decay * float_var;
    update[i] = next_mm / (sqrt(next_vv) + epsilon[0]);
    update[i] += float_decay * float_var;
    m[i] = __float2half(next_m);
    v[i] = __float2half(next_v);
  }
}

template <typename T>
__global__ void ApplyLambAfterNormKernel(const size_t size, T *variable, const float *lr, const float *update,
                                         const float *trust_ratio) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    variable[i] = variable[i] - trust_ratio[0] * lr[0] * update[i];
  }
}

template <>
__global__ void ApplyLambAfterNormKernel(const size_t size, half *variable, const float *lr, const float *update,
                                         const float *trust_ratio) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
    variable[i] = __float2half(__half2float(variable[i]) - trust_ratio[0] * lr[0] * update[i]);
  }
}

template <typename T>
cudaError_t ApplyLambEraly(const size_t size, T *variable, T *m, T *v, const float *beta1, const float *beta2,
                           const float *epsilon, const float *decay, const int32_t *global_step, const T *gradient,
                           float *update, float *var_float, float *grad_float, float *g_hat_var,
                           cudaStream_t cuda_stream) {
  ApplyLambEralyKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, variable, m, v, beta1, beta2, epsilon,
                                                                          decay, global_step, gradient, update,
                                                                          var_float, grad_float, g_hat_var);
  return GetCudaStatus();
}

template <typename T>
cudaError_t ApplyLambLater(const size_t size, T *variable, const float *lr, const float *update,
                           const float *trust_ratio, cudaStream_t cuda_stream) {
  ApplyLambAfterNormKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, variable, lr, update, trust_ratio);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t ApplyLambEraly<float>(const size_t size, float *variable, float *m, float *v,
                                                           const float *beta1, const float *beta2, const float *epsilon,
                                                           const float *decay, const int32_t *global_step,
                                                           const float *gradient, float *update, float *w_square_ptr,
                                                           float *g_square_ptr, float *g_hat_square_ptr,
                                                           cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyLambEraly<half>(const size_t size, half *variable, half *m, half *v,
                                                          const float *beta1, const float *beta2, const float *epsilon,
                                                          const float *decay, const int32_t *global_step,
                                                          const half *gradient, float *update, float *w_square_ptr,
                                                          float *g_square_ptr, float *g_hat_square_ptr,
                                                          cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyLambLater<float>(const size_t size, float *variable, const float *lr,
                                                           const float *update, const float *trust_ratio,
                                                           cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyLambLater<half>(const size_t size, half *variable, const float *lr,
                                                          const float *update, const float *trust_ratio,
                                                          cudaStream_t cuda_stream);
