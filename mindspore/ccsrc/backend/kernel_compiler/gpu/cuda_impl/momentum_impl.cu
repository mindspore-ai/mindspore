/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "momentum_impl.cuh"
template <typename T, typename S, typename G>
__global__ void MomentumUpdateVariableKernel(const size_t size, T *variable, T *accumulation, const S *learning_rate,
                                             const G *gradient, const S *momentum) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (size); i += blockDim.x * gridDim.x) {
    accumulation[i] = momentum[0] * accumulation[i] + gradient[i];
    variable[i] -= learning_rate[0] * accumulation[i];
  }
  return;
}
template <>
__global__ void MomentumUpdateVariableKernel(const size_t size, half *variable, half *accumulation,
                                             const float *learning_rate, const half *gradient, const float *momentum) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (size); i += blockDim.x * gridDim.x) {
    accumulation[i] = __float2half(momentum[0]) * accumulation[i] + gradient[i];
    variable[i] -= __float2half(learning_rate[0]) * accumulation[i];
  }
  return;
}
template <>
__global__ void MomentumUpdateVariableKernel(const size_t size, float *variable, float *accumulation,
                                             const float *learning_rate, const half *gradient, const float *momentum) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (size); i += blockDim.x * gridDim.x) {
    accumulation[i] = momentum[0] * accumulation[i] + __half2float(gradient[i]);
    variable[i] -= learning_rate[0] * accumulation[i];
  }
  return;
}
template <typename T, typename S, typename G>
void MomentumUpdateVariable(const size_t size, T *variable, T *accumulation, const S *learning_rate, const G *gradient,
                            const S *momentum, cudaStream_t cuda_stream) {
  MomentumUpdateVariableKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, variable, accumulation,
                                                                                  learning_rate, gradient, momentum);
  return;
}

template <typename T, typename S>
__global__ void FusedMomentumWeightDecayScaleMomentum(const size_t element_num, T *weight_decay, T *scale, T *variable,
                                                      T *accumulation, const T *learning_rate, const S *gradient,
                                                      const T *momentum) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (element_num); i += blockDim.x * gridDim.x) {
    T grad = (variable[i] * weight_decay[0] + static_cast<T>(gradient[i])) * scale[0];
    accumulation[i] = momentum[0] * accumulation[i] + grad;
    variable[i] -= learning_rate[0] * accumulation[i];
  }
}

template <typename T, typename S>
void FusedWeightDecayScaleMomentum(const size_t element_num, T *weight_decay, T *scale, T *variable, T *accumulation,
                                   const T *learning_rate, const S *gradient, const T *momentum,
                                   cudaStream_t cuda_stream) {
  size_t thread_per_block = 256;
  size_t block_per_grid = (element_num + thread_per_block - 1) / thread_per_block;
  FusedMomentumWeightDecayScaleMomentum<<<block_per_grid, thread_per_block, 0, cuda_stream>>>(
    element_num, weight_decay, scale, variable, accumulation, learning_rate, gradient, momentum);
}

template <typename T, typename S>
__global__ void FusedMomentumScaleMomentum(const size_t element_num, T *scale, T *variable, T *accumulation,
                                           const T *learning_rate, const S *gradient, const T *momentum) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (element_num); i += blockDim.x * gridDim.x) {
    accumulation[i] = momentum[0] * accumulation[i] + static_cast<T>(gradient[i]);
    variable[i] -= learning_rate[0] * accumulation[i];
  }
}

template <typename T, typename S>
void FusedScaleMomentum(const size_t element_num, T *scale, T *variable, T *accumulation, const T *learning_rate,
                        const S *gradient, const T *momentum, cudaStream_t cuda_stream) {
  size_t thread_per_block = 256;
  size_t block_per_grid = (element_num + thread_per_block - 1) / thread_per_block;
  FusedMomentumScaleMomentum<<<block_per_grid, thread_per_block, 0, cuda_stream>>>(
    element_num, scale, variable, accumulation, learning_rate, gradient, momentum);
}

template void MomentumUpdateVariable<float, float, float>(const size_t size, float *variable, float *accumulation,
                                                          const float *learning_rate, const float *gradient,
                                                          const float *momentum, cudaStream_t cuda_stream);
template void MomentumUpdateVariable<half, half, half>(const size_t size, half *variable, half *accumulation,
                                                       const half *learning_rate, const half *gradient,
                                                       const half *momentum, cudaStream_t cuda_stream);
template void MomentumUpdateVariable<half, float, half>(const size_t size, half *variable, half *accumulation,
                                                        const float *learning_rate, const half *gradient,
                                                        const float *momentum, cudaStream_t cuda_stream);
template void MomentumUpdateVariable<float, float, half>(const size_t size, float *variable, float *accumulation,
                                                         const float *learning_rate, const half *gradient,
                                                         const float *momentum, cudaStream_t cuda_stream);

template void FusedWeightDecayScaleMomentum(const size_t element_num, float *weight_decay, float *scale,
                                            float *variable, float *accumulation, const float *learning_rate,
                                            const float *gradient, const float *momentum, cudaStream_t cuda_stream);
template void FusedWeightDecayScaleMomentum(const size_t element_num, float *weight_decay, float *scale,
                                            float *variable, float *accumulation, const float *learning_rate,
                                            const half *gradient, const float *momentum, cudaStream_t cuda_stream);
template void FusedScaleMomentum(const size_t element_num, float *scale, float *variable, float *accumulation,
                                 const float *learning_rate, const float *gradient, const float *momentum,
                                 cudaStream_t cuda_stream);
template void FusedScaleMomentum(const size_t element_num, float *scale, float *variable, float *accumulation,
                                 const float *learning_rate, const half *gradient, const float *momentum,
                                 cudaStream_t cuda_stream);
