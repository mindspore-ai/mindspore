/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
template <typename T, typename S>
__global__ void MomentumUpdateVariableKernel(const size_t size, T *variable, T *accumulation, const S *learning_rate,
                                             const T *gradient, const S *momentum) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (size); i += blockDim.x * gridDim.x) {
    accumulation[i] = momentum[0] * accumulation[i] + gradient[i];
    variable[i] -= learning_rate[0] * accumulation[i];
  }
  return;
}
template <>
__global__ void MomentumUpdateVariableKernel(const size_t size, half *variable, half *accumulation,
                                             const float *learning_rate, const half *gradient,
                                             const float *momentum) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (size); i += blockDim.x * gridDim.x) {
    accumulation[i] = __float2half(momentum[0]) * accumulation[i] + gradient[i];
    variable[i] -= __float2half(learning_rate[0]) * accumulation[i];
  }
  return;
}
template <typename T, typename S>
void MomentumUpdateVariable(const size_t size, T *variable, T *accumulation, const S *learning_rate, const T *gradient,
                            const S *momentum, cudaStream_t cuda_stream) {
  MomentumUpdateVariableKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, variable, accumulation,
                                                                                  learning_rate, gradient, momentum);
  return;
}
template void MomentumUpdateVariable<float, float>(const size_t size, float *variable, float *accumulation,
                                                   const float *learning_rate, const float *gradient,
                                                   const float *momentum, cudaStream_t cuda_stream);
template void MomentumUpdateVariable<half, half>(const size_t size, half *variable, half *accumulation,
                                                 const half *learning_rate, const half *gradient,
                                                 const half *momentum, cudaStream_t cuda_stream);
template void MomentumUpdateVariable<half, float>(const size_t size, half *variable, half *accumulation,
                                                  const float *learning_rate, const half *gradient,
                                                  const float *momentum, cudaStream_t cuda_stream);
