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

#include "backend/kernel_compiler/gpu/cuda_impl/apply_gradient_descent_impl.cuh"

template <typename T>
__global__ void ApplyGradientDescent(const size_t size, T *var, const T *alpha, const T *delta, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
      const T alpha_value = alpha[0];
      var[pos] -= alpha_value * delta[pos];
      output[pos] = var[pos];
  }
}

template <typename T>
void CalApplyGradientDescent(const size_t &size, T *var, const T *alpha, const T *delta, T *output,
                             cudaStream_t cuda_stream) {
  ApplyGradientDescent<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, var, alpha, delta, output);
}

template void CalApplyGradientDescent<float>(const size_t &size, float *var, const float *alpha, const float *delta,
                                             float *output, cudaStream_t cuda_stream);
template void CalApplyGradientDescent<half>(const size_t &size, half *var, const half *alpha, const half *delta,
                                            half *output, cudaStream_t cuda_stream);
