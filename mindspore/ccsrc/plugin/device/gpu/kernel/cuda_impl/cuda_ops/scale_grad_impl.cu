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

#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/scale_grad_impl.cuh"

template <typename T, typename S>
__global__ void ScaleGrad(const int nums, const T *x0, const S &x1, T *y) {
  T x1_t = static_cast<T>(x1);
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nums; pos += blockDim.x * gridDim.x) {
    y[pos] = x0[pos] * x1_t;
  }
}

template <typename T, typename S>
void ScaleGradKernel(const int &nums, const T *x0, const S &x1, T *y, cudaStream_t stream) {
  ScaleGrad<<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
  return;
}

template CUDA_LIB_EXPORT void ScaleGradKernel<float, float>(const int &nums, const float *x0, const float &x1, float *y,
                                                            cudaStream_t stream);
template CUDA_LIB_EXPORT void ScaleGradKernel<float, half>(const int &nums, const float *x0, const half &x1, float *y,
                                                           cudaStream_t stream);
template CUDA_LIB_EXPORT void ScaleGradKernel<half, float>(const int &nums, const half *x0, const float &x1, half *y,
                                                           cudaStream_t stream);
template CUDA_LIB_EXPORT void ScaleGradKernel<half, half>(const int &nums, const half *x0, const half &x1, half *y,
                                                          cudaStream_t stream);
