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

#include "backend/kernel_compiler/gpu/cuda_impl/relu_grad_impl.cuh"
#include "runtime/device/gpu/cuda_common.h"

template <typename T>
__global__ void CalReLUGradKernel(int size, T *dy, T *y, T *dx) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    dx[pos] = y[pos] > static_cast<T>(0) ? dy[pos] : static_cast<T>(0);
  }
}

template <typename T>
void CalReLUGrad(int size, T *dy, T *y, T *dx, cudaStream_t cuda_stream) {
  CalReLUGradKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, dy, y, dx);
  return;
}

template void CalReLUGrad(int size, double *dy, double *y, double *dx, cudaStream_t cuda_stream);
template void CalReLUGrad(int size, float *dy, float *y, float *dx, cudaStream_t cuda_stream);
template void CalReLUGrad(int size, half *dy, half *y, half *dx, cudaStream_t cuda_stream);
template void CalReLUGrad(int size, int8_t *dy, int8_t *y, int8_t *dx, cudaStream_t cuda_stream);
template void CalReLUGrad(int size, int16_t *dy, int16_t *y, int16_t *dx, cudaStream_t cuda_stream);
template void CalReLUGrad(int size, int32_t *dy, int32_t *y, int32_t *dx, cudaStream_t cuda_stream);
template void CalReLUGrad(int size, int64_t *dy, int64_t *y, int64_t *dx, cudaStream_t cuda_stream);
template void CalReLUGrad(int size, uint8_t *dy, uint8_t *y, uint8_t *dx, cudaStream_t cuda_stream);
