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

#include "backend/kernel_compiler/gpu/cuda_impl/prelu_grad_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/util.cuh"
#include "runtime/device/gpu/cuda_common.h"

template <typename T>
__global__ void CalPReLUGradKernel(size_t size, size_t weight_size, size_t per_channel_size,
                                   const T *dy, const T *x, const T *w, T *dx, T *dw) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    size_t index = 0;
    if (weight_size != 1) {
      index = (pos / per_channel_size) % weight_size;
    }
    T threshold = static_cast<T>(0);
    dx[pos] = pos[x] <= threshold ? w[index] * dy[pos] : dy[pos];
    if (pos[x] < threshold) {
      MsAtomicAdd(dw + index, x[pos] * dy[pos]);
    }
  }
}

template <typename T>
__global__ void InitDwData(size_t weight_size, T *dw) {
  T init_value = static_cast<T>(0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < weight_size; i += blockDim.x * gridDim.x) {
    dw[i] = init_value;
  }
}


template <typename T>
void CalPReLUGrad(size_t size, size_t weight_size, size_t per_channel_size,
                  const T *dy, const T *x, const T *w, T *dx, T *dw, cudaStream_t cuda_stream) {
  InitDwData<<<GET_BLOCKS(weight_size), GET_THREADS, 0, cuda_stream>>>(weight_size, dw);
  CalPReLUGradKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, weight_size, per_channel_size,
                                                                        dy, x, w, dx, dw);
  return;
}

template void CalPReLUGrad(size_t, size_t, size_t, const float *, const float *, const float *, float *, float *,
                           cudaStream_t);
template void CalPReLUGrad(size_t, size_t, size_t, const half *, const half *, const half *, half *, half *,
                           cudaStream_t);
