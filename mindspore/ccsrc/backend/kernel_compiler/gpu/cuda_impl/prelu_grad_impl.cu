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
                                   const T *dy, const T *x, const T *w, T *dx, float *dw_array) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t channel_id = weight_size == 1 ? 0 : (pos / per_channel_size) % weight_size;
    size_t index = channel_id * blockDim.x * gridDim.x + thread_id;
    T threshold = static_cast<T>(0);
    dx[pos] = x[pos] <= threshold ? w[channel_id] * dy[pos] : dy[pos];
    if (x[pos] < threshold) {
      dw_array[index] += static_cast<float>(x[pos] * dy[pos]);
    }
  }
}

__global__ void InitDwArrayData(size_t dw_array_size, float *dw_array) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < dw_array_size; i += blockDim.x * gridDim.x) {
    dw_array[i] = 0.0;
  }
}

template <typename T>
__global__ void ComputeDwData(size_t weight_size, size_t thread_num, const float *dw_array, T *dw) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < weight_size; i += blockDim.x * gridDim.x) {
    float value = 0.0;
    for (size_t j = 0; j < thread_num; j++) {
      value += dw_array[i * thread_num + j];
    }
    dw[i] = static_cast<T>(value);
  }
}

template <typename T>
void CalPReLUGrad(size_t size, size_t weight_size, size_t per_channel_size,
                  const T *dy, const T *x, const T *w, T *dx, T *dw, float *dw_array, cudaStream_t cuda_stream) {
  size_t thread_num = static_cast<size_t>(GET_BLOCKS(size) * GET_THREADS);
  size_t dw_array_size = weight_size * thread_num;
  InitDwArrayData<<<GET_BLOCKS(dw_array_size), GET_THREADS, 0, cuda_stream>>>(dw_array_size, dw_array);
  CalPReLUGradKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, weight_size, per_channel_size,
                                                                        dy, x, w, dx, dw_array);
  ComputeDwData<<<GET_BLOCKS(weight_size), GET_THREADS, 0, cuda_stream>>>(weight_size, thread_num, dw_array, dw);
  return;
}

template void CalPReLUGrad(size_t, size_t, size_t, const float *, const float *, const float *,
                           float *, float *, float *, cudaStream_t);
template void CalPReLUGrad(size_t, size_t, size_t, const half *, const half *, const half *,
                           half *, half *, float *, cudaStream_t);
