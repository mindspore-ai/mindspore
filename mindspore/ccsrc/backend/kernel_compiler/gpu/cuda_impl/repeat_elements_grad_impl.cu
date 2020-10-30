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
#include <cuda_runtime.h>

#include "repeat_elements_grad_impl.cuh"
#include "runtime/device/gpu/cuda_common.h"

template <typename T>
__global__ void RepeatElementsGrad(const int dx_size, const T *dy, const int rep, T *dx, const int outer_size,
                                   const int repeat_dim_size, const int inner_size) {
  for (size_t t_id = blockIdx.x * blockDim.x + threadIdx.x; t_id < dx_size; t_id += gridDim.x * blockDim.x) {
    int inner_id = t_id % inner_size;
    int repeat_dim_id = t_id / inner_size % repeat_dim_size;
    int outer_id = t_id / inner_size / repeat_dim_size;
    T dx_i = static_cast<T>(0);
    for (int i = 0; i < rep; i++) {
      dx_i += dy[(outer_id * rep * repeat_dim_size * inner_size) + (repeat_dim_id * rep * inner_size) +
                 (i * inner_size) + inner_id];
    }
    dx[t_id] = dx_i;
  }
}

template <typename T>
void CalRepeatElementsGrad(const T *dy, const int rep, T *dx, const int outer_size, const int repeat_dim_size,
                           const int inner_size, cudaStream_t cuda_stream) {
  const int dx_size = outer_size * repeat_dim_size * inner_size;
  RepeatElementsGrad<<<GET_BLOCKS(dx_size), GET_THREADS, 0, cuda_stream>>>(dx_size, dy, rep, dx, outer_size,
                                                                           repeat_dim_size, inner_size);
}

template void CalRepeatElementsGrad<int>(const int *dy, const int rep, int *dx, const int outer_size,
                                         const int repeat_dim_size, const int inner_size, cudaStream_t cuda_stream);
template void CalRepeatElementsGrad<half>(const half *dy, const int rep, half *dx, const int outer_size,
                                          const int repeat_dim_size, const int inner_size, cudaStream_t cuda_stream);
