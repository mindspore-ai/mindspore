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

#include <iostream>
#include "backend/kernel_compiler/gpu/cuda_impl/gather_grad.cuh"
#include "runtime/device/gpu/cuda_common.h"
template <typename T, typename S>
__global__ void GatherGradKernel(const T *index, const S *grad, S *output, const size_t output_dim0,
                                 const size_t output_dim1, const size_t output_dim2) {
  size_t num = output_dim0 * output_dim1 * output_dim2;
  size_t i, k;
  for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < num;
       id += blockDim.x * gridDim.x) {
    i = id / (output_dim1 * output_dim2) % output_dim0;
    k = id % output_dim2;

    size_t j_read = static_cast<size_t>(index[id]);
    size_t read_id = i * output_dim1 * output_dim2 + j_read * output_dim2 + k;
    output[read_id] = grad[id];
  }
  return;
}
template <typename T, typename S>
void GatherGrad(const T *index, const S *grad, S *output, const size_t output_dim0,
                const size_t output_dim1, const size_t output_dim2, cudaStream_t stream) {
  size_t size = output_dim0 * output_dim1 * output_dim2;
  GatherGradKernel<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(index, grad, output,
                                                                 output_dim0, output_dim1, output_dim2);
  return;
}

template void GatherGrad<int, float>(const int *index, const float *grad, float *output,
                                     const size_t output_dim0, const size_t output_dim1,
                                     const size_t output_dim2, cudaStream_t stream);

template void GatherGrad<int, half>(const int *index, const half *grad, half *output,
                                    const size_t output_dim0, const size_t output_dim1,
                                    const size_t output_dim2, cudaStream_t stream);

template void GatherGrad<int64_t, float>(const int64_t *index, const float *grad, float *output,
                                         const size_t output_dim0, const size_t output_dim1,
                                         const size_t output_dim2, cudaStream_t stream);

template void GatherGrad<int64_t, half>(const int64_t *index, const half *grad, half *output,
                                        const size_t output_dim0, const size_t output_dim1,
                                        const size_t output_dim2, cudaStream_t stream);
