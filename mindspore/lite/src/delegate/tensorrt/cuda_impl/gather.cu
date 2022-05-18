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

#include "src/delegate/tensorrt/cuda_impl/gather.cuh"
#include "src/delegate/tensorrt/cuda_impl/cuda_helper.h"
template <typename T, typename S>
__global__ void GatherKernel(const T *input, const S *index, T *output, const size_t dim_before_axis,
                             const size_t dim_at_axis_input, const size_t dim_at_axis_output,
                             const size_t dim_after_axis) {
  size_t num = dim_before_axis * dim_at_axis_output * dim_after_axis;
  size_t i, k;
  for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < num; id += blockDim.x * gridDim.x) {
    i = id / (dim_at_axis_output * dim_after_axis);
    k = id % dim_after_axis;

    S j = index[id];
    if (j < 0) {
      j += static_cast<S>(dim_at_axis_input);
    }
    size_t j_read = static_cast<size_t>(j);
    size_t read_id = i * dim_at_axis_input * dim_after_axis + j_read * dim_after_axis + k;
    output[id] = input[read_id];
  }
  return;
}
template <typename T, typename S>
void Gather(const T *input, const S *index, T *output, const size_t dim_before_axis, const size_t dim_at_axis_input,
            const size_t dim_at_axis_output, const size_t dim_after_axis, cudaStream_t stream) {
  size_t size = dim_before_axis * dim_at_axis_output * dim_after_axis;
  GatherKernel<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(input, index, output, dim_before_axis, dim_at_axis_input,
                                                             dim_at_axis_output, dim_after_axis);
  return;
}

template void Gather<float, int>(const float *input, const int *index, float *output, const size_t dim_before_axis,
                                 const size_t dim_at_axis_input, const size_t dim_at_axis_output,
                                 const size_t dim_after_axis, cudaStream_t stream);

template void Gather<int, int>(const int *input, const int *index, int *output, const size_t dim_before_axis,
                               const size_t dim_at_axis_input, const size_t dim_at_axis_output,
                               const size_t dim_after_axis, cudaStream_t stream);
