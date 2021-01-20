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

#include "one_hot_impl.cuh"
#include "runtime/device/gpu/cuda_common.h"
template <typename T, typename S>
__global__ void OneHotKernel(size_t size, const S *indices, size_t depth, const T *on_value, const T *off_value,
                             size_t left_dim_size, size_t right_dim_size, T *output) {
  T on_v = *on_value;
  T off_v = *off_value;
  for (size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x; thread_idx < size;
       thread_idx += blockDim.x * gridDim.x) {
    if (thread_idx < size) {
      size_t left_idx = (thread_idx / (depth * right_dim_size)) % left_dim_size;
      size_t d_idx = thread_idx / right_dim_size % depth;
      size_t right_idx = thread_idx % right_dim_size;
      size_t input_idx = left_idx * right_dim_size + right_idx;
      size_t output_idx = left_idx * depth * right_dim_size + d_idx * right_dim_size + right_idx;
      if (indices[input_idx] == d_idx) {
        output[output_idx] = on_v;
      } else {
        output[output_idx] = off_v;
      }
    }
  }
}
template <typename T, typename S>
void OneHot(const S *indices, size_t depth, const T *on_value, const T *off_value, size_t left_dim_size,
            size_t right_dim_size, T *output, cudaStream_t cuda_stream) {
  size_t size = left_dim_size * depth * right_dim_size;
  OneHotKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, indices, depth, on_value, off_value,
                                                                  left_dim_size, right_dim_size, output);
  return;
}
template void OneHot<float, int>(const int *indices, size_t depth, const float *on_value, const float *off_value,
                                 size_t left_dim_size, size_t right_dim_size, float *output, cudaStream_t cuda_stream);
template void OneHot<half, int>(const int *indices, size_t depth, const half *on_value, const half *off_value,
                                size_t left_dim_size, size_t right_dim_size, half *output, cudaStream_t cuda_stream);
template void OneHot<float, int64_t>(const int64_t *indices, size_t depth, const float *on_value,
                                     const float *off_value, size_t left_dim_size, size_t right_dim_size, float *output,
                                     cudaStream_t cuda_stream);
template void OneHot<half, int64_t>(const int64_t *indices, size_t depth, const half *on_value, const half *off_value,
                                    size_t left_dim_size, size_t right_dim_size, half *output,
                                    cudaStream_t cuda_stream);
