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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/index_fill_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T>
__global__ void IndexFillKernel(T *out_ptr, const int *index_ptr, const size_t index_size, const size_t outer_size,
                                const int dim_size, const size_t inner_size, const T value, bool *out_bound_ptr) {
  size_t stride1 = outer_size * inner_size;
  size_t stride2 = dim_size * inner_size;
  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < index_size; tid += blockDim.x * gridDim.x) {
    // Each index must be [-dim_size, dim_size)
    int index = index_ptr[tid / stride1];
    if (index < -dim_size || index >= dim_size) {
      *out_bound_ptr = true;
      break;
    } else if (index < 0) {
      index += dim_size;
    }
    size_t offset = tid % stride1;
    size_t inner_idx = offset % inner_size;
    size_t outer_idx = offset / inner_size;
    size_t out_idx = outer_idx * stride2 + index * inner_size + inner_idx;
    out_ptr[out_idx] = value;
  }
}

template <typename T>
bool IndexFill(T *out_ptr, const int *index_ptr, const size_t index_size, const size_t outer_size, const int dim_size,
               const size_t inner_size, const T value, cudaStream_t cuda_stream) {
  // We use a bool to indicate whether all the index is valid.
  bool out_bound, *out_bound_ptr;
  cudaMalloc(&out_bound_ptr, sizeof(bool));
  IndexFillKernel<<<GET_BLOCKS(index_size), GET_THREADS, 0, cuda_stream>>>(out_ptr, index_ptr, index_size, outer_size,
                                                                           dim_size, inner_size, value, out_bound_ptr);
  cudaMemcpyAsync(&out_bound, out_bound_ptr, sizeof(bool), cudaMemcpyDeviceToHost, cuda_stream);
  cudaFree(out_bound_ptr);
  return out_bound;
}

template CUDA_LIB_EXPORT bool IndexFill<double>(double *out_ptr, const int *index_ptr, const size_t index_size,
                                                const size_t outer_size, const int dim_size, const size_t inner_size,
                                                const double value, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT bool IndexFill<float>(float *out_ptr, const int *index_ptr, const size_t index_size,
                                               const size_t outer_size, const int dim_size, const size_t inner_size,
                                               const float value, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT bool IndexFill<half>(half *out_ptr, const int *index_ptr, const size_t index_size,
                                              const size_t outer_size, const int dim_size, const size_t inner_size,
                                              const half value, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT bool IndexFill<int8_t>(int8_t *out_ptr, const int *index_ptr, const size_t index_size,
                                                const size_t outer_size, const int dim_size, const size_t inner_size,
                                                const int8_t value, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT bool IndexFill<int16_t>(int16_t *out_ptr, const int *index_ptr, const size_t index_size,
                                                 const size_t outer_size, const int dim_size, const size_t inner_size,
                                                 const int16_t value, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT bool IndexFill<int>(int *out_ptr, const int *index_ptr, const size_t index_size,
                                             const size_t outer_size, const int dim_size, const size_t inner_size,
                                             const int value, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT bool IndexFill<int64_t>(int64_t *out_ptr, const int *index_ptr, const size_t index_size,
                                                 const size_t outer_size, const int dim_size, const size_t inner_size,
                                                 const int64_t value, cudaStream_t cuda_stream);
