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

template <typename DataType>
__global__ void IndexFillKernel(DataType *out_ptr, const int *index_ptr, int64_t index_size, int64_t dim_size,
                                int64_t inner_size, const DataType *value_ptr, bool *out_bound_ptr, int64_t stride1,
                                int64_t stride2) {
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
    out_ptr[out_idx] = *value_ptr;
  }
}

template <typename DataType>
void IndexFill(DataType *out_ptr, const int *index_ptr, int64_t index_size, int64_t outer_size, int64_t dim_size,
               int64_t inner_size, const DataType *value_ptr, bool *out_bound_ptr, cudaStream_t cuda_stream) {
  int64_t stride1 = outer_size * inner_size;
  int64_t stride2 = dim_size * inner_size;
  IndexFillKernel<<<GET_BLOCKS(index_size), GET_THREADS, 0, cuda_stream>>>(
    out_ptr, index_ptr, index_size, dim_size, inner_size, value_ptr, out_bound_ptr, stride1, stride2);
}

template CUDA_LIB_EXPORT void IndexFill<uint8_t>(uint8_t *out_ptr, const int *index_ptr, int64_t index_size,
                                                 int64_t outer_size, int64_t dim_size, int64_t inner_size,
                                                 const uint8_t *value_ptr, bool *out_bound_ptr,
                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void IndexFill<uint16_t>(uint16_t *out_ptr, const int *index_ptr, int64_t index_size,
                                                  int64_t outer_size, int64_t dim_size, int64_t inner_size,
                                                  const uint16_t *value_ptr, bool *out_bound_ptr,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void IndexFill<uint32_t>(uint32_t *out_ptr, const int *index_ptr, int64_t index_size,
                                                  int64_t outer_size, int64_t dim_size, int64_t inner_size,
                                                  const uint32_t *value_ptr, bool *out_bound_ptr,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void IndexFill<uint64_t>(uint64_t *out_ptr, const int *index_ptr, int64_t index_size,
                                                  int64_t outer_size, int64_t dim_size, int64_t inner_size,
                                                  const uint64_t *value_ptr, bool *out_bound_ptr,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void IndexFill<int8_t>(int8_t *out_ptr, const int *index_ptr, int64_t index_size,
                                                int64_t outer_size, int64_t dim_size, int64_t inner_size,
                                                const int8_t *value_ptr, bool *out_bound_ptr, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void IndexFill<int16_t>(int16_t *out_ptr, const int *index_ptr, int64_t index_size,
                                                 int64_t outer_size, int64_t dim_size, int64_t inner_size,
                                                 const int16_t *value_ptr, bool *out_bound_ptr,
                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void IndexFill<int32_t>(int32_t *out_ptr, const int *index_ptr, int64_t index_size,
                                                 int64_t outer_size, int64_t dim_size, int64_t inner_size,
                                                 const int32_t *value_ptr, bool *out_bound_ptr,
                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void IndexFill<int64_t>(int64_t *out_ptr, const int *index_ptr, int64_t index_size,
                                                 int64_t outer_size, int64_t dim_size, int64_t inner_size,
                                                 const int64_t *value_ptr, bool *out_bound_ptr,
                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void IndexFill<half>(half *out_ptr, const int *index_ptr, int64_t index_size,
                                              int64_t outer_size, int64_t dim_size, int64_t inner_size,
                                              const half *value_ptr, bool *out_bound_ptr, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void IndexFill<float>(float *out_ptr, const int *index_ptr, int64_t index_size,
                                               int64_t outer_size, int64_t dim_size, int64_t inner_size,
                                               const float *value_ptr, bool *out_bound_ptr, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void IndexFill<double>(double *out_ptr, const int *index_ptr, int64_t index_size,
                                                int64_t outer_size, int64_t dim_size, int64_t inner_size,
                                                const double *value_ptr, bool *out_bound_ptr, cudaStream_t cuda_stream);
