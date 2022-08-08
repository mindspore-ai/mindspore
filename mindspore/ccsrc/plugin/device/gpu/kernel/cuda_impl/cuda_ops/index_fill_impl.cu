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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

template <typename DataType, typename Int>
__global__ void IndexFillKernel(const int *__restrict__ index_ptr, const DataType *__restrict__ value_ptr,
                                bool *__restrict__ out_bound_ptr, DataType *__restrict__ out_ptr, Int dim_size,
                                Int inner_size, Int outer_inner_size, Int index_num) {
  DataType fill_value = *value_ptr;
  Int start_idx = static_cast<Int>(blockIdx.x * blockDim.x + threadIdx.x);
  Int step = static_cast<Int>(blockDim.x * gridDim.x);
  for (Int tid = start_idx; tid < index_num; tid += step) {
    Int index_idx = tid / outer_inner_size;
    Int outer_inner_idx = tid % outer_inner_size;
    // Each index must be [-dim_size, dim_size)
    Int dim_idx = static_cast<Int>(index_ptr[index_idx]);
    if (dim_idx < -dim_size || dim_idx >= dim_size) {
      *out_bound_ptr = true;
      break;
    } else if (dim_idx < 0) {
      dim_idx += dim_size;
    }
    Int inner_idx = outer_inner_idx % inner_size;
    Int outer_idx = (outer_inner_idx - inner_idx) * dim_size;
    Int out_idx = outer_idx + dim_idx * inner_size + inner_idx;
    out_ptr[out_idx] = fill_value;
  }
}

template <typename DataType>
void IndexFill(DataType *out_ptr, const int *index_ptr, int64_t index_size, int64_t outer_size, int64_t dim_size,
               int64_t inner_size, const DataType *value_ptr, bool *out_bound_ptr, const uint32_t &device_id,
               cudaStream_t cuda_stream) {
  int64_t outer_inner_size = outer_size * inner_size;
  int64_t index_num = outer_inner_size * index_size;
  int64_t element_num = outer_inner_size * dim_size;
  int64_t max_int32_value = std::numeric_limits<int>::max();
  auto grids = CUDA_BLOCKS(device_id, index_num);
  auto blocks = CUDA_THREADS(device_id);
  if (index_num <= max_int32_value && element_num <= max_int32_value) {
    IndexFillKernel<DataType, int><<<grids, blocks, 0, cuda_stream>>>(
      index_ptr, value_ptr, out_bound_ptr, out_ptr, static_cast<int>(dim_size), static_cast<int>(inner_size),
      static_cast<int>(outer_inner_size), static_cast<int>(index_num));
  } else {
    IndexFillKernel<DataType, int64_t><<<grids, blocks, 0, cuda_stream>>>(
      index_ptr, value_ptr, out_bound_ptr, out_ptr, dim_size, inner_size, outer_inner_size, index_num);
  }
}

template CUDA_LIB_EXPORT void IndexFill<bool>(bool *out_ptr, const int *index_ptr, int64_t index_size,
                                              int64_t outer_size, int64_t dim_size, int64_t inner_size,
                                              const bool *value_ptr, bool *out_bound_ptr, const uint32_t &device_id,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void IndexFill<uint8_t>(uint8_t *out_ptr, const int *index_ptr, int64_t index_size,
                                                 int64_t outer_size, int64_t dim_size, int64_t inner_size,
                                                 const uint8_t *value_ptr, bool *out_bound_ptr,
                                                 const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void IndexFill<uint16_t>(uint16_t *out_ptr, const int *index_ptr, int64_t index_size,
                                                  int64_t outer_size, int64_t dim_size, int64_t inner_size,
                                                  const uint16_t *value_ptr, bool *out_bound_ptr,
                                                  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void IndexFill<uint32_t>(uint32_t *out_ptr, const int *index_ptr, int64_t index_size,
                                                  int64_t outer_size, int64_t dim_size, int64_t inner_size,
                                                  const uint32_t *value_ptr, bool *out_bound_ptr,
                                                  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void IndexFill<uint64_t>(uint64_t *out_ptr, const int *index_ptr, int64_t index_size,
                                                  int64_t outer_size, int64_t dim_size, int64_t inner_size,
                                                  const uint64_t *value_ptr, bool *out_bound_ptr,
                                                  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void IndexFill<int8_t>(int8_t *out_ptr, const int *index_ptr, int64_t index_size,
                                                int64_t outer_size, int64_t dim_size, int64_t inner_size,
                                                const int8_t *value_ptr, bool *out_bound_ptr, const uint32_t &device_id,
                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void IndexFill<int16_t>(int16_t *out_ptr, const int *index_ptr, int64_t index_size,
                                                 int64_t outer_size, int64_t dim_size, int64_t inner_size,
                                                 const int16_t *value_ptr, bool *out_bound_ptr,
                                                 const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void IndexFill<int32_t>(int32_t *out_ptr, const int *index_ptr, int64_t index_size,
                                                 int64_t outer_size, int64_t dim_size, int64_t inner_size,
                                                 const int32_t *value_ptr, bool *out_bound_ptr,
                                                 const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void IndexFill<int64_t>(int64_t *out_ptr, const int *index_ptr, int64_t index_size,
                                                 int64_t outer_size, int64_t dim_size, int64_t inner_size,
                                                 const int64_t *value_ptr, bool *out_bound_ptr,
                                                 const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void IndexFill<half>(half *out_ptr, const int *index_ptr, int64_t index_size,
                                              int64_t outer_size, int64_t dim_size, int64_t inner_size,
                                              const half *value_ptr, bool *out_bound_ptr, const uint32_t &device_id,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void IndexFill<float>(float *out_ptr, const int *index_ptr, int64_t index_size,
                                               int64_t outer_size, int64_t dim_size, int64_t inner_size,
                                               const float *value_ptr, bool *out_bound_ptr, const uint32_t &device_id,
                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void IndexFill<double>(double *out_ptr, const int *index_ptr, int64_t index_size,
                                                int64_t outer_size, int64_t dim_size, int64_t inner_size,
                                                const double *value_ptr, bool *out_bound_ptr, const uint32_t &device_id,
                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void IndexFill<Complex<float>>(Complex<float> *out_ptr, const int *index_ptr,
                                                        int64_t index_size, int64_t outer_size, int64_t dim_size,
                                                        int64_t inner_size, const Complex<float> *value_ptr,
                                                        bool *out_bound_ptr, const uint32_t &device_id,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void IndexFill<Complex<double>>(Complex<double> *out_ptr, const int *index_ptr,
                                                         int64_t index_size, int64_t outer_size, int64_t dim_size,
                                                         int64_t inner_size, const Complex<double> *value_ptr,
                                                         bool *out_bound_ptr, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
