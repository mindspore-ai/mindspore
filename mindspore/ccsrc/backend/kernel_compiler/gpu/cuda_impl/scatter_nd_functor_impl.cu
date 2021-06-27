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

#include "backend/kernel_compiler/gpu/cuda_impl/util.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/scatter_nd_functor_impl.cuh"

template <typename T, typename S>
__global__ void ScatterNdUpdate(const size_t unit_size, const size_t index_depth, const size_t updates_size,
                                const S *out_strides, const S *indices, const T *updates, T *input) {
  int i, j;
  for (size_t read_index = blockIdx.x * blockDim.x + threadIdx.x; read_index < (updates_size);
       read_index += blockDim.x * gridDim.x) {
    size_t write_index = 0;
    bool out_bound = false;

    i = read_index / unit_size;
    j = read_index % unit_size;

    for (size_t k = 0; k < index_depth; k++) {
      S indices_i = indices[i * index_depth + k];
      out_bound |= indices_i < 0;
      write_index += indices_i * out_strides[k] * unit_size;
    }

    write_index += j;

    if (!out_bound) {
      input[write_index] = updates[read_index];
    }
  }
}

template <typename T, typename S>
__global__ void ScatterNdAdd(const size_t unit_size, const size_t index_depth, const size_t updates_size,
                             const S *out_strides, const S *indices, const T *updates, T *input) {
  int i, j;
  for (size_t read_index = blockIdx.x * blockDim.x + threadIdx.x; read_index < (updates_size);
       read_index += blockDim.x * gridDim.x) {
    size_t write_index = 0;
    bool out_bound = false;

    i = read_index / unit_size;
    j = read_index % unit_size;

    for (size_t k = 0; k < index_depth; k++) {
      S indices_i = indices[i * index_depth + k];
      out_bound |= indices_i < 0;
      write_index += indices_i * out_strides[k] * unit_size;
    }

    write_index += j;

    if (!out_bound) {
      MsAtomicAdd(&input[write_index], updates[read_index]);
    }
  }
}

template <typename T, typename S>
__global__ void ScatterNdSub(const size_t unit_size, const size_t index_depth, const size_t updates_size,
                             const S *out_strides, const S *indices, const T *updates, T *input) {
  int i, j;
  for (size_t read_index = blockIdx.x * blockDim.x + threadIdx.x; read_index < (updates_size);
       read_index += blockDim.x * gridDim.x) {
    size_t write_index = 0;
    bool out_bound = false;

    i = read_index / unit_size;
    j = read_index % unit_size;

    for (size_t k = 0; k < index_depth; k++) {
      S indices_i = indices[i * index_depth + k];
      out_bound |= indices_i < 0;
      write_index += indices_i * out_strides[k] * unit_size;
    }

    write_index += j;

    if (!out_bound) {
      MsAtomicAdd(&input[write_index], -updates[read_index]);
    }
  }
}

template <typename T, typename S>
void CalScatterNdFunctor(enum ScatterNdFunctorType func_type, const size_t &unit_size, const size_t &num_units,
                         const size_t &index_depth, const S *out_strides, const S *indices, const T *updates, T *input,
                         cudaStream_t cuda_stream) {
  const size_t updates_size = unit_size * num_units;
  switch (func_type) {
    case SCATTER_ND_FUNC_UPDATE:
      return ScatterNdUpdate<<<GET_BLOCKS(updates_size), GET_THREADS, 0, cuda_stream>>>(
        unit_size, index_depth, updates_size, out_strides, indices, updates, input);
    case SCATTER_ND_FUNC_ADD:
      return ScatterNdAdd<<<GET_BLOCKS(updates_size), GET_THREADS, 0, cuda_stream>>>(
        unit_size, index_depth, updates_size, out_strides, indices, updates, input);
    case SCATTER_ND_FUNC_SUB:
      return ScatterNdSub<<<GET_BLOCKS(updates_size), GET_THREADS, 0, cuda_stream>>>(
        unit_size, index_depth, updates_size, out_strides, indices, updates, input);
    default:
      break;
  }
}

template void CalScatterNdFunctor<double, int64_t>(enum ScatterNdFunctorType func_type, const size_t &unit_size,
                                                   const size_t &num_units, const size_t &index_depth,
                                                   const int64_t *out_strides, const int64_t *indices,
                                                   const double *updates, double *input, cudaStream_t cuda_stream);
template void CalScatterNdFunctor<double, int32_t>(enum ScatterNdFunctorType func_type, const size_t &unit_size,
                                                   const size_t &num_units, const size_t &index_depth,
                                                   const int32_t *out_strides, const int32_t *indices,
                                                   const double *updates, double *input, cudaStream_t cuda_stream);
template void CalScatterNdFunctor<float, int64_t>(enum ScatterNdFunctorType func_type, const size_t &unit_size,
                                                  const size_t &num_units, const size_t &index_depth,
                                                  const int64_t *out_strides, const int64_t *indices,
                                                  const float *updates, float *input, cudaStream_t cuda_stream);
template void CalScatterNdFunctor<float, int32_t>(enum ScatterNdFunctorType func_type, const size_t &unit_size,
                                                  const size_t &num_units, const size_t &index_depth,
                                                  const int32_t *out_strides, const int32_t *indices,
                                                  const float *updates, float *input, cudaStream_t cuda_stream);
template void CalScatterNdFunctor<half, int64_t>(enum ScatterNdFunctorType func_type, const size_t &unit_size,
                                                 const size_t &num_units, const size_t &index_depth,
                                                 const int64_t *out_strides, const int64_t *indices,
                                                 const half *updates, half *input, cudaStream_t cuda_stream);
template void CalScatterNdFunctor<half, int32_t>(enum ScatterNdFunctorType func_type, const size_t &unit_size,
                                                 const size_t &num_units, const size_t &index_depth,
                                                 const int32_t *out_strides, const int32_t *indices,
                                                 const half *updates, half *input, cudaStream_t cuda_stream);
template void CalScatterNdFunctor<int32_t, int64_t>(enum ScatterNdFunctorType func_type, const size_t &unit_size,
                                                    const size_t &num_units, const size_t &index_depth,
                                                    const int64_t *out_strides, const int64_t *indices,
                                                    const int32_t *updates, int32_t *input, cudaStream_t cuda_stream);
template void CalScatterNdFunctor<int32_t, int32_t>(enum ScatterNdFunctorType func_type, const size_t &unit_size,
                                                    const size_t &num_units, const size_t &index_depth,
                                                    const int32_t *out_strides, const int32_t *indices,
                                                    const int32_t *updates, int32_t *input, cudaStream_t cuda_stream);
template void CalScatterNdFunctor<int16_t, int64_t>(enum ScatterNdFunctorType func_type, const size_t &unit_size,
                                                    const size_t &num_units, const size_t &index_depth,
                                                    const int64_t *out_strides, const int64_t *indices,
                                                    const int16_t *updates, int16_t *input, cudaStream_t cuda_stream);
template void CalScatterNdFunctor<int16_t, int32_t>(enum ScatterNdFunctorType func_type, const size_t &unit_size,
                                                    const size_t &num_units, const size_t &index_depth,
                                                    const int32_t *out_strides, const int32_t *indices,
                                                    const int16_t *updates, int16_t *input, cudaStream_t cuda_stream);
template void CalScatterNdFunctor<uint8_t, int64_t>(enum ScatterNdFunctorType func_type, const size_t &unit_size,
                                                    const size_t &num_units, const size_t &index_depth,
                                                    const int64_t *out_strides, const int64_t *indices,
                                                    const uint8_t *updates, uint8_t *input, cudaStream_t cuda_stream);
template void CalScatterNdFunctor<uint8_t, int32_t>(enum ScatterNdFunctorType func_type, const size_t &unit_size,
                                                    const size_t &num_units, const size_t &index_depth,
                                                    const int32_t *out_strides, const int32_t *indices,
                                                    const uint8_t *updates, uint8_t *input, cudaStream_t cuda_stream);
template void CalScatterNdFunctor<int8_t, int64_t>(enum ScatterNdFunctorType func_type, const size_t &unit_size,
                                                   const size_t &num_units, const size_t &index_depth,
                                                   const int64_t *out_strides, const int64_t *indices,
                                                   const int8_t *updates, int8_t *input, cudaStream_t cuda_stream);
template void CalScatterNdFunctor<int8_t, int32_t>(enum ScatterNdFunctorType func_type, const size_t &unit_size,
                                                   const size_t &num_units, const size_t &index_depth,
                                                   const int32_t *out_strides, const int32_t *indices,
                                                   const int8_t *updates, int8_t *input, cudaStream_t cuda_stream);
template void CalScatterNdFunctor<bool, int64_t>(enum ScatterNdFunctorType func_type, const size_t &unit_size,
                                                 const size_t &num_units, const size_t &index_depth,
                                                 const int64_t *out_strides, const int64_t *indices,
                                                 const bool *updates, bool *input, cudaStream_t cuda_stream);
template void CalScatterNdFunctor<bool, int32_t>(enum ScatterNdFunctorType func_type, const size_t &unit_size,
                                                 const size_t &num_units, const size_t &index_depth,
                                                 const int32_t *out_strides, const int32_t *indices,
                                                 const bool *updates, bool *input, cudaStream_t cuda_stream);
