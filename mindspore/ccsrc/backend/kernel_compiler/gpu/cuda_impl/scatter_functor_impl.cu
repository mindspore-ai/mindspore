/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "backend/kernel_compiler/gpu/cuda_impl/scatter_functor_impl.cuh"

template <typename T, typename S>
__global__ void ScatterUpdateKernel(const size_t inner_size, const size_t updates_size, const S *indices,
                                    const T *updates, T *input) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < updates_size; pos += blockDim.x * gridDim.x) {
    const size_t index = pos / inner_size;
    const size_t offset = pos % inner_size;
    const size_t current_pos = indices[index] * inner_size + offset;
    input[current_pos] = updates[pos];
  }
}

template <typename T, typename S>
__global__ void ScatterAddKernel(const size_t inner_size, const size_t updates_size, const S *indices, const T *updates,
                                 T *input) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < updates_size; pos += blockDim.x * gridDim.x) {
    const size_t index = pos / inner_size;
    const size_t offset = pos % inner_size;
    const size_t current_pos = indices[index] * inner_size + offset;
    MsAtomicAdd(&input[current_pos], updates[pos]);
  }
}

template <typename T, typename S>
__global__ void ScatterSubKernel(const size_t inner_size, const size_t updates_size, const S *indices, const T *updates,
                                 T *input) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < updates_size; pos += blockDim.x * gridDim.x) {
    const size_t index = pos / inner_size;
    const size_t offset = pos % inner_size;
    const size_t current_pos = indices[index] * inner_size + offset;
    MsAtomicAdd(&input[current_pos], -updates[pos]);
  }
}

template <typename T, typename S>
void ScatterFunc(enum ScatterFunctorType func_type, const size_t &inner_size, const size_t &indices_size,
                 const S *indices, const T *updates, T *input, cudaStream_t cuda_stream) {
  const size_t updates_size = inner_size * indices_size;
  switch (func_type) {
    case SCATTER_FUNC_UPDATE:
      return ScatterUpdateKernel<<<GET_BLOCKS(updates_size), GET_THREADS, 0, cuda_stream>>>(inner_size, updates_size,
                                                                                            indices, updates, input);
    case SCATTER_FUNC_ADD:
      return ScatterAddKernel<<<GET_BLOCKS(updates_size), GET_THREADS, 0, cuda_stream>>>(inner_size, updates_size,
                                                                                         indices, updates, input);
    case SCATTER_FUNC_SUB:
      return ScatterSubKernel<<<GET_BLOCKS(updates_size), GET_THREADS, 0, cuda_stream>>>(inner_size, updates_size,
                                                                                         indices, updates, input);
    default:
      break;
  }
}

template void ScatterFunc<float, int>(enum ScatterFunctorType func_type, const size_t &inner_size,
                                      const size_t &indices_size, const int *indices, const float *updates,
                                      float *input, cudaStream_t cuda_stream);
template void ScatterFunc<float, int64_t>(enum ScatterFunctorType func_type, const size_t &inner_size,
                                          const size_t &indices_size, const int64_t *indices, const float *updates,
                                          float *input, cudaStream_t cuda_stream);
template void ScatterFunc<half, int>(enum ScatterFunctorType func_type, const size_t &inner_size,
                                     const size_t &indices_size, const int *indices, const half *updates, half *input,
                                     cudaStream_t cuda_stream);
template void ScatterFunc<half, int64_t>(enum ScatterFunctorType func_type, const size_t &inner_size,
                                         const size_t &indices_size, const int64_t *indices, const half *updates,
                                         half *input, cudaStream_t cuda_stream);
template void ScatterFunc<int, int>(enum ScatterFunctorType func_type, const size_t &inner_size,
                                    const size_t &indices_size, const int *indices, const int *updates, int *input,
                                    cudaStream_t cuda_stream);
template void ScatterFunc<int, int64_t>(enum ScatterFunctorType func_type, const size_t &inner_size,
                                        const size_t &indices_size, const int64_t *indices, const int *updates,
                                        int *input, cudaStream_t cuda_stream);
template void ScatterFunc<unsigned char, int>(enum ScatterFunctorType func_type, const size_t &inner_size,
                                              const size_t &indices_size, const int *indices,
                                              const unsigned char *updates, unsigned char *input,
                                              cudaStream_t cuda_stream);
template void ScatterFunc<unsigned char, int64_t>(enum ScatterFunctorType func_type, const size_t &inner_size,
                                                  const size_t &indices_size, const int64_t *indices,
                                                  const unsigned char *updates, unsigned char *input,
                                                  cudaStream_t cuda_stream);
template void ScatterFunc<int8_t, int>(enum ScatterFunctorType func_type, const size_t &inner_size,
                                       const size_t &indices_size, const int *indices, const int8_t *updates,
                                       int8_t *input, cudaStream_t cuda_stream);
template void ScatterFunc<int8_t, int64_t>(enum ScatterFunctorType func_type, const size_t &inner_size,
                                           const size_t &indices_size, const int64_t *indices, const int8_t *updates,
                                           int8_t *input, cudaStream_t cuda_stream);
