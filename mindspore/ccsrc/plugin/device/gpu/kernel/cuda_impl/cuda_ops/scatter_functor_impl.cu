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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/scatter_functor_impl.cuh"

template <typename T, typename S>
__global__ void ScatterUpdateKernel(S size_limit, const size_t inner_size, const size_t updates_size, const S *indices,
                                    const T *updates, T *input) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < updates_size; pos += blockDim.x * gridDim.x) {
    const size_t index = pos / inner_size;
    const size_t offset = pos % inner_size;
    if (indices[index] < 0 || indices[index] >= size_limit) {
      continue;
    }
    const size_t current_pos = indices[index] * inner_size + offset;
    input[current_pos] = updates[pos];
  }
}

template <typename T, typename S>
__global__ void ScatterAddKernel(S size_limit, const size_t inner_size, const size_t updates_size, const S *indices,
                                 const T *updates, T *input) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < updates_size; pos += blockDim.x * gridDim.x) {
    const size_t index = pos / inner_size;
    const size_t offset = pos % inner_size;
    if (indices[index] < 0 || indices[index] >= size_limit) {
      continue;
    }
    const size_t current_pos = indices[index] * inner_size + offset;
    MsAtomicAdd(&input[current_pos], updates[pos]);
  }
}

template <typename T, typename S>
__global__ void ScatterSubKernel(S size_limit, const size_t inner_size, const size_t updates_size, const S *indices,
                                 const T *updates, T *input) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < updates_size; pos += blockDim.x * gridDim.x) {
    const size_t index = pos / inner_size;
    const size_t offset = pos % inner_size;
    if (indices[index] < 0 || indices[index] >= size_limit) {
      continue;
    }
    const size_t current_pos = indices[index] * inner_size + offset;
    MsAtomicSub(&input[current_pos], updates[pos]);
  }
}

template <typename T, typename S>
__global__ void ScatterMaxKernel(S size_limit, const size_t inner_size, const size_t updates_size, const S *indices,
                                 const T *updates, T *input) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < updates_size; pos += blockDim.x * gridDim.x) {
    const size_t index = pos / inner_size;
    const size_t offset = pos % inner_size;
    if (indices[index] < 0 || indices[index] >= size_limit) {
      continue;
    }
    const size_t current_pos = indices[index] * inner_size + offset;
    MsAtomicMax(&input[current_pos], updates[pos]);
  }
}

template <typename T, typename S>
__global__ void ScatterMinKernel(S size_limit, const size_t inner_size, const size_t updates_size, const S *indices,
                                 const T *updates, T *input) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < updates_size; pos += blockDim.x * gridDim.x) {
    const size_t index = pos / inner_size;
    const size_t offset = pos % inner_size;
    if (indices[index] < 0 || indices[index] >= size_limit) {
      continue;
    }
    const size_t current_pos = indices[index] * inner_size + offset;
    MsAtomicMin(&input[current_pos], updates[pos]);
  }
}

template <typename T, typename S>
void ScatterFunc(enum ScatterFunctorType func_type, S size_limit, const size_t &inner_size, const size_t &indices_size,
                 const S *indices, const T *updates, T *input, cudaStream_t cuda_stream) {
  const size_t updates_size = inner_size * indices_size;
  switch (func_type) {
    case SCATTER_FUNC_UPDATE:
      return ScatterUpdateKernel<<<GET_BLOCKS(updates_size), GET_THREADS, 0, cuda_stream>>>(
        size_limit, inner_size, updates_size, indices, updates, input);
    case SCATTER_FUNC_ADD:
      return ScatterAddKernel<<<GET_BLOCKS(updates_size), GET_THREADS, 0, cuda_stream>>>(
        size_limit, inner_size, updates_size, indices, updates, input);
    case SCATTER_FUNC_SUB:
      return ScatterSubKernel<<<GET_BLOCKS(updates_size), GET_THREADS, 0, cuda_stream>>>(
        size_limit, inner_size, updates_size, indices, updates, input);
    case SCATTER_FUNC_MAX:
      return ScatterMaxKernel<<<GET_BLOCKS(updates_size), GET_THREADS, 0, cuda_stream>>>(
        size_limit, inner_size, updates_size, indices, updates, input);
    case SCATTER_FUNC_MIN:
      return ScatterMinKernel<<<GET_BLOCKS(updates_size), GET_THREADS, 0, cuda_stream>>>(
        size_limit, inner_size, updates_size, indices, updates, input);
    default:
      break;
  }
}

template CUDA_LIB_EXPORT void ScatterFunc<float, int>(enum ScatterFunctorType func_type, int size_limit,
                                                      const size_t &inner_size, const size_t &indices_size,
                                                      const int *indices, const float *updates, float *input,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ScatterFunc<float, int64_t>(enum ScatterFunctorType func_type, int64_t size_limit,
                                                          const size_t &inner_size, const size_t &indices_size,
                                                          const int64_t *indices, const float *updates, float *input,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ScatterFunc<half, int>(enum ScatterFunctorType func_type, int size_limit,
                                                     const size_t &inner_size, const size_t &indices_size,
                                                     const int *indices, const half *updates, half *input,
                                                     cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ScatterFunc<half, int64_t>(enum ScatterFunctorType func_type, int64_t size_limit,
                                                         const size_t &inner_size, const size_t &indices_size,
                                                         const int64_t *indices, const half *updates, half *input,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ScatterFunc<double, int>(enum ScatterFunctorType func_type, int size_limit,
                                                       const size_t &inner_size, const size_t &indices_size,
                                                       const int *indices, const double *updates, double *input,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ScatterFunc<double, int64_t>(enum ScatterFunctorType func_type, int64_t size_limit,
                                                           const size_t &inner_size, const size_t &indices_size,
                                                           const int64_t *indices, const double *updates, double *input,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ScatterFunc<int, int>(enum ScatterFunctorType func_type, int size_limit,
                                                    const size_t &inner_size, const size_t &indices_size,
                                                    const int *indices, const int *updates, int *input,
                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ScatterFunc<int, int64_t>(enum ScatterFunctorType func_type, int64_t size_limit,
                                                        const size_t &inner_size, const size_t &indices_size,
                                                        const int64_t *indices, const int *updates, int *input,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ScatterFunc<int64_t, int>(enum ScatterFunctorType func_type, int size_limit,
                                                        const size_t &inner_size, const size_t &indices_size,
                                                        const int *indices, const int64_t *updates, int64_t *input,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ScatterFunc<int64_t, int64_t>(enum ScatterFunctorType func_type, int64_t size_limit,
                                                            const size_t &inner_size, const size_t &indices_size,
                                                            const int64_t *indices, const int64_t *updates,
                                                            int64_t *input, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ScatterFunc<unsigned char, int>(enum ScatterFunctorType func_type, int size_limit,
                                                              const size_t &inner_size, const size_t &indices_size,
                                                              const int *indices, const unsigned char *updates,
                                                              unsigned char *input, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ScatterFunc<unsigned char, int64_t>(enum ScatterFunctorType func_type, int64_t size_limit,
                                                                  const size_t &inner_size, const size_t &indices_size,
                                                                  const int64_t *indices, const unsigned char *updates,
                                                                  unsigned char *input, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ScatterFunc<int8_t, int>(enum ScatterFunctorType func_type, int size_limit,
                                                       const size_t &inner_size, const size_t &indices_size,
                                                       const int *indices, const int8_t *updates, int8_t *input,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ScatterFunc<int8_t, int64_t>(enum ScatterFunctorType func_type, int64_t size_limit,
                                                           const size_t &inner_size, const size_t &indices_size,
                                                           const int64_t *indices, const int8_t *updates, int8_t *input,
                                                           cudaStream_t cuda_stream);
