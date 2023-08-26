/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include <curand_kernel.h>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/uniform_impl.cuh"

template <typename T>
__global__ void Uniform(T *x, T *y, const float from, const float to, uint64_t seed, uint64_t seed_offset,
                        const size_t size) {
  curandStatePhilox4_32_10_t state;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    curand_init(seed, pos, seed_offset, &state);
    y[pos] = curand_uniform_double(&state) * (to - from) + from;
  }
}

template <>
__global__ void Uniform(float *x, float *y, const float from, const float to, uint64_t seed, uint64_t seed_offset,
                        const size_t size) {
  curandStatePhilox4_32_10_t state;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    curand_init(seed, pos, seed_offset, &state);
    y[pos] = curand_uniform(&state) * (to - from) + from;
  }
}

template <>
__global__ void Uniform(half *x, half *y, const float from, const float to, uint64_t seed, uint64_t seed_offset,
                        const size_t size) {
  curandStatePhilox4_32_10_t state;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    curand_init(seed, pos, seed_offset, &state);
    y[pos] = __float2half(curand_uniform(&state) * (to - from) + from);
  }
}

template <typename T>
cudaError_t CalUniform(T *x, T *y, float from, float to, uint64_t seed, uint64_t seed_offset, const size_t size,
                       const uint32_t &device_id, cudaStream_t cuda_stream) {
  Uniform<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(x, y, from, to, seed, seed_offset,
                                                                                     size);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalUniform<double>(double *x, double *y, const float from, const float to,
                                                        uint64_t seed, uint64_t seed_offset, const size_t size,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalUniform<float>(float *x, float *y, const float from, const float to,
                                                       uint64_t seed, uint64_t seed_offset, const size_t size,
                                                       const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalUniform<half>(half *x, half *y, const float from, const float to, uint64_t seed,
                                                      uint64_t seed_offset, const size_t size,
                                                      const uint32_t &device_id, cudaStream_t cuda_stream);
