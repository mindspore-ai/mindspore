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

#include <stdint.h>
#include "bias_dropout_add_impl.cuh"
#include "include/cuda_runtime.h"
#include "include/cuda_fp16.h"
#include "include/curand_kernel.h"

constexpr size_t kDropoutTileSize = 4;
template <typename T>
struct alignas(sizeof(T) * kDropoutTileSize) TArray {
  T data[kDropoutTileSize];
};

template <typename T>
__global__ void BiasDropoutAddKernel(const T *x, const T *bias, const T *residual, T *y, T *mask, size_t num_count,
                                     size_t n_strides, size_t channel_strides, float keep_prob, uint64_t seed,
                                     uint64_t seed_offset) {
  T scale = (T)(1.f / keep_prob);
  size_t inc = blockDim.x * gridDim.x * kDropoutTileSize;
  size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * kDropoutTileSize;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, seed_offset, &state);
  size_t align_count = (num_count / kDropoutTileSize) * kDropoutTileSize;

  for (size_t i = idx; i < align_count; i += inc) {
    float4 rand = curand_uniform4(&state);
    rand.x = rand.x < keep_prob;
    rand.y = rand.y < keep_prob;
    rand.z = rand.z < keep_prob;
    rand.w = rand.w < keep_prob;
    T x_tile[kDropoutTileSize];
    T y_tile[kDropoutTileSize];
    T residual_tile[kDropoutTileSize];
    T mask_tile[kDropoutTileSize];
    TArray<T> *temp = reinterpret_cast<TArray<T> *>(&x_tile);
    *temp = *reinterpret_cast<const TArray<T> *>(&x[i]);
    temp = reinterpret_cast<TArray<T> *>(&residual_tile);
    *temp = *reinterpret_cast<const TArray<T> *>(&residual[i]);
    for (size_t j = 0; j < kDropoutTileSize; ++j) {
      mask_tile[j] = (T)((&rand.x)[j]);
      y_tile[j] =
        residual_tile[j] + (x_tile[j] + bias[((i + j) % n_strides) / channel_strides]) * (T)(mask_tile[j]) * scale;
    }
    *reinterpret_cast<TArray<T> *>(&mask[i]) = *reinterpret_cast<TArray<T> *>(&mask_tile[0]);
    *reinterpret_cast<TArray<T> *>(&y[i]) = *reinterpret_cast<TArray<T> *>(&y_tile[0]);
    __syncthreads();
  }

  float4 rand = curand_uniform4(&state);
  rand.x = rand.x < keep_prob;
  rand.y = rand.y < keep_prob;
  rand.z = rand.z < keep_prob;
  for (size_t i = align_count; i < num_count; ++i) {
    mask[i] = (T)((&rand.x)[i - align_count]);
    y[i] = residual[i] + (x[i] + bias[(i % n_strides) / channel_strides]) * mask[i] * scale;
  }
}

template <typename T>
cudaError_t BiasDropoutAdd(const T *x, const T *bias, const T *residual, T *y, T *mask, size_t num_count,
                           size_t n_strides, size_t channel_strides, float drop_prob, uint64_t seed,
                           uint64_t seed_offset, cudaStream_t cuda_stream) {
  BiasDropoutAddKernel<<<GET_BLOCKS(num_count), GET_THREADS, 0, cuda_stream>>>(
    x, bias, residual, y, mask, num_count, n_strides, channel_strides, drop_prob, seed, seed_offset);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t BiasDropoutAdd<float>(const float *x, const float *bias, const float *residual,
                                                           float *y, float *mask, size_t num_count, size_t n_strides,
                                                           size_t channel_strides, float drop_prob, uint64_t seed,
                                                           uint64_t seed_offset, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t BiasDropoutAdd<half>(const half *x, const half *bias, const half *residual,
                                                          half *y, half *mask, size_t num_count, size_t n_strides,
                                                          size_t channel_strides, float drop_prob, uint64_t seed,
                                                          uint64_t seed_offset, cudaStream_t cuda_stream);
