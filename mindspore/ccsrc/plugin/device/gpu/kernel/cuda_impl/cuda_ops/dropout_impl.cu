/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "dropout_impl.cuh"
#include "include/cuda_runtime.h"
#include "include/cuda_fp16.h"
#include "include/curand_kernel.h"
template <typename T>
__global__ void DropoutForwardKernel(const T *input, T *mask, T *output, float *mask_f, size_t num_count,
                                     float keep_prob) {
  T scale = (T)(1.f / keep_prob);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_count; i += blockDim.x * gridDim.x) {
    mask[i] = mask_f[i] <= keep_prob;
    output[i] = scale * input[i] * (T)(mask[i]);
  }
}

template <typename T>
cudaError_t DropoutForward(const T *input, T *mask, T *output, float *mask_f, size_t num_count, float drop_prob,
                           cudaStream_t cuda_stream) {
  DropoutForwardKernel<<<GET_BLOCKS(num_count), GET_THREADS, 0, cuda_stream>>>(input, mask, output, mask_f, num_count,
                                                                               drop_prob);
  return GetCudaStatus();
}
template <typename T>
__global__ void DropoutBackwardKernel(const T *dy, const T *mask, T *dx, size_t num_count, float keep_prob) {
  T scale = T(1.f / keep_prob);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_count; i += blockDim.x * gridDim.x) {
    dx[i] = scale * dy[i] * (T)(mask[i]);
  }
}

template <typename T>
cudaError_t DropoutBackward(const T *dy, const T *mask, T *dx, size_t num_count, float drop_prob,
                            cudaStream_t cuda_stream) {
  DropoutBackwardKernel<<<GET_BLOCKS(num_count), GET_THREADS, 0, cuda_stream>>>(dy, mask, dx, num_count, drop_prob);
  return GetCudaStatus();
}

template <typename T>
struct alignas(sizeof(T) * kDropoutTileSize) TArray {
  T data[kDropoutTileSize];
};

template <typename T>
__global__ void FusedDropoutForwardKernel(const T *input, T *mask, T *output, size_t num_count, float keep_prob,
                                          uint64_t seed, uint64_t seed_offset) {
  T scale = (T)(1.f / keep_prob);
  size_t inc = blockDim.x * gridDim.x * kDropoutTileSize;
  size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * kDropoutTileSize;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, seed_offset, &state);
  for (size_t i = idx; i < num_count; i += inc) {
    float4 rand = curand_uniform4(&state);
    rand.x = rand.x < keep_prob;
    rand.y = rand.y < keep_prob;
    rand.z = rand.z < keep_prob;
    rand.w = rand.w < keep_prob;
    T input_tile[kDropoutTileSize];
    T output_tile[kDropoutTileSize];
    T mask_tile[kDropoutTileSize];
    TArray<T> *temp = reinterpret_cast<TArray<T> *>(&input_tile);
    *temp = *reinterpret_cast<const TArray<T> *>(&input[i]);
    for (size_t j = 0; j < kDropoutTileSize; ++j) {
      mask_tile[j] = (T)((&rand.x)[j]);
      output_tile[j] = input_tile[j] * (T)(mask_tile[j]) * scale;
    }
    *reinterpret_cast<TArray<T> *>(&mask[i]) = *reinterpret_cast<TArray<T> *>(&mask_tile[0]);
    *reinterpret_cast<TArray<T> *>(&output[i]) = *reinterpret_cast<TArray<T> *>(&output_tile[0]);
    __syncthreads();
  }
}

template <typename T>
__global__ void FusedDropoutForwardOnlyMaskKernel(T *mask, size_t num_count, float keep_prob, uint64_t seed,
                                                  uint64_t seed_offset) {
  T scale = (T)(1.f / keep_prob);
  size_t inc = blockDim.x * gridDim.x * kDropoutTileSize;
  size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * kDropoutTileSize;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, seed_offset, &state);
  for (size_t i = idx; i < num_count; i += inc) {
    float4 rand = curand_uniform4(&state);
    rand.x = rand.x < keep_prob;
    rand.y = rand.y < keep_prob;
    rand.z = rand.z < keep_prob;
    rand.w = rand.w < keep_prob;
    T mask_tile[kDropoutTileSize];
    for (size_t j = 0; j < kDropoutTileSize; ++j) {
      mask_tile[j] = (T)((&rand.x)[j]);
    }
    *reinterpret_cast<TArray<T> *>(&mask[i]) = *reinterpret_cast<TArray<T> *>(&mask_tile[0]);
    __syncthreads();
  }
}

template <typename T>
__global__ void FusedDropoutOnlyOutputKernel(const T *input, T *output, size_t num_count, float keep_prob,
                                             uint64_t seed, uint64_t seed_offset) {
  T scale = (T)(1.f / keep_prob);
  size_t inc = blockDim.x * gridDim.x * kDropoutTileSize;
  size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * kDropoutTileSize;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, seed_offset, &state);
  for (size_t i = idx; i < num_count; i += inc) {
    float4 rand = curand_uniform4(&state);
    rand.x = rand.x < keep_prob;
    rand.y = rand.y < keep_prob;
    rand.z = rand.z < keep_prob;
    rand.w = rand.w < keep_prob;
    T input_tile[kDropoutTileSize];
    T output_tile[kDropoutTileSize];
    TArray<T> *temp = reinterpret_cast<TArray<T> *>(&input_tile);
    *temp = *reinterpret_cast<const TArray<T> *>(&input[i]);
    for (size_t j = 0; j < kDropoutTileSize; ++j) {
      output_tile[j] = input_tile[j] * (T)((&rand.x)[j]) * scale;
    }
    *reinterpret_cast<TArray<T> *>(&output[i]) = *reinterpret_cast<TArray<T> *>(&output_tile[0]);
    __syncthreads();
  }
}

template <typename T>
cudaError_t FusedDropoutForward(const T *input, T *mask, T *output, size_t num_count, float drop_prob, uint64_t seed,
                                uint64_t seed_offset, cudaStream_t cuda_stream) {
  FusedDropoutForwardKernel<<<GET_BLOCKS(num_count), GET_THREADS, 0, cuda_stream>>>(input, mask, output, num_count,
                                                                                    drop_prob, seed, seed_offset);
  return GetCudaStatus();
}

template <typename T>
cudaError_t FusedDropoutForwardOnlyMask(T *mask, size_t num_count, float drop_prob, uint64_t seed, uint64_t seed_offset,
                                        cudaStream_t cuda_stream) {
  FusedDropoutForwardOnlyMaskKernel<<<GET_BLOCKS(num_count), GET_THREADS, 0, cuda_stream>>>(mask, num_count, drop_prob,
                                                                                            seed, seed_offset);
  return GetCudaStatus();
}

template <typename T>
cudaError_t FusedDropoutForwardOnlyOutput(const T *input, T *output, size_t num_count, float drop_prob, uint64_t seed,
                                          uint64_t seed_offset, cudaStream_t cuda_stream) {
  FusedDropoutOnlyOutputKernel<<<GET_BLOCKS(num_count), GET_THREADS, 0, cuda_stream>>>(input, output, num_count,
                                                                                       drop_prob, seed, seed_offset);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t DropoutForward<float>(const float *input, float *mask, float *output,
                                                           float *mask_f, size_t num_count, float drop_prob,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t DropoutForward<half>(const half *input, half *mask, half *output, float *mask_f,
                                                          size_t num_count, float drop_prob, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t DropoutForward<double>(const double *input, double *mask, double *output,
                                                            float *mask_f, size_t num_count, float drop_prob,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t DropoutBackward<float>(const float *dy, const float *mask, float *dx,
                                                            size_t num_count, float drop_prob,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t DropoutBackward<half>(const half *dy, const half *mask, half *dx, size_t num_count,
                                                           float drop_prob, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t DropoutBackward<double>(const double *dy, const double *mask, double *dx,
                                                             size_t num_count, float drop_prob,
                                                             cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t FusedDropoutForward<float>(const float *input, float *mask, float *output,
                                                                size_t num_count, float drop_prob, uint64_t seed,
                                                                uint64_t seed_offset, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t FusedDropoutForward<half>(const half *input, half *mask, half *output,
                                                               size_t num_count, float drop_prob, uint64_t seed,
                                                               uint64_t seed_offset, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t FusedDropoutForward<double>(const double *input, double *mask, double *output,
                                                                 size_t num_count, float drop_prob, uint64_t seed,
                                                                 uint64_t seed_offset, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t FusedDropoutForwardOnlyMask<float>(float *mask, size_t num_count, float drop_prob,
                                                                        uint64_t seed, uint64_t seed_offset,
                                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t FusedDropoutForwardOnlyMask<half>(half *mask, size_t num_count, float drop_prob,
                                                                       uint64_t seed, uint64_t seed_offset,
                                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t FusedDropoutForwardOnlyMask<double>(double *mask, size_t num_count,
                                                                         float drop_prob, uint64_t seed,
                                                                         uint64_t seed_offset,
                                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t FusedDropoutForwardOnlyOutput<float>(const float *input, float *output,
                                                                          size_t num_count, float drop_prob,
                                                                          uint64_t seed, uint64_t seed_offset,
                                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t FusedDropoutForwardOnlyOutput<half>(const half *input, half *output,
                                                                         size_t num_count, float drop_prob,
                                                                         uint64_t seed, uint64_t seed_offset,
                                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t FusedDropoutForwardOnlyOutput<double>(const double *input, double *output,
                                                                           size_t num_count, float drop_prob,
                                                                           uint64_t seed, uint64_t seed_offset,
                                                                           cudaStream_t cuda_stream);
