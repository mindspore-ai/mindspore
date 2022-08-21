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

#include "random_shuffle_impl.cuh"
#include <cuda_runtime.h>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "include/cuda_fp16.h"

template <typename T>
using Complex = mindspore::utils::Complex<T>;

#define SHUFFLE_DECLARE(type)                                                                                          \
  template CUDA_LIB_EXPORT void ScalarShuffle<type>(const int64_t size, const int *perm, const type *input,            \
                                                    type *output, const uint32_t device_id, cudaStream_t cuda_stream); \
  template CUDA_LIB_EXPORT void TensorShuffle<type>(const int64_t shuffle_size, const int64_t inner_size,              \
                                                    const int *perm, const type *input, type *output,                  \
                                                    const uint32_t device_id, cudaStream_t cuda_stream);

template <typename T>
__global__ void ScalarShuffleKernel(const int64_t size, const int *perm, const T *input, T *output) {
  for (int64_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = input[perm[pos]];
  }
}

template <typename T>
__global__ void TensorShuffleKernel(const int64_t shuffle_size, const int64_t inner_size, const int *perm,
                                    const T *input, T *output) {
  for (int64_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < shuffle_size * inner_size;
       pos += blockDim.x * gridDim.x) {
    int64_t row = pos / inner_size;
    int64_t col = pos % inner_size;
    int64_t output_offset = perm[row] * inner_size + col;
    output[output_offset] = input[pos];
  }
}

template <typename T>
void ScalarShuffle(const int64_t size, const int *perm, const T *input, T *output, const uint32_t device_id,
                   cudaStream_t cuda_stream) {
  ScalarShuffleKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, perm, input,
                                                                                                 output);
}

template <typename T>
void TensorShuffle(const int64_t shuffle_size, const int64_t inner_size, const int *perm, const T *input, T *output,
                   const uint32_t device_id, cudaStream_t cuda_stream) {
  int64_t total_size = shuffle_size * inner_size;
  TensorShuffleKernel<<<CUDA_BLOCKS(device_id, total_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    shuffle_size, inner_size, perm, input, output);
}

SHUFFLE_DECLARE(half);
SHUFFLE_DECLARE(float);
SHUFFLE_DECLARE(double);
SHUFFLE_DECLARE(int8_t);
SHUFFLE_DECLARE(int16_t);
SHUFFLE_DECLARE(int32_t);
SHUFFLE_DECLARE(int64_t);
SHUFFLE_DECLARE(uint8_t);
SHUFFLE_DECLARE(uint16_t);
SHUFFLE_DECLARE(uint32_t);
SHUFFLE_DECLARE(uint64_t);
SHUFFLE_DECLARE(bool);
SHUFFLE_DECLARE(Complex<float>);
SHUFFLE_DECLARE(Complex<double>);
