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

#include "hash_impl.cuh"
#include "runtime/device/gpu/cuda_common.h"

template <typename T>
__global__ void HashSwapOut(const T *hash_table, T *swap_out_value, const int *swap_out_index, const int index_size,
                            const int hash_dim) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < index_size; i += blockDim.x * gridDim.x) {
    int hash_index = swap_out_index[i];
    for (int j = 0; j < hash_dim; j++) {
      swap_out_value[i * hash_dim + j] = hash_table[hash_index * hash_dim + j];
    }
  }
  return;
}

template <typename T>
__global__ void HashSwapIn(T *hash_table, const T *swap_in_value, const int *swap_in_index, const int index_size,
                           const int hash_dim) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < index_size; i += blockDim.x * gridDim.x) {
    int hash_index = swap_in_index[i];
    for (int j = 0; j < hash_dim; j++) {
      hash_table[hash_index * hash_dim + j] = swap_in_value[i * hash_dim + j];
    }
  }
  return;
}

template <typename T>
void DoHashSwapOut(const T *hash_table, T *swap_out_value, const int *swap_out_index, const int index_size,
                   const int hash_dim, cudaStream_t cuda_stream) {
  HashSwapOut<<<GET_BLOCKS(index_size), GET_THREADS, 0, cuda_stream>>>(hash_table, swap_out_value, swap_out_index,
                                                                       index_size, hash_dim);
  return;
}

template <typename T>
void DoHashSwapIn(T *hash_table, const T *swap_in_value, const int *swap_in_index, const int index_size,
                  const int hash_dim, cudaStream_t cuda_stream) {
  HashSwapIn<<<GET_BLOCKS(index_size), GET_THREADS, 0, cuda_stream>>>(hash_table, swap_in_value, swap_in_index,
                                                                      index_size, hash_dim);
  return;
}

template void DoHashSwapOut<float>(const float *hash_table, float *swap_out_value, const int *swap_out_index,
                                   const int index_size, const int hash_dim, cudaStream_t cuda_stream);

template void DoHashSwapIn<float>(float *hash_table, const float *swap_in_value, const int *swap_in_index,
                                  const int index_size, const int hash_dim, cudaStream_t cuda_stream);
