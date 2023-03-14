/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "src/extendrt/delegate/tensorrt/cuda_impl/fse_decode.cuh"
#include <cuda_fp16.h>
#include <stdio.h>
#include <inttypes.h>
#include "plugin/device/cpu/kernel/nnacl/op_base.h"

__device__ __forceinline__ uint64_t Pop(const uint64_t *chunks, uint64_t *curr_chunk, uint8_t bit_count,
                                        int32_t *curr_bit_count, int32_t *curr_chunk_index) {
  const int kMaxBitCount = 64;
  uint64_t right = *curr_chunk >> static_cast<size_t>(kMaxBitCount - *curr_bit_count);
  uint64_t res = right & ((1u << bit_count) - 1);
  *curr_bit_count -= static_cast<int32_t>(bit_count);
  if (*curr_bit_count > 0) {
    return res;
  }
  if (*curr_bit_count == 0) {
    if (*curr_chunk_index > -1) {
      *curr_bit_count = kMaxBitCount;
      *curr_chunk = chunks[(*curr_chunk_index)--];
    }
    return res;
  }
  *curr_bit_count += static_cast<int32_t>(bit_count);
  *curr_chunk = chunks[(*curr_chunk_index)--];
  right |= (*curr_chunk & ((1u << (static_cast<int8_t>(bit_count) - *curr_bit_count)) - 1)) << *curr_bit_count;
  *curr_bit_count = kMaxBitCount - (static_cast<int8_t>(bit_count) - *curr_bit_count);
  return right;
}

template <typename T>
__global__ void FSE_Decode_kernel(const uint64_t *chunks, const uint16_t *states_table, const uint8_t *bit_count_table,
                                  const uint16_t *symbol_table, const uint64_t *ptable, int ptable_size,
                                  const T *centroids, uint64_t out_size, T *output, const uint64_t current_chunk_input,
                                  bool use_curr_chunk) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ptable_size) {
    return;
  }
  int32_t curr_chunk_index = static_cast<int32_t>(ptable[idx] >> 32);
  uint16_t state = static_cast<uint16_t>((ptable[idx] >> 16) & 0xffff);
  int32_t curr_bit_count = static_cast<int32_t>(ptable[idx] & 0xffff);
  if (curr_bit_count == 0) {
    curr_chunk_index--;
    curr_bit_count = 64;
  }
  uint64_t curr_chunk =
    ((idx == ptable_size - 1) && (use_curr_chunk)) ? current_chunk_input : chunks[curr_chunk_index + 1];
  const uint64_t output_offset = out_size * idx / ptable_size;
  uint64_t out_count = out_size * (idx + 1) / ptable_size - output_offset;
  T *output_ptr = output + output_offset;
  while ((curr_chunk_index >= 0) || (bit_count_table[state] == 0) || (curr_bit_count > 0)) {
    if (out_count == 0) {
      break;
    }
    output_ptr[--out_count] = centroids[symbol_table[state]];
    state = states_table[state] + Pop(chunks, &curr_chunk, bit_count_table[state], &curr_bit_count, &curr_chunk_index);
  }
}

template <typename T>
void FSE_Decode(const uint64_t *chunks, const uint16_t *states_table, const uint8_t *bit_count_table,
                const uint16_t *symbol_table, const uint64_t *ptable, int ptable_size, const T *centroids,
                uint64_t out_size, T *output, const uint32_t &device_id, uint64_t current_chunk_input,
                bool use_curr_chunk, cudaStream_t cuda_stream) {
  const int kThreads = 256;
  const int kBlocks = UP_DIV(ptable_size, kThreads);
  FSE_Decode_kernel<<<kBlocks, kThreads, 0, cuda_stream>>>(chunks, states_table, bit_count_table, symbol_table, ptable,
                                                           ptable_size, centroids, out_size, output,
                                                           current_chunk_input, use_curr_chunk);
}

template void FSE_Decode(const uint64_t *chunks, const uint16_t *states_table, const uint8_t *bit_count_table,
                         const uint16_t *symbol_table, const uint64_t *ptable, int ptable_size, const float *centroids,
                         uint64_t out_size, float *output, const uint32_t &device_id, uint64_t current_chunk_input,
                         bool use_curr_chunk, cudaStream_t cuda_stream);

template void FSE_Decode(const uint64_t *chunks, const uint16_t *states_table, const uint8_t *bit_count_table,
                         const uint16_t *symbol_table, const uint64_t *ptable, int ptable_size, const half *centroids,
                         uint64_t out_size, half *output, const uint32_t &device_id, uint64_t current_chunk_input,
                         bool use_curr_chunk, cudaStream_t cuda_stream);
