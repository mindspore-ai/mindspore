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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/quant_impl.cuh"
#include <stdio.h>
#include <math.h>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"

__global__ void DeQuantWithPerLayer(const int8_t *input, float *output, int element_cnt, float scale, int zp) {
  // dequnt = (x - zp) * scale
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < element_cnt; pos += blockDim.x * gridDim.x) {
    output[pos] = (input[pos] - zp) * scale;
  }
}

__global__ void DeQuantWithPerChannel(const int8_t *input, float *output, int element_cnt, const float *scale,
                                      const int *zp, size_t stride, size_t bucket_count) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < element_cnt; pos += blockDim.x * gridDim.x) {
    size_t bucket_index = (pos / stride) % bucket_count;
    // dequnt = (x - zp) * scale
    output[pos] = (input[pos] - zp[bucket_index]) * scale[bucket_index];
  }
}

__device__ uint64_t Pop(BitStreamState *bs, const uint64_t *chunks, uint8_t bit_count) {
  const int kMaxBitCount = 64;
  uint64_t right = bs->curr_chunk >> static_cast<size_t>(kMaxBitCount - bs->curr_bit_count);
  uint64_t res = right & ((1u << bit_count) - 1);
  bs->curr_bit_count -= static_cast<int8_t>(bit_count);
  if (bs->curr_bit_count > 0) {
    return res;
  }
  if (bs->curr_bit_count == 0) {
    if (bs->curr_chunk_index > -1) {
      bs->curr_bit_count = kMaxBitCount;
      bs->curr_chunk = chunks[bs->curr_chunk_index--];
    }
    return res;
  }
  bs->curr_bit_count += static_cast<int8_t>(bit_count);
  bs->curr_chunk = chunks[bs->curr_chunk_index--];
  right |= (bs->curr_chunk & ((1u << (static_cast<int8_t>(bit_count) - bs->curr_bit_count)) - 1)) << bs->curr_bit_count;
  bs->curr_bit_count = kMaxBitCount - (static_cast<int8_t>(bit_count) - bs->curr_bit_count);
  return right;
}

__global__ void FSEDeCompressed(BitStreamState *bs, const uint64_t *chunks, float *out, int out_count,
                                const uint16_t *states_table, const uint8_t *bit_count_table,
                                const uint16_t *symbol_table, const float *centroids, size_t table_log) {
  uint64_t state = Pop(bs, chunks, table_log);
  while ((bs->curr_chunk_index >= 0) || (bit_count_table[state] == 0) || (bs->curr_bit_count > 0)) {
    if (out_count == 0) {
      return;
    }
    out[--out_count] = static_cast<float>(centroids[symbol_table[state]]);
    // state = newStateBaseline + rest
    state = states_table[state] + Pop(bs, chunks, bit_count_table[state]);
  }
}

void FSEDeCompressed(BitStreamState *bs, const uint64_t *chunks, float *buff, int buff_count,
                     const uint16_t *states_table, const uint8_t *bit_count_table, const uint16_t *symbol_table,
                     const float *centroids, size_t table_log, cudaStream_t stream, uint32_t device_id) {
  FSEDeCompressed<<<1, 1, 0, stream>>>(bs, chunks, buff, buff_count, states_table, bit_count_table, symbol_table,
                                       centroids, table_log);
}

void DeQuantWithPerLayer(const int8_t *input, float *output, int element_cnt, float scale, int zp, cudaStream_t stream,
                         uint32_t device_id) {
  auto grid = CUDA_BLOCKS(device_id, element_cnt);
  auto block = CUDA_THREADS(device_id);
  DeQuantWithPerLayer<<<grid, block, 0, stream>>>(input, output, element_cnt, scale, zp);
}

void DeQuantWithPerChannel(const int8_t *input, float *output, int element_cnt, const float *scale, const int *zp,
                           size_t stride, size_t bucket_count, cudaStream_t stream, uint32_t device_id) {
  auto grid = CUDA_BLOCKS(device_id, element_cnt);
  auto block = CUDA_THREADS(device_id);
  DeQuantWithPerChannel<<<grid, block, 0, stream>>>(input, output, element_cnt, scale, zp, stride, bucket_count);
}
