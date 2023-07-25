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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_QUANT_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_QUANT_IMPL_CUH_

#include <cstdint>
struct BitStreamState {
  int32_t curr_chunk_index{-1};  // the index of the next chunk that we will write to
  uint64_t curr_chunk{0};
  int8_t curr_bit_count{0};  // the number of bits that are currently written in the register.
};

cudaError_t DeQuantWithPerLayer(const int8_t *input, float *output, int element_cnt, float scale, int zp,
                                cudaStream_t stream, uint32_t device_id);
cudaError_t DeQuantWithPerChannel(const int8_t *input, float *output, int element_cnt, const float *scale,
                                  const int *zp, size_t stride, size_t bucket_count, cudaStream_t stream,
                                  uint32_t device_id);
cudaError_t FSEDeCompressed(BitStreamState *bs, const uint64_t *chunks, float *out, int out_count,
                            const uint16_t *states_table, const uint8_t *bit_count_table, const uint16_t *symbol_table,
                            const float *centroids, size_t table_log, cudaStream_t stream, uint32_t device_id);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_QUANT_IMPL_CUH_
