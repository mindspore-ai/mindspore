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

#include <stdio.h>
#include <stdlib.h>
#include "extendrt/delegate/ascend_native/ascend_native_impl/vector_core/attn_vec_core.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/vector_core/binary_op.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/vector_core/normalization_layer.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/vector_core/encoder_vector_kernels.h"
using AscendC::AIV;
using AscendC::GetBlockNum;

namespace mindspore::ascend_native {
__global__ __aicore__ void QKVPermute(GM_ADDR qkv_ptr, GM_ADDR bias_ptr, GM_ADDR q_ptr, GM_ADDR k_cache_ptr,
                                      GM_ADDR v_cache_ptr, GM_ADDR q_seq_len, GM_ADDR kv_seq_len,
                                      GM_ADDR q_padding_offset, GM_ADDR kv_padding_offset, GM_ADDR mode,
                                      int actual_token, int batch, int seq_len, int head_num, int head_size,
                                      bool incremental_mode = false) {
  int block_numbers = GetBlockNum();
  int total_blocks = (incremental_mode) ? batch : actual_token;
  int per_core_blocks = UP_DIV(total_blocks, block_numbers);
  KernelQKVPermuteOperator<half, 1>(qkv_ptr,            // qkv
                                    bias_ptr,           // bias
                                    q_ptr,              // q
                                    k_cache_ptr,        // k
                                    v_cache_ptr,        // v
                                    q_seq_len,          // q_seq_len
                                    kv_seq_len,         // kv_seq_len
                                    q_padding_offset,   // qpadding
                                    kv_padding_offset,  // kvpadding
                                    mode,               // mode
                                    actual_token, batch, seq_len, head_num, head_size, total_blocks, per_core_blocks);
}
__global__ __aicore__ void Transpose0213(GM_ADDR src_gm, GM_ADDR dst_gm, GM_ADDR seq_len_gm, GM_ADDR padding_offset_gm,
                                         GM_ADDR mode_gm, uint32_t total_token, uint32_t batch_size, uint32_t seq_len,
                                         uint32_t head_num, uint32_t head_size, bool incremental_mode = false) {
  int constexpr pipe_size = 1;
  int block_numbers = GetBlockNum();
  int elem_number = (incremental_mode) ? batch_size : total_token;
  int elem_per_core = UP_DIV(elem_number, block_numbers);
  KernelTranspose0213Operator<pipe_size, BLOCK_SIZE, half>(src_gm, dst_gm, seq_len_gm, padding_offset_gm, mode_gm,
                                                           elem_per_core, total_token, batch_size, seq_len, head_num,
                                                           head_size);
}
__global__ __aicore__ void VocabEmbedding(GM_ADDR position_idx_gm, GM_ADDR embedding_table_gm, GM_ADDR out_gm,
                                          GM_ADDR seq_len_gm, GM_ADDR padding_offset_gm, GM_ADDR mode_gm,
                                          uint32_t total_token, uint32_t batch_size, uint32_t seq_len,
                                          uint32_t hidden_size) {
  int constexpr pipe_size = 1;
  int block_numbers = GetBlockNum();
  int elem_number = total_token;
  int elem_per_core = UP_DIV(elem_number, block_numbers);
  KernelVocabEmbeddingOperator<pipe_size, BLOCK_SIZE, half>(position_idx_gm, embedding_table_gm, out_gm, seq_len_gm,
                                                            padding_offset_gm, mode_gm, elem_per_core, total_token,
                                                            batch_size, seq_len, hidden_size);
}
void Transpose0213Ascendc(void *src_gm, void *dst_gm, void *seq_len_gm, void *padding_offset_gm, void *mode_gm,
                          uint32_t total_token, uint32_t batch_size, uint32_t seq_len, uint32_t head_num,
                          uint32_t head_size, int core_num, void *stream) {
  Transpose0213<<<core_num, nullptr, stream>>>(
    reinterpret_cast<GM_ADDR>(src_gm), reinterpret_cast<GM_ADDR>(dst_gm), reinterpret_cast<GM_ADDR>(seq_len_gm),
    reinterpret_cast<GM_ADDR>(padding_offset_gm), reinterpret_cast<GM_ADDR>(mode_gm), total_token, batch_size, seq_len,
    head_num, head_size);
}

void QKVPermuteAscendc(void *qkv_ptr, void *bias_ptr, void *q_ptr, void *k_cache_ptr, void *v_cache_ptr,
                       void *q_seq_len, void *kv_seq_len, void *q_padding_offset, void *kv_padding_offset, void *mode,
                       int actual_token, int batch, int seq_len, int head_num, int head_size, int core_num,
                       void *stream) {
  QKVPermute<<<core_num, nullptr, stream>>>(
    reinterpret_cast<GM_ADDR>(qkv_ptr), reinterpret_cast<GM_ADDR>(bias_ptr), reinterpret_cast<GM_ADDR>(q_ptr),
    reinterpret_cast<GM_ADDR>(k_cache_ptr), reinterpret_cast<GM_ADDR>(v_cache_ptr),
    reinterpret_cast<GM_ADDR>(q_seq_len), reinterpret_cast<GM_ADDR>(kv_seq_len),
    reinterpret_cast<GM_ADDR>(q_padding_offset), reinterpret_cast<GM_ADDR>(kv_padding_offset),
    reinterpret_cast<GM_ADDR>(mode), actual_token, batch, seq_len, head_num, head_size);
}

__global__ __aicore__ void LayerNorm(GM_ADDR inputX_gm, GM_ADDR inputY_gm, GM_ADDR bias_gm, GM_ADDR gamm_gm,
                                     GM_ADDR beta_gm, GM_ADDR output_gm, GM_ADDR output_norm_gm, GM_ADDR input_ids_gm,
                                     GM_ADDR input_pos_gm, GM_ADDR emmbeding_word_gm, GM_ADDR emmbeding_pos_gm,
                                     uint32_t total_token, int h_length, float h_length_float, float epsilon,
                                     uint32_t batch_size, uint32_t v_length, uint32_t seq_length, GM_ADDR seq_len_gm,
                                     GM_ADDR padding_offset_gm, GM_ADDR mode_gm, GM_ADDR token_to_token_gm) {
  int constexpr pipe_size = 1;
  int blockNum = GetBlockNum();
  int token_number = total_token;
  int tokenPerBlock = UP_DIV(token_number, blockNum);
  KernelLayernormOperator<pipe_size, BLOCK_SIZE, half>(
    inputX_gm, inputY_gm, bias_gm, gamm_gm, beta_gm, output_gm, output_norm_gm, input_ids_gm, input_pos_gm,
    emmbeding_word_gm, emmbeding_pos_gm, tokenPerBlock, total_token, h_length, h_length_float, (half)epsilon,
    batch_size, v_length, seq_length, seq_len_gm, padding_offset_gm, mode_gm, token_to_token_gm);
}
void LayerNormAscendc(void *inputX_gm, void *inputY_gm, void *bias_gm, void *gamm_gm, void *beta_gm, void *output_gm,
                      void *output_norm_gm, void *input_ids_gm, void *input_pos_gm, void *emmbeding_word_gm,
                      void *emmbeding_pos_gm, uint32_t total_token, uint32_t h_length, float epsilon,
                      uint32_t batch_size, uint32_t v_length, uint32_t seq_length, int core_num, void *stream,
                      void *seq_len_gm, void *padding_offset_gm, void *mode_gm, void *token_to_token_gm) {
  LayerNorm<<<core_num, nullptr, stream>>>(
    reinterpret_cast<GM_ADDR>(inputX_gm), reinterpret_cast<GM_ADDR>(inputY_gm), reinterpret_cast<GM_ADDR>(bias_gm),
    reinterpret_cast<GM_ADDR>(gamm_gm), reinterpret_cast<GM_ADDR>(beta_gm), reinterpret_cast<GM_ADDR>(output_gm),
    reinterpret_cast<GM_ADDR>(output_norm_gm), reinterpret_cast<GM_ADDR>(input_ids_gm),
    reinterpret_cast<GM_ADDR>(input_pos_gm), reinterpret_cast<GM_ADDR>(emmbeding_word_gm),
    reinterpret_cast<GM_ADDR>(emmbeding_pos_gm), total_token, h_length, static_cast<float>(h_length), epsilon,
    batch_size, v_length, seq_length, reinterpret_cast<GM_ADDR>(seq_len_gm),
    reinterpret_cast<GM_ADDR>(padding_offset_gm), reinterpret_cast<GM_ADDR>(mode_gm),
    reinterpret_cast<GM_ADDR>(token_to_token_gm));
}

__global__ __aicore__ void KernelBinaryOperator(GM_ADDR in1, GM_ADDR in2, GM_ADDR out, int len) {
  if (g_coreType == AIV) {
    constexpr int pipe_size = 1;

    int tot_block_num = GetBlockNum();
    int len_per_core = ALIGN32((ALIGN(len, BLOCK_SIZE) / tot_block_num));

    KernelBinaryOp<pipe_size, BLOCK_SIZE, half, kernelAdd<half>> op;
    op.Init(in1, in2, out, len, len_per_core);
    op.Process();
  }
}

void KernelAddAscendc(void *in1, void *in2, void *out, int len, int vec_core_num, void *stream) {
  int core_num = ALIGN(len, BLOCK_SIZE) / BLOCK_SIZE;
  if (core_num > vec_core_num) core_num = vec_core_num;
  KernelBinaryOperator<<<core_num, nullptr, stream>>>(reinterpret_cast<GM_ADDR>(in1), reinterpret_cast<GM_ADDR>(in2),
                                                      reinterpret_cast<GM_ADDR>(out), len);
}

__global__ __aicore__ void KernelAddScatterOperator(GM_ADDR in1, GM_ADDR in2, GM_ADDR out, int token_num,
                                                    int hidden_size, GM_ADDR token_to_token_gm) {
  constexpr int pipe_size = 1;
  int block_numbers = GetBlockNum();
  int elem_number = token_num;
  int elem_per_core = UP_DIV(elem_number, block_numbers);
  KernelAddScatter<pipe_size, BLOCK_SIZE, half> op;
  op.Init(in1, in2, out, token_num, hidden_size, elem_per_core, token_to_token_gm);
  op.Process();
}

void KernelAddScatterAscendc(void *in1, void *in2, void *out, int token_num, int hidden_size, void *token_to_token_gm,
                             int core_num, void *stream) {
  KernelAddScatterOperator<<<core_num, nullptr, stream>>>(
    reinterpret_cast<GM_ADDR>(in1), reinterpret_cast<GM_ADDR>(in2), reinterpret_cast<GM_ADDR>(out), token_num,
    hidden_size, reinterpret_cast<GM_ADDR>(token_to_token_gm));
}
void VocabEmbeddingAscendc(void *position_idx, void *embedding_table, void *out, void *seq_lens, void *padding_offset,
                           void *mode, uint32_t total_token, uint32_t batch_size, uint32_t seq_len,
                           uint32_t hidden_size, int core_num, void *stream) {
  VocabEmbedding<<<core_num, nullptr, stream>>>(
    reinterpret_cast<GM_ADDR>(position_idx), reinterpret_cast<GM_ADDR>(embedding_table), reinterpret_cast<GM_ADDR>(out),
    reinterpret_cast<GM_ADDR>(seq_lens), reinterpret_cast<GM_ADDR>(padding_offset), reinterpret_cast<GM_ADDR>(mode),
    total_token, batch_size, seq_len, hidden_size);
}
__global__ __aicore__ void CreateVSL(GM_ADDR batch_valid_len_gm, GM_ADDR position_idx_gm, GM_ADDR q_seq_len_gm,
                                     GM_ADDR kv_seq_len_gm, GM_ADDR q_padding_offset_gm, GM_ADDR kv_padding_offset_gm,
                                     GM_ADDR mode_gm, GM_ADDR token_num_gm, uint32_t batch_size, uint32_t max_seq_len) {
  KernelCreateVSLOperator(batch_valid_len_gm, position_idx_gm, q_seq_len_gm, kv_seq_len_gm, q_padding_offset_gm,
                          kv_padding_offset_gm, mode_gm, token_num_gm, batch_size, max_seq_len);
}

void CreateVSLAscendc(void *batch_valid_len_gm, void *position_idx_gm, void *q_seq_len_gm, void *kv_seq_len_gm,
                      void *q_padding_offset_gm, void *kv_padding_offset_gm, void *mode, void *token_num_gm,
                      uint32_t batch_size, uint32_t max_seq_len, void *stream) {
  CreateVSL<<<1, nullptr, stream>>>(
    reinterpret_cast<GM_ADDR>(batch_valid_len_gm), reinterpret_cast<GM_ADDR>(position_idx_gm),
    reinterpret_cast<GM_ADDR>(q_seq_len_gm), reinterpret_cast<GM_ADDR>(kv_seq_len_gm),
    reinterpret_cast<GM_ADDR>(q_padding_offset_gm), reinterpret_cast<GM_ADDR>(kv_padding_offset_gm),
    reinterpret_cast<GM_ADDR>(mode), reinterpret_cast<GM_ADDR>(token_num_gm), batch_size, max_seq_len);
  SyncDevice(stream);
}
__global__ __aicore__ void GatherHead(GM_ADDR src_gm, GM_ADDR dst_gm, GM_ADDR seq_len_gm, GM_ADDR padding_offset_gm,
                                      GM_ADDR mode_gm, uint32_t total_token, uint32_t batch_size, uint32_t seq_len,
                                      uint32_t hidden_size) {
  int constexpr pipe_size = 1;
  int block_numbers = GetBlockNum();
  int elem_number = batch_size;
  int elem_per_core = UP_DIV(elem_number, block_numbers);
  KernelGatherHeadOperator<pipe_size, BLOCK_SIZE, half>(src_gm, dst_gm, seq_len_gm, padding_offset_gm, mode_gm,
                                                        elem_per_core, total_token, batch_size, seq_len, hidden_size);
}
void GatherHeadAscendc(void *src_gm, void *dst_gm, void *seq_len_gm, void *padding_offset_gm, void *mode_gm,
                       uint32_t total_token, uint32_t batch_size, uint32_t seq_len, uint32_t hidden_size, int core_num,
                       void *stream) {
  GatherHead<<<core_num, nullptr, stream>>>(
    reinterpret_cast<GM_ADDR>(src_gm), reinterpret_cast<GM_ADDR>(dst_gm), reinterpret_cast<GM_ADDR>(seq_len_gm),
    reinterpret_cast<GM_ADDR>(padding_offset_gm), reinterpret_cast<GM_ADDR>(mode_gm), total_token, batch_size, seq_len,
    hidden_size);
}

__global__ __aicore__ void KernelCreateMoeParamOperator(GM_ADDR expert_ids, GM_ADDR expert_count_by_batch,
                                                        GM_ADDR expert_count, GM_ADDR token_to_token, GM_ADDR seq_lens,
                                                        GM_ADDR padding_offset, GM_ADDR mode, uint32_t expert_num,
                                                        uint32_t moe_num, uint32_t batch_size, uint32_t seq_len,
                                                        uint32_t total_token, uint32_t moe_id, float capacity,
                                                        bool is_query) {
  int block_numbers = GetBlockNum();
  int elem_number = batch_size;
  int elem_per_core = UP_DIV(elem_number, block_numbers);
  KernelCreateMoeParam op;
  op.Init(expert_ids, expert_count_by_batch, expert_count, token_to_token, seq_lens, padding_offset, mode, expert_num,
          moe_num, batch_size, seq_len, total_token, moe_id, capacity, is_query, elem_per_core);
  op.Process();
}

void KernelCreateMoeParamAscendc(void *expert_ids, void *expert_count_by_batch, void *expert_count,
                                 void *token_to_token, void *seq_lens, void *padding_offset, void *mode,
                                 uint32_t expert_num, uint32_t moe_num, uint32_t batch_size, uint32_t seq_len,
                                 uint32_t total_token, uint32_t moe_id, float capacity, bool is_query, int core_num,
                                 void *stream) {
  KernelCreateMoeParamOperator<<<1, nullptr, stream>>>(
    reinterpret_cast<GM_ADDR>(expert_ids), reinterpret_cast<GM_ADDR>(expert_count_by_batch),
    reinterpret_cast<GM_ADDR>(expert_count), reinterpret_cast<GM_ADDR>(token_to_token),
    reinterpret_cast<GM_ADDR>(seq_lens), reinterpret_cast<GM_ADDR>(padding_offset), reinterpret_cast<GM_ADDR>(mode),
    expert_num, moe_num, batch_size, seq_len, total_token, moe_id, capacity, is_query);
}

__global__ __aicore__ void KernelCreateCountExpertOperator(GM_ADDR expert_ids, GM_ADDR out, GM_ADDR seq_lens,
                                                           GM_ADDR padding_offset, GM_ADDR mode, uint32_t moe_num,
                                                           uint32_t expert_num, float capacity, uint32_t batch_size,
                                                           uint32_t seq_len, uint32_t moe_id, bool is_query) {
  int block_numbers = GetBlockNum();
  int elem_number = batch_size;
  int elem_per_core = UP_DIV(elem_number, block_numbers);
  KernelCreateCountExpert op;
  op.Init(expert_ids, out, seq_lens, padding_offset, mode, moe_num, expert_num, capacity, batch_size, seq_len, moe_id,
          is_query, elem_per_core);
  op.Process();
}

void KernelCreateCountExpertAscendc(void *expert_ids, void *out, void *seq_lens, void *padding_offset, void *mode,
                                    uint32_t moe_num, uint32_t expert_num, float capacity, uint32_t batch_size,
                                    uint32_t seq_len, uint32_t moe_id, bool is_query, int core_num, void *stream) {
  KernelCreateCountExpertOperator<<<1, nullptr, stream>>>(
    reinterpret_cast<GM_ADDR>(expert_ids), reinterpret_cast<GM_ADDR>(out), reinterpret_cast<GM_ADDR>(seq_lens),
    reinterpret_cast<GM_ADDR>(padding_offset), reinterpret_cast<GM_ADDR>(mode), moe_num, expert_num, capacity,
    batch_size, seq_len, moe_id, is_query);
}
}  // namespace mindspore::ascend_native
