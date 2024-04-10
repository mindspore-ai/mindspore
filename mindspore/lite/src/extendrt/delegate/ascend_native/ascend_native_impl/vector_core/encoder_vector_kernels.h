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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_ENCODER_VECTOR_KERNELS_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_ENCODER_VECTOR_KERNELS_H_
#define BLOCK_SIZE 5 * 1024
namespace mindspore::ascend_native {
void Transpose0213Ascendc(void *src, void *dst, void *seq_lens, void *padding_offset, void *mode, uint32_t total_token,
                          uint32_t batch_size, uint32_t seq_len, uint32_t head_num, uint32_t head_size, int core_num,
                          void *stream);
void QKVPermuteAscendc(void *qkv_ptr, void *bias_ptr, void *q_ptr, void *k_cache_ptr, void *v_cache_ptr,
                       void *q_seq_len, void *kv_seq_len, void *q_padding_offset, void *kv_padding_offset, void *mode,
                       int actual_token, int batch, int seq_len, int head_num, int head_size, int core_num,
                       void *stream);
void LayerNormAscendc(void *inputX_gm, void *inputY_gm, void *bias_gm, void *gamm_gm, void *beta_gm, void *output_gm,
                      void *output_norm_gm, void *input_ids_gm, void *input_pos_gm, void *emmbeding_word_gm,
                      void *emmbeding_pos_gm, uint32_t totalToken, uint32_t hLength, float epsilon, uint32_t batch_size,
                      uint32_t vLength, uint32_t seqLength, int core_num, void *stream, void *seq_len_gm = nullptr,
                      void *padding_offset_gm = nullptr, void *mode_gm = nullptr, void *token_to_token_gm = nullptr);
void KernelAddAscendc(void *in1, void *in2, void *out, int len, int vec_core_num, void *stream);
void VocabEmbeddingAscendc(void *position_idx, void *embedding_table, void *out, void *seq_lens, void *padding_offset,
                           void *mode, uint32_t total_token, uint32_t batch_size, uint32_t seq_len,
                           uint32_t hidden_size, int core_num, void *stream);
void CreateVSLAscendc(void *batch_valid_len_gm, void *position_idx_gm, void *q_seq_len_gm, void *kv_seq_len_gm,
                      void *q_padding_offset_gm, void *kv_padding_offset_gm, void *mode, void *token_num_gm,
                      uint32_t batch_size, uint32_t max_seq_len, void *stream);
void GatherHeadAscendc(void *src_gm, void *dst_gm, void *seq_len_gm, void *padding_offset_gm, void *mode_gm,
                       uint32_t total_token, uint32_t batch_size, uint32_t seq_len, uint32_t hidden_size, int core_num,
                       void *stream);
void KernelAddScatterAscendc(void *in1, void *in2, void *out, int token_num, int hidden_size, void *token_to_token_gm,
                             int core_num, void *stream);
void KernelCreateMoeParamAscendc(void *expert_ids, void *expert_count_by_batch, void *expert_count,
                                 void *token_to_token, void *seq_lens, void *padding_offset, void *mode,
                                 uint32_t expert_num, uint32_t moe_num, uint32_t batch_size, uint32_t seq_len,
                                 uint32_t total_token, uint32_t moe_id, float capacity, bool is_query, int core_num,
                                 void *stream);
void KernelCreateCountExpertAscendc(void *expert_ids, void *out, void *seq_lens, void *padding_offset, void *mode,
                                    uint32_t moe_num, uint32_t expert_num, float capacity, uint32_t batch_size,
                                    uint32_t seq_len, uint32_t moe_id, bool is_query, int core_num, void *stream);
}  // namespace mindspore::ascend_native
#endif
