
/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "extendrt/delegate/ascend_native/ascend_native_impl/encoder_ps.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/attn.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/ffn.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/hccl_adapter.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/vector_core/encoder_vector_kernels.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/utils.h"

namespace mindspore::ascend_native {

template <typename T>
class panguEncoder {
 public:
  panguEncoder() = default;
  ~panguEncoder() {}

  void SetupParam(EncoderParams *p_h) { p_h_ = p_h; }

  void Prepare(std::vector<void *> *ins, int vcores, int ccores) {
    cube_core_num_ = ccores;
    vec_core_num_ = vcores;

    void *q_w = ins->at(ENCODER_DENSE_CONCAT_IDX);
    void *kv_w = nullptr;
    if (p_h_->is_query_) {
      q_w = ins->at(ENCODER_DENSE_Q_IDX);
      kv_w = ins->at(ENCODER_DENSE_KV_CONCAT_IDX);
    }
    void *qkv_b = ins->at(ENCODER_DENSE_BIAS_IDX);
    void *projection_w = ins->at(ENCODER_PROJECTION_IDX);
    void *projection_b = ins->at(ENCODER_PROJECTION_BIAS_IDX);
    void *k_cache = ins->at(ENCODER_K_CACHE_IDX);
    void *v_cache = ins->at(ENCODER_V_CACHE_IDX);
    flash_attn_.Prepare(q_w, kv_w, qkv_b, k_cache, v_cache, projection_w, projection_b, p_h_, vec_core_num_,
                        cube_core_num_);
    void *ffn_map_w = ins->at(ENCODER_FFN_OUT_IDX);
    void *ffn_map_b = ins->at(ENCODER_FFN_OUT_BIAS_IDX);
    void *ffn_projection_w = ins->at(ENCODER_FFN_PROJ_IDX);
    void *ffn_projection_b = ins->at(ENCODER_FFN_PROJ_BIAS_IDX);
    ffn_.Prepare(ffn_map_w, ffn_map_b, ffn_projection_w, ffn_projection_b, p_h_, vec_core_num_, cube_core_num_);
  }

  void Compute(std::vector<void *> *ins, std::vector<void *> *outs, void *user_ws, void *system_ws, void *alt_stream,
               void *stream) {
    int hid_dim = p_h_->hid_dim_;
    int max_seq = p_h_->seq_;
    int max_io_elem = p_h_->batch_size_ * max_seq * hid_dim;
    // map real token to moe token
    ins->at(ENCODER_TOKEN_TO_TOKEN_IDX) = nullptr;
    // count valid token per expert per batch
    void *expert_count_by_batch = nullptr;
    // count valid token per expert
    void *expert_count = nullptr;
    void *ws = user_ws;
    void *sys_ws = system_ws;

    for (int i = 0; i < p_h_->expert_num_; i++) {
      p_h_->expert_to_tokens_[i] = 0;
    }
    p_h_->expert_to_tokens_[0] = p_h_->token_num_;
    p_h_->token_num_to_expert_ = p_h_->token_num_;
    if (p_h_->is_moe_) {
      ins->at(ENCODER_TOKEN_TO_TOKEN_IDX) = user_ws;
      expert_count_by_batch =
        reinterpret_cast<uint8_t *>(ins->at(ENCODER_TOKEN_TO_TOKEN_IDX)) + sizeof(int) * p_h_->batch_size_ * max_seq;
      expert_count =
        reinterpret_cast<uint8_t *>(expert_count_by_batch) + sizeof(int) * p_h_->expert_num_ * p_h_->batch_size_;
      ws = reinterpret_cast<uint8_t *>(expert_count) + sizeof(int) * p_h_->expert_num_;
      CreateMoeParam(ins, expert_count_by_batch, expert_count, alt_stream, stream);
    }
    void *ws_attn_ffn = reinterpret_cast<uint8_t *>(ws) + GetEncoderWsSize();

    // step I - normalization
    void *norm1_out = nullptr;
    if (p_h_->is_embedding_) {
      ins->at(ENCODER_INPUT_IDX) = ws;
      norm1_out = reinterpret_cast<uint8_t *>(ws) + max_io_elem * sizeof(T);
      ComputeEmbeddingLN1(ins, norm1_out, stream);
    } else {
      norm1_out = ws;
      ComputeLN1(ins, norm1_out, stream);
    }

    // workspase illustarion:
    // ---------------------------------------------
    // - encoder_input | norm1_out | empty | empty -
    // ---------------------------------------------

    // step II - attention
    void *attn_out = reinterpret_cast<uint8_t *>(norm1_out) + max_io_elem * sizeof(T);
    ComputeAttn(norm1_out, ins, attn_out, ws_attn_ffn, sys_ws, stream, alt_stream);

    // workspase illustarion:
    // ------------------------------------------------
    // - encoder_input | norm1_out | attn_out | Empty -
    // ------------------------------------------------

    // step III - add residual + norm
    void *residual_out = norm1_out;  // norm1_out is not needed any more
    void *residual_norm_out = reinterpret_cast<uint8_t *>(attn_out) + max_io_elem * sizeof(T);
    ComputeResidualAndNorm(ins, attn_out, residual_out, residual_norm_out, alt_stream, stream);
    // workspase illustarion:
    // ----------------------------------------------------
    // - empty | residual_out | empty | residual_norm_out -
    // ----------------------------------------------------

    // step IV - FFN
    if (p_h_->is_moe_ && p_h_->incremental_mode_)
      aclrtSynchronizeStream(stream);  // make sure expert_to_tokens_ is ready
    void *ffn_out = attn_out;          // attn_out is not needed any more
    ComputeFfn(residual_norm_out, ffn_out, expert_count, ws_attn_ffn, sys_ws, alt_stream, stream);
    if (p_h_->is_ln3 || p_h_->is_query_) {
      outs->at(ENCODER_OUTPUT_IDX) = residual_norm_out;
    }

    // workspase illustarion:
    // ------------------------------------------------
    // - empty | residual_out | ffn_out | encoder_out -
    // ------------------------------------------------
    void *encoder_out = outs->at(ENCODER_OUTPUT_IDX);
    // step V - Add residual (can be fused into ffn)
    ComputeAdd(ins, ffn_out, residual_out, encoder_out, stream);

    // step VI
    if (p_h_->is_ln3) {
      void *norm_out = outs->at(NORM_OUTPUT_IDX);
      ComputeLN3(ins, outs, norm_out, stream);
    } else if (p_h_->is_query_) {
      void *head_out = outs->at(HEAD_OUTPUT_IDX);
      ComputeHead(encoder_out, head_out, ins, ws, sys_ws, stream);
    }
  }

  size_t GetWsSize() {
    size_t encoder_ws = GetEncoderWsSize();
    size_t attn_ws = flash_attn_.GetWsSize(p_h_);
    size_t ffn_ws = ffn_.GetWsSize(p_h_);
    size_t mx_size = (attn_ws > ffn_ws) ? attn_ws : ffn_ws;
    return encoder_ws + mx_size;
  }

 private:
  FlashAttn<T> flash_attn_;
  Ffn<T> ffn_;
  Gemm gemm_prompt_;
  Gemm gemm_inc_;
  void *mm_tiling_;
  void *dev_mm_tiling_;
  int vec_core_num_;
  int cube_core_num_;
  EncoderParams *p_h_ = nullptr;
  bool is_first_gemm_prompt_{true};
  bool is_first_gemm_inc_{true};

  size_t GetEncoderWsSize() {
    int max_token_num = p_h_->batch_size_ * p_h_->seq_;
    int hid_dim = p_h_->hid_dim_;
    size_t size = 4 * max_token_num * hid_dim * sizeof(T) +
                  (p_h_->expert_num_ * p_h_->batch_size_ + max_token_num + p_h_->expert_num_) * sizeof(int);
    return size;
  }

  void ComputeHead(void *in, void *out, std::vector<void *> *ins, void *ws, void *sys_ws, void *stream) {
    void *in_mul = in;
    if (!p_h_->incremental_mode_) {
      if (p_h_->batch_size_ > 1) {
        ascend_native::GatherHeadAscendc(in, ws, ins->at(ENCODER_SEQ_LEN_Q_IDX), ins->at(ENCODER_PADDING_Q_IDX),
                                         ins->at(ENCODER_MODE_IDX), p_h_->token_num_, p_h_->batch_size_, p_h_->seq_,
                                         p_h_->hid_dim_, vec_core_num_, stream);
        in_mul = ws;
      } else {
        in_mul = reinterpret_cast<uint8_t *>(in) + ((p_h_->token_num_ - 1) * p_h_->hid_dim_) * sizeof(T);
      }
      if (is_first_gemm_prompt_) {
        aclrtMemsetAsync(out, p_h_->batch_size_ * p_h_->vocab_size_ * sizeof(T), 0,
                         p_h_->batch_size_ * p_h_->vocab_size_ * sizeof(T), stream);
        gemm_prompt_.init(1, p_h_->batch_size_, p_h_->vocab_size_, p_h_->hid_dim_, in_mul,
                          ins->at(ENCODER_V_EMBEDDING_IDX), out, stream, false, true);
        is_first_gemm_prompt_ = false;
      }
      gemm_prompt_.compute(sys_ws, SYS_WS_RESERVED, stream);
    } else {
      if (is_first_gemm_inc_) {
        is_first_gemm_inc_ = false;
        gemm_inc_.init(1, p_h_->batch_size_, p_h_->vocab_size_, p_h_->hid_dim_, in_mul,
                       ins->at(ENCODER_V_EMBEDDING_IDX), out, stream, false, true);
      }
      gemm_inc_.compute(sys_ws, SYS_WS_RESERVED, stream);
    }
  }
  void ComputeLN1(std::vector<void *> *ins, void *norm_out, void *stream) {
    int token_num = p_h_->token_num_;
    int hid_dim = p_h_->hid_dim_;
    float eps1 = p_h_->eps1_;
    LayerNormAscendc(ins->at(ENCODER_INPUT_IDX), nullptr, nullptr, ins->at(ENCODER_LN1_GAMMA_IDX),
                     ins->at(ENCODER_LN1_BETA_IDX), nullptr, norm_out, nullptr, nullptr, nullptr, nullptr, token_num,
                     hid_dim, eps1, 1, 1, 1, vec_core_num_, stream, nullptr, nullptr, nullptr, nullptr);
  }

  void ComputeEmbeddingLN1(std::vector<void *> *ins, void *output, void *stream) {
    int token_num = p_h_->token_num_;
    int hid_dim = p_h_->hid_dim_;
    float eps = p_h_->eps1_;
    int vcob_size = p_h_->vocab_size_;
    int seq = p_h_->seq_;
    int batch_size = p_h_->batch_size_;
    void *embed_out = ins->at(ENCODER_INPUT_IDX);
    void *gamma = ins->at(ENCODER_LN1_GAMMA_IDX);
    void *beta = ins->at(ENCODER_LN1_BETA_IDX);
    void *input_ids = ins->at(ENCODER_INPUT_IDS_IDX);
    void *pos_ids = ins->at(ENCODER_POS_IDS_IDX);
    void *word_embed = ins->at(ENCODER_V_EMBEDDING_IDX);
    void *pos_embed = ins->at(ENCODER_P_EMBEDDING_IDX);

    LayerNormAscendc(nullptr, nullptr, nullptr, gamma, beta, embed_out, output, input_ids, pos_ids, word_embed,
                     pos_embed, token_num, hid_dim, eps, batch_size, vcob_size, seq, vec_core_num_, stream,
                     ins->at(ENCODER_SEQ_LEN_Q_IDX), ins->at(ENCODER_PADDING_Q_IDX), ins->at(ENCODER_MODE_IDX),
                     nullptr);
  }

  void ComputeLN3(std::vector<void *> *ins, std::vector<void *> *outs, void *norm_out, void *stream) {
    int token_num = p_h_->token_num_;
    int hid_dim = p_h_->hid_dim_;
    float eps = p_h_->eps3_;

    void *input_x = outs->at(ENCODER_OUTPUT_IDX);
    void *gamma = ins->at(ENCODER_LN3_GAMMA_IDX);
    void *beta = ins->at(ENCODER_LN3_BETA_IDX);
    LayerNormAscendc(input_x, nullptr, nullptr, gamma, beta, nullptr, norm_out, nullptr, nullptr, nullptr, nullptr,
                     token_num, hid_dim, eps, 1, 1, 1, vec_core_num_, stream, nullptr, nullptr, nullptr, nullptr);
  }

  void ComputeAttn(void *attn_in, std::vector<void *> *ins, void *attn_out, void *ws, void *sys_ws, void *stream,
                   void *alt_stream) {
    void *mask = ins->at(ENCODER_MASK_IDX);
    void *encoder_pos_idx = ins->at(ENCODER_POS_IDS_IDX);
    void *query_embed_idx = ins->at(ENCODER_QUERY_EMBEDDING_IDX);
    void *q_seq_len = ins->at(ENCODER_SEQ_LEN_Q_IDX);
    void *kv_seq_len = ins->at(ENCODER_SEQ_LEN_KV_IDX);
    void *q_pad = ins->at(ENCODER_PADDING_Q_IDX);
    void *kv_pad = ins->at(ENCODER_PADDING_KV_IDX);
    void *mode = ins->at(ENCODER_MODE_IDX);
    flash_attn_.Compute(attn_in, mask, encoder_pos_idx, query_embed_idx, q_seq_len, kv_seq_len, q_pad, kv_pad, mode,
                        attn_out, p_h_, ws, sys_ws, stream, alt_stream);
  }

  void ComputeResidualAndNorm(std::vector<void *> *ins, void *attn_in, void *residual_out, void *residual_norm_out,
                              void *alt_stream, void *stream) {
    int token_num = p_h_->token_num_;
    int hid_dim = p_h_->hid_dim_;
    float eps = p_h_->eps2_;
    void *input_x = ins->at(ENCODER_INPUT_IDX);
    void *gamma = ins->at(ENCODER_LN2_GAMMA_IDX);
    void *beta = ins->at(ENCODER_LN2_BETA_IDX);
    void *proj_bias = ins->at(ENCODER_PROJECTION_BIAS_IDX);
    void *token_gather = ins->at(ENCODER_TOKEN_TO_TOKEN_IDX);

    LayerNormAscendc(input_x, attn_in, proj_bias, gamma, beta, residual_out, residual_norm_out, nullptr, nullptr,
                     nullptr, nullptr, token_num, hid_dim, eps, 1, 1, 1, vec_core_num_, stream, nullptr, nullptr,
                     nullptr, token_gather);
  }

  void ComputeFfn(void *ffn_in, void *ffn_out, void *expert_count, void *ws, void *sys_ws, void *alt_stream,
                  void *stream) {
    if (expert_count) {
      p_h_->token_num_to_expert_ = 0;
      int inc_max_capacity = UP_DIV((p_h_->capacity_ * p_h_->batch_size_), p_h_->expert_num_);
      for (int i = 0; i < p_h_->expert_num_; i++) {
        if (p_h_->incremental_mode_) {
          p_h_->expert_to_tokens_[i] = std::min(p_h_->expert_to_tokens_[i], inc_max_capacity);
        }
        p_h_->token_num_to_expert_ += p_h_->expert_to_tokens_[i];
      }
    }
    ffn_.Compute(ffn_in, ffn_out, p_h_, ws, sys_ws, stream, alt_stream);
  }

  void ComputeAdd(std::vector<void *> *ins, void *in1, void *in2, void *out, void *stream) {
    int token_num = p_h_->token_num_;
    int hid_dim = p_h_->hid_dim_;
    KernelAddScatterAscendc(in1, in2, out, token_num, hid_dim, ins->at(ENCODER_TOKEN_TO_TOKEN_IDX), vec_core_num_,
                            stream);
  }

  void CreateMoeParam(std::vector<void *> *ins, void *expert_count_by_batch, void *expert_count, void *alt_stream,
                      void *stream) {
    void *act_stream = p_h_->incremental_mode_ ? stream : alt_stream;
    if (!p_h_->incremental_mode_) aclrtSynchronizeStream(stream);
    KernelCreateCountExpertAscendc(
      ins->at(ENCODER_EXPERT_IDS_IDX), expert_count_by_batch, ins->at(ENCODER_SEQ_LEN_Q_IDX),
      ins->at(ENCODER_PADDING_Q_IDX), ins->at(ENCODER_MODE_IDX), p_h_->moe_num_, p_h_->expert_num_, p_h_->capacity_,
      p_h_->batch_size_, p_h_->seq_, p_h_->moe_id_, p_h_->is_query_, vec_core_num_, act_stream);
    KernelCreateMoeParamAscendc(ins->at(ENCODER_EXPERT_IDS_IDX), expert_count_by_batch, expert_count,
                                ins->at(ENCODER_TOKEN_TO_TOKEN_IDX), ins->at(ENCODER_SEQ_LEN_Q_IDX),
                                ins->at(ENCODER_PADDING_Q_IDX), ins->at(ENCODER_MODE_IDX), p_h_->expert_num_,
                                p_h_->moe_num_, p_h_->batch_size_, p_h_->seq_, p_h_->token_num_, p_h_->moe_id_,
                                p_h_->capacity_, p_h_->is_query_, vec_core_num_, act_stream);
    int count = p_h_->expert_num_ * sizeof(int);
    aclrtMemcpyAsync(reinterpret_cast<void *>(p_h_->expert_to_tokens_), count, expert_count, count,
                     ACL_MEMCPY_DEVICE_TO_HOST, act_stream);
  }
};

template <typename T>
void pangu_encoder_prepare(EncoderParams *p, std::vector<void *> *ins, size_t *ws_size, void **executer) {
  *executer = nullptr;
  auto encoder = new panguEncoder<T>;
  if (encoder) {
    encoder->SetupParam(p);
    encoder->Prepare(ins, getVecNum(), getCubeNum());
    *ws_size = encoder->GetWsSize();
    *executer = reinterpret_cast<void *>(encoder);
  }
}

template <typename T>
void pangu_encoder_run(void *executer, std::vector<void *> *ins, std::vector<void *> *outs, EncoderParams *p, void *ws,
                       void *sys_ws, void *alt_stream, void *stream) {
  auto encoder = reinterpret_cast<panguEncoder<T> *>(executer);
  if (encoder) {
    encoder->SetupParam(p);
    encoder->Compute(ins, outs, ws, sys_ws, alt_stream, stream);
  }
}

template <typename T>
void pangu_encoder_delete(void *executer) {
  if (executer) {
    auto encoder = reinterpret_cast<panguEncoder<T> *>(executer);
    delete encoder;
  }
}

template void pangu_encoder_prepare<aclFloat16>(EncoderParams *p, std::vector<void *> *ins, size_t *ws_size,
                                                void **executer);
template void pangu_encoder_run<aclFloat16>(void *executer, std::vector<void *> *ins, std::vector<void *> *outs,
                                            EncoderParams *p, void *ws, void *sys_ws, void *alt_stream, void *stream);
template void pangu_encoder_delete<aclFloat16>(void *executer);

}  // namespace mindspore::ascend_native
