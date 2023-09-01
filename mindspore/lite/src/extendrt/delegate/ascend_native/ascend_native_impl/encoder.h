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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either convolutionress or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_ENCODER_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_ENCODER_H_

#include <vector>

namespace mindspore::ascend_native {
#define ENCODER_INPUT_IDX 0
#define ENCODER_LN1_GAMMA_IDX 1
#define ENCODER_LN1_BETA_IDX 2
#define ENCODER_DENSE_CONCAT_IDX 3
#define ENCODER_DENSE_Q_IDX 4
#define ENCODER_DENSE_KV_CONCAT_IDX 5
#define ENCODER_DENSE_BIAS_IDX 6
#define ENCODER_PROJECTION_IDX 7
#define ENCODER_PROJECTION_BIAS_IDX 8
#define ENCODER_LN2_GAMMA_IDX 9
#define ENCODER_LN2_BETA_IDX 10
#define ENCODER_FFN_OUT_IDX 11
#define ENCODER_FFN_OUT_BIAS_IDX 12
#define ENCODER_FFN_PROJ_IDX 13
#define ENCODER_FFN_PROJ_BIAS_IDX 14
#define ENCODER_INPUT_IDS_IDX 15
#define ENCODER_BATCH_VALID_LENGTH_IDX 16
// #define ENCODER_CURRENT_INDEX_IDX 17
#define ENCODER_V_EMBEDDING_IDX 18
#define ENCODER_P_EMBEDDING_IDX 19
#define ENCODER_QUERY_EMBEDDING_IDX 20
#define ENCODER_K_CACHE_IDX 21
#define ENCODER_V_CACHE_IDX 22
#define ENCODER_POS_IDS_IDX 23
#define ENCODER_LN3_GAMMA_IDX 24
#define ENCODER_LN3_BETA_IDX 25
#define ENCODER_Q_IDX 26
#define ENCODER_LAST_IDX 27

#define ENCODER_OUTPUT_IDX 0
#define HEAD_OUTPUT_IDX 1
#define NORM_OUTPUT_IDX 2
#define ENCODER_OUTPUT_LAST_IDX 3

// worspace
// we don't get ENCODER_INDEX_OFFSET_IDX from the MS, we must prepare it
// it's size is min(sizeof(int) * batch size, 32)
// [                                                         | ENCODER_INDEX_OFFSET_IDX ]
//
typedef int (*allGatherFuncT)(const void *in, void *out, size_t size, int data_type, void *stream);
typedef int (*allReduceSumFuncT)(const void *in, void *out, size_t size, int data_type, void *stream);

class CommParam {
 public:
  CommParam() : rank_id_(1), rank_num_(0) {}
  CommParam(uint32_t rank_id, uint32_t rank_num) : rank_id_(rank_id), rank_num_(rank_num) {}

  void set_all_gather(allGatherFuncT allGather) { allGather_ = allGather; }
  void set_all_reduce_sum(allReduceSumFuncT allReduceSum) { allReduceSum_ = allReduceSum; }
  int allReduceSum(const void *in, void *out, size_t size, int data_type, void *stream) {
    if (allReduceSum_ == nullptr) return false;
    return allReduceSum_(in, out, size, data_type, stream);
  }
  int allGather(const void *in, void *out, size_t size, int data_type, void *stream) {
    if (allGather_ == nullptr) return false;
    return allGather_(in, out, size, data_type, stream);
  }
  void set_rank_id(uint32_t rank_id) { rank_id_ = rank_id; }
  uint32_t get_rank_id() { return rank_id_; }

  void set_rank_num(uint32_t rank_num) { rank_num_ = rank_num; }
  uint32_t get_rank_num() { return rank_num_; }

 private:
  uint32_t rank_id_;
  uint32_t rank_num_;
  allGatherFuncT allGather_;
  allReduceSumFuncT allReduceSum_;
};

class NormParam {
 public:
  NormParam() : gamma_(nullptr), beta_(nullptr), eps_(1e-5f) {}
  NormParam(void *gamma, void *beta, float eps) : gamma_(gamma), beta_(beta), eps_(eps) {}

  void set_gamma(void *gamma) { gamma_ = gamma; }
  void *get_gamma() { return gamma_; }
  void set_beta(void *beta) { beta_ = beta; }
  void *get_beta() { return beta_; }
  void set_eps(float eps) { eps_ = eps; }
  float get_eps() { return eps_; }

 private:
  void *gamma_{nullptr};
  void *beta_{nullptr};
  float eps_;
};

class FfnParam {
 public:
  FfnParam()
      : ffn_hidden_size_(0),
        projection_weight_(nullptr),
        projection_bias_(nullptr),
        mapping_weight_(nullptr),
        mapping_bias_(nullptr) {}

  void set_projection_weight(void *projection_weight) { projection_weight_ = projection_weight; }
  void *get_projection_weight() { return projection_weight_; }
  void set_projection_bias(void *projection_bias) { projection_bias_ = projection_bias; }
  void *get_projection_bias() { return projection_bias_; }
  void set_mapping_weight(void *mapping_weight) { mapping_weight_ = mapping_weight; }
  void *get_mapping_weight() { return mapping_weight_; }
  void set_mapping_bias(void *mapping_bias) { mapping_bias_ = mapping_bias; }
  void *get_mapping_bias() { return mapping_bias_; }
  void set_ffn_hidden_size(size_t ffn_hidden_size) { ffn_hidden_size_ = ffn_hidden_size; }
  size_t get_ffn_hidden_size() { return ffn_hidden_size_; }

 private:
  size_t ffn_hidden_size_;
  void *projection_weight_{nullptr};
  void *projection_bias_{nullptr};
  void *mapping_weight_{nullptr};
  void *mapping_bias_{nullptr};
};

class MoeParam : public FfnParam {
 public:
  MoeParam() : FfnParam(), expert_number_(1), moe_id_(0), expert_ids_(nullptr), capacity_factor_(1.1f) {}

  void set_expert_number(uint32_t expert_number) { expert_number_ = expert_number; }
  uint32_t get_expert_number() { return expert_number_; }
  void set_moe_id(uint32_t moe_id) { moe_id_ = moe_id; }
  uint32_t get_moe_id() { return moe_id_; }
  void set_expert_ids(int *expert_ids) { expert_ids_ = expert_ids; }
  int *get_expert_ids() { return expert_ids_; }
  void set_expert_capacity_factor(float capacity_factor) { capacity_factor_ = capacity_factor; }
  float get_expert_capacity_factor() { return capacity_factor_; }

 private:
  uint32_t expert_number_;
  uint32_t moe_id_;
  int *expert_ids_{nullptr};
  float capacity_factor_;
};

class AttentionParam {
 public:
  AttentionParam()
      : is_cross_(false),
        head_number_(0),
        head_size_(0),
        hidden_dim_(0),
        q_seq_len_(0),
        kv_seq_len_(0),
        scale_(0.0f),
        qkv_weight_(nullptr),
        qkv_bias_(nullptr),
        k_cache_(nullptr),
        v_cache_(nullptr),
        projection_weight_(nullptr),
        projection_bias_(nullptr) {}
  void set_is_cross(size_t is_cross) { is_cross_ = is_cross; }
  bool get_is_cross() { return is_cross_; }
  void set_head_number(size_t head_number) { head_number_ = head_number; }
  size_t get_head_number() { return head_number_; }
  void set_head_size(size_t head_size) { head_size_ = head_size; }
  size_t get_head_size() { return head_size_; }
  void set_q_seq_len(size_t q_seq_len) { q_seq_len_ = q_seq_len; }
  size_t get_q_seq_len() { return q_seq_len_; }
  void set_kv_seq_len(size_t kv_seq_len) { kv_seq_len_ = kv_seq_len; }
  size_t get_kv_seq_len() { return kv_seq_len_; }
  void set_hidden_dim(size_t hidden_dim) { hidden_dim_ = hidden_dim; }
  size_t get_hidden_dim() { return hidden_dim_; }
  void set_scale(float scale) { scale_ = scale; }
  float get_scale() { return scale_; }
  void set_projection_weight(void *projection_weight) { projection_weight_ = projection_weight; }
  void *get_projection_weight() { return projection_weight_; }
  void set_projection_bias(void *projection_bias) { projection_bias_ = projection_bias; }
  void *get_projection_bias() { return projection_bias_; }
  void set_qkv_weight(void *qkv_weight) { qkv_weight_ = qkv_weight; }
  void *get_qkv_weight() { return qkv_weight_; }
  void set_qkv_bias(void *qkv_bias) { qkv_bias_ = qkv_bias; }
  void *get_qkv_bias() { return qkv_bias_; }
  void set_kv_weight(void *kv_weight) { kv_weight_ = kv_weight; }
  void *get_kv_weight() { return kv_weight_; }
  void set_k_cache(void *k_cache) { k_cache_ = k_cache; }
  void *get_k_cache() { return k_cache_; }
  void set_v_cache(void *v_cache) { v_cache_ = v_cache; }
  void *get_v_cache() { return v_cache_; }

 private:
  bool is_cross_;
  size_t head_number_;
  size_t head_size_;
  size_t hidden_dim_;
  size_t q_seq_len_;
  size_t kv_seq_len_;
  float scale_;
  void *qkv_weight_{nullptr};
  void *qkv_bias_{nullptr};
  void *kv_weight_{nullptr};
  void *k_cache_{nullptr};
  void *v_cache_{nullptr};
  void *projection_weight_{nullptr};
  void *projection_bias_{nullptr};
};

class VslParam {
 public:
  VslParam() : token_number_(0), padding_offset_(nullptr), seq_len_q_(nullptr), seq_len_kv_(nullptr) {}

  void set_padding_offset(int *padding_offset) { padding_offset_ = padding_offset; }
  int *get_padding_offset() { return padding_offset_; }

  void set_seq_len_q(int *seq_len_q) { seq_len_q_ = seq_len_q; }
  int *get_seq_len_q() { return seq_len_q_; }

  void set_seq_len_kv(int *seq_len_kv) { seq_len_kv_ = seq_len_kv; }
  int *get_seq_len_kv() { return seq_len_kv_; }

  void set_token_number(size_t token_number) { token_number_ = token_number; }
  size_t get_token_number() { return token_number_; }

 private:
  size_t token_number_;
  int *padding_offset_{nullptr};
  int *seq_len_q_{nullptr};
  int *seq_len_kv_{nullptr};
};

class EmbeddingParam {
 public:
  EmbeddingParam()
      : vcobalary_size_(0), word_embedding_(nullptr), position_embedding_(nullptr), top_query_embedding_(nullptr) {}

  void set_vcobalary_size(size_t vcobalary_size) { vcobalary_size_ = vcobalary_size; }
  size_t get_vcobalary_size() { return vcobalary_size_; }

  void set_word_embedding(void *word_embedding) { word_embedding_ = word_embedding; }
  void *get_word_embedding() { return word_embedding_; }

  void set_position_embedding(void *position_embedding) { position_embedding_ = position_embedding; }
  void *get_position_embedding() { return position_embedding_; }

  void set_top_query_embedding(void *top_query_embedding) { top_query_embedding_ = top_query_embedding; }
  void *get_top_query_embedding() { return top_query_embedding_; }

 private:
  size_t vcobalary_size_;
  void *word_embedding_{nullptr};
  void *position_embedding_{nullptr};
  void *top_query_embedding_{nullptr};
};

class EncoderParams {
 public:
  EncoderParams() {}
  size_t B;                 // batch size
  size_t D;                 // embedding size
  size_t E;                 // num of experts
  size_t H;                 // num of heads
  size_t HS;                // head size
  size_t HFFN;              // hidden size (of FFN)
  size_t MAX_N;             // max num of tokens
  size_t vocabulary_size_;  // max num of tokens

  size_t capacity;       // capacity of tokens per expert
  size_t num_of_tokens;  // total num of tokens (in all batches)
  size_t vocabulary_size;
  // embedding parameters
  EmbeddingParam embedding_param;
  // normalization parameters
  NormParam norm1;
  NormParam norm2;
  NormParam norm3;
  // Attention param
  AttentionParam attn_param;
  // moe param
  MoeParam moe;
  // ffn param
  FfnParam ffn_param;
  // communication parameters
  CommParam comm_param;
  // vsl parameters
  VslParam vsl_param;

  bool is_moe_;
  bool is_query_;
  bool is_embedding_;
  bool is_last_norm_;
};

class AscendNativeEncoder {
 public:
  explicit AscendNativeEncoder(bool embedding = false) : embedding_(embedding) {}
  virtual ~AscendNativeEncoder() {}
  virtual void FFN(std::vector<void *> *ins, std::vector<void *> *outs, void *ws, EncoderParams *p, void *q);
  virtual void MoE(std::vector<void *> *ins, std::vector<void *> *outs, void *ws, EncoderParams *p, void *q);
  virtual void LN(void *dst_norm, void *src, void *g, void *b, float epsilon, EncoderParams *p, void *q);
  virtual void Attn(std::vector<void *> *ins, std::vector<void *> *outs, void *ws, EncoderParams *p, void *q);
  virtual void Forward(std::vector<void *> *ins, std::vector<void *> *outs, void *ws, EncoderParams *p, void *q);
  virtual void Embed(std::vector<void *> *ins, std::vector<void *> *outs, void *ws, EncoderParams *p, void *q);
  virtual size_t GetWorkspaceSize(const EncoderParams &p);

 protected:
  virtual void Prepare(std::vector<void *> *ins, std::vector<void *> *outs, void *ws, EncoderParams *p, void *q);
  bool embedding_;
};

class AscendNativeEncoderFuseLastNorm : public AscendNativeEncoder {
 public:
  AscendNativeEncoderFuseLastNorm() : AscendNativeEncoder(false) {}
  virtual ~AscendNativeEncoderFuseLastNorm() {}
  size_t GetWorkspaceSize(const EncoderParams &p) override;
  void Forward(std::vector<void *> *ins, std::vector<void *> *outs, void *ws, EncoderParams *p, void *q) override;
};

}  // namespace mindspore::ascend_native
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_ENCODER_H_
