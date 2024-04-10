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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_PARAM_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_PARAM_H_
#include <stdint.h>
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
#define ENCODER_CURRENT_INDEX_IDX 17
#define ENCODER_V_EMBEDDING_IDX 18
#define ENCODER_P_EMBEDDING_IDX 19
#define ENCODER_QUERY_EMBEDDING_IDX 20
#define ENCODER_K_CACHE_IDX 21
#define ENCODER_V_CACHE_IDX 22
#define ENCODER_POS_IDS_IDX 23
#define ENCODER_LN3_GAMMA_IDX 24
#define ENCODER_LN3_BETA_IDX 25

#define ENCODER_Q_IDX 26
#define ENCODER_MASK_IDX 27
#define ENCODER_PADDING_Q_IDX 28
#define ENCODER_PADDING_KV_IDX 29
#define ENCODER_SEQ_LEN_Q_IDX 30
#define ENCODER_SEQ_LEN_KV_IDX 31
#define ENCODER_MODE_IDX 32
#define ENCODER_EXPERT_IDS_IDX 33
#define ENCODER_TOKEN_TO_TOKEN_IDX 34
#define ENCODER_LAST_IDX 35

#define ENCODER_OUTPUT_IDX 0
#define HEAD_OUTPUT_IDX 1
#define NORM_OUTPUT_IDX 2
#define ENCODER_OUTPUT_LAST_IDX 3

typedef struct {
  int batch_size_;
  int head_size_;
  int head_num_;
  int hid_dim_;
  int ffn_hid_dim_;
  int vocab_size_;
  int seq_;
  int kv_seq_;
  int *act_q_seq_;
  int *act_kv_seq_;
  float eps1_;
  float eps2_;
  float eps3_;
  float scale_;
  int token_num_;
  int token_num2_;
  int rank_id_;
  int rank_num_;
  int expert_num_;
  bool incremental_mode_;
  bool is_query_;
  bool is_cross_;
  bool is_embedding_;
  bool is_ln3;
  bool is_mask_;
  bool is_moe_;
  int moe_id_;
  int moe_num_;
  float capacity_;
  int token_num_to_expert_;
  int *expert_to_tokens_;
  void *ctx;
} EncoderParams;
}  // namespace mindspore::ascend_native
#endif
