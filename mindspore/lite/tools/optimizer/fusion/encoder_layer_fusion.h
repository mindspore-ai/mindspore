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
#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_ENCODER_LAYER_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_ENCODER_LAYER_FUSION_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "tools/optimizer/common/multiple_pattern_process_pass.h"
#include "include/common/utils/utils.h"
#include "include/errorcode.h"
#include "ops/encoder_layer.h"
#include "tools/optimizer/fusion/multi_head_attention_fusion.h"
#include "ops/fusion/layer_norm_fusion.h"
#include "ops/fusion/activation.h"

namespace mindspore {
namespace opt {
class EncoderLayerFusion : public MultiplePatternProcessPass {
 public:
  explicit EncoderLayerFusion(bool embedding_layer = false, const std::string &name = "EncoderLayerFusion",
                              bool multigraph = true)
      : MultiplePatternProcessPass(name, multigraph) {
    embedding_layer_ = embedding_layer;
  }

  ~EncoderLayerFusion() override = default;

  AnfNodePtr Process(const std::string &pattern_name, const FuncGraphPtr &, const AnfNodePtr &,
                     const EquivPtr &) const override;
  std::unordered_map<std::string, VectorRef> DefinePatterns() const override;

 protected:
  virtual bool Init() const;

 private:
  const std::string kPatternEncoderLayerPreNormUsePast = "PatternEncoderLayerPreNormUsePast";
  const std::string kPatternEncoderLayerUsePastWithLastNorm = "PatternEncoderLayerPreNormUsePastWithLastNorm";
  const std::string kPatternEncoderLayerPost = "PatternTEncoderLayerPost";
  const std::string kPatternEncoderLayerPre = "PatternTEncoderLayerPre";
  const std::string kPatternEncoderLayerPostNorm = "PatternTEncoderLayerPostNorm";
  const std::string kPatternEncoderLayerPreNorm = "PatternTEncoderLayerPreNorm";
  const std::string kPatternEncoderLayerT5Post = "PatternEncoderLayerT5Post";
  const std::string kPatternEncoderLayerT5Pre = "PatternEncoderLayerT5Pre";
  const std::string kPatternEncoderLayerNormT5Pre = "PatternEncoderLayerNormT5Pre";
  const std::string kPatternQueryLayerUsePast = "PatternQueryLayerUsePast";
  const std::string kPatternSigmaDistributed = "PatternSigmaDistributed";
  const std::string kPatternSigmaDistributedEmbedding = "PatternSigmaDistributedEmbedding";
  const std::string kPatternSigmaMoeDistributed = "PatternSigmaMoeDistributed";
  const std::string kPatternSigmaMoeWithLastLayerNormDistributed = "PatternSigmaMoeWithLastLayerNormDistributed";
  const std::string kPatternSigmaWithLastLayerNormDistributed = "PatternSigmaWithLastLayerNormDistributed";
  const std::string kPatternSigmaQueryLayerDistributed = "PatternSigmaQueryLayerDistributed";
  const std::string kPatternDistributedAlpha = "PatternDistributedAlpha";
  const std::string kPatternDistributedAlphaWithLastLayerNorm = "PatternDistributedAlphaWithLastLayerNorm";
  const std::string kPatternQueryLayerUsePastDistributed = "PatternQueryLayerUsePastDistributed";
  const std::string kPatternSigma = "kPatternSigma";
  const std::string kPatternSigmaEmbedding = "kPatternSigmaEmbedding";
  const std::string kPatternSigmaQuery = "kPatternSigmaQuery";
  const std::string kPatternSigmaMoe = "kPatternSigmaMoe";
  const std::string kPatternSigmaMoeWithLastLayerNorm = "PatternSigmaMoeWithLastLayerNorm";
  const std::string kPatternSigmaWithLastLayerNorm = "PatternSigmaWithLastLayerNorm";
  const std::string kPatternSigmaQueryLayerMoe = "PatternSigmaQueryLayerMoe";
  const std::string kPatternSigmaDistributedMB = "PatternSigmaDistributedMB";
  const std::string kPatternSigmaDistributedEmbeddingMB = "PatternSigmaDistributedEmbeddingMB";
  const std::string kPatternSigmaMoeWithLastLayerNormDistributedMB = "PatternSigmaMoeWithLastLayerNormDistributedMB";
  const std::string kPatternSigmaWithLastLayerNormDistributedMB = "PatternSigmaWithLastLayerNormMB";
  const std::string kPatternSigmaQueryLayerDistributedMB = "PatternSigmaQueryLayerDistributedMB";
  const std::string kPatternSigmaMoeDistributedMB = "PatternSigmaMoeDistributedMB";
  const std::string kPatternSigmaDistributedMBGELU = "PatternSigmaDistributedMBGELU";
  const std::string kPatternSigmaDistributedEmbeddingMBGELU = "PatternSigmaDistributedEmbeddingMBGELU";
  const std::string kPatternSigmaMoeWithLastLayerNormDistributedMBGELU =
    "PatternSigmaMoeWithLastLayerNormDistributedMBGELU";
  const std::string kPatternSigmaWithLastLayerNormDistributedMBGELU = "PatternSigmaWithLastLayerNormMBGELU";
  const std::string kPatternSigmaQueryLayerDistributedMBGELU = "PatternSigmaQueryLayerDistributedMBGELU";
  const std::string kPatternSigmaMoeDistributedMBGELU = "PatternSigmaMoeDistributedMBGELU";
  const std::string kPatternSigmaDistributedMBFirst = "PatternSigmaMoeDistributedMBFirst";
  const std::string kPatternSigmaFirst = "kPatternSigmaFirsts";
  const std::string kPatternSigmaQueryLayerDistributedMBMoe = "kPatternSigmaQueryLayerDistributedMBMoe";
  const std::string kPatternSigmaQueryLayerDistributedMoe = "kPatternSigmaQueryLayerDistributedMoe";
  const std::string kPatternSigmaEmbeddingDistributed = "kPatternSigmaEmbeddingDistributed";

  VectorRef DefinePatternEncoderLayer(bool post_layernorm, bool layernorm_fusion, bool is_position_bias_, bool mask,
                                      bool is_layer_norm) const;
  VectorRef DefinePatternEncoderSigma(bool moe, bool use_past, bool distributed, bool is_layer_norm, bool query_layer,
                                      bool multi_batch, bool first_encoder, bool gelu) const;

  VectorRef DefinePatternEncoderAlpha(bool moe, bool distributed, bool is_layer_norm, bool query_layer,
                                      bool use_past) const;
  VectorRef getTuple(bool post_layernorm, bool layernorm_fusion, bool is_position_bias) const;
  VectorRef DefineLayerNorm(bool is_position_bias, BaseRef input, VarPtr gamma, VarPtr beta, VarPtr eps) const;
  CNodePtr CreateMaskedEncoderLayerFusionNode(const FuncGraphPtr &func_graph, const EquivPtr &equiv,
                                              const AnfNodePtr &node, bool post_layernorm, bool mask) const;
  AnfNodePtr GetAttribute(const FuncGraphPtr &func_graph, const EquivPtr &equiv, VarPtr node_name) const;
  bool IsActGELU(const FuncGraphPtr &func_graph, const EquivPtr &equiv, const VarPtr &input_prim) const;
  lite::STATUS GetEps(const EquivPtr &equiv, VarPtr node_name, float *eps) const;
  lite::STATUS CheckPattern(const FuncGraphPtr &func_graph, const EquivPtr &equiv, int *head_num, int *head_size,
                            float *eps1, float *eps2, float *eps3, float *scale) const;
  std::shared_ptr<ops::EncoderLayer> CreatePrim(const FuncGraphPtr &func_graph, const EquivPtr &equiv,
                                                int64_t ffn_hidden_size, int64_t expert_num, int64_t expert_offset,
                                                float capacity_factor) const;
  VectorRef DefinePatternInitReset(VectorRef input, bool is_value_reset = false, bool is_key_reset = false) const;
  VectorRef DefinePatternMultiBatch(VectorRef input) const;
  BaseRef DefineBatchValidLength(const BaseRef &input) const;
  VectorRef DefinePatternMoERouter(VectorRef input_layernorm) const;
  VectorRef DefinePatternMoE(VectorRef input_layernorm, bool multi_batch, bool gelu) const;
  VectorRef DefinePatternSigmaFfn(BaseRef input, bool gelu, bool distributed) const;
  VectorRef DefinePatternMoETopKRouter(VectorRef input) const;
  VectorRef DefinePatternMoEFfn(VectorRef input_reshape, bool gelu) const;
  VectorRef DefineDependKV(VectorRef input_layernorm, VectorRef deppend_v_input, bool moe) const;
  VectorRef DefineFfn(VectorRef input) const;
  VectorRef DefineFirstEncoder(bool distributed) const;
  lite::STATUS InitAttributes(AnfNodePtr k_past, AnfNodePtr begin_expert_ids, AnfNodePtr weight_m,
                              AnfNodePtr expert_capacity_node, int *ffn_hidden_size, int *expert_num,
                              int *expert_offset, float *capacity_factor) const;
  void InitParams(bool post_layernorm, bool layernorm_fusion, bool is_position_bias, bool mask, bool is_layer_norm,
                  bool use_past, bool query_layer, bool sigma, bool distributed, bool moe) const;
  bool IsUsePast(const std::string &pattern_name) const;
  bool IsUsePastMB(const std::string &pattern_name) const;
  bool IsUsePastAlpha(const std::string &pattern_name) const;
  bool IsLastLayerNorm(const std::string &pattern_name) const;
  bool IsLayerNormFusion(const std::string &pattern_name) const;
  bool IsMoe(const std::string &pattern_name) const;
  bool IsFastGelu(const std::string &pattern_name) const;
  bool IsFastGeluDistributed(const std::string &pattern_name) const;
  bool IsQueryLayer(const std::string &pattern_name) const;

 protected:
  mutable VarPtr input_{nullptr};
  mutable VarPtr expert_ids_input_{nullptr};
  mutable VarPtr expert_ids_{nullptr};
  mutable VarPtr expert_capacity_{nullptr};
  mutable VarPtr begin_expert_ids_{nullptr};
  mutable VarPtr position_bias_{nullptr};
  mutable VarPtr beta1_{nullptr};
  mutable VarPtr gamma1_{nullptr};
  mutable VarPtr beta2_{nullptr};
  mutable VarPtr gamma2_{nullptr};
  mutable VarPtr beta3_{nullptr};
  mutable VarPtr gamma3_{nullptr};
  mutable VarPtr weight_attn_qkv_{nullptr};
  mutable VarPtr weight_attn_q_{nullptr};
  mutable VarPtr weight_attn_o_{nullptr};
  mutable VarPtr weight_m_{nullptr};
  mutable VarPtr weight_p_{nullptr};
  mutable VarPtr bias_attn_qkv_{nullptr};
  mutable VarPtr bias_attn_o_{nullptr};
  mutable VarPtr bias_m_{nullptr};
  mutable VarPtr bias_p_{nullptr};
  mutable VarPtr mask_{nullptr};
  mutable VarPtr is_attention_{nullptr};
  mutable VarPtr is_layernorm1_{nullptr};
  mutable VarPtr is_layernorm2_{nullptr};
  mutable VarPtr is_layernorm3_{nullptr};
  mutable ActType act_type_{ActType::ActType_No};
  mutable VarPtr is_act_{nullptr};
  mutable VarPtr eps1_{nullptr};
  mutable VarPtr eps2_{nullptr};
  mutable VarPtr eps3_{nullptr};
  mutable VarPtr init_reset_{nullptr};
  mutable VarPtr k_past_{nullptr};
  mutable VarPtr v_past_{nullptr};
  mutable VarPtr input_q_{nullptr};
  mutable VarPtr batch_valid_length_{nullptr};
  mutable VarPtr embedding_table_{nullptr};
  mutable VarPtr weight_router_{nullptr};

  mutable VarPtr position_ids_{nullptr};
  mutable VarPtr embedding_table_input_{nullptr};
  mutable VarPtr current_index_{nullptr};
  mutable VarPtr embedding_table_pos_{nullptr};

  mutable bool is_position_bias_{false};
  mutable bool is_post_layernorm_{false};
  mutable bool is_layernorm_fusion_{false};
  mutable bool is_layernorm_{false};
  mutable bool is_use_past_{false};
  mutable bool is_query_layer_{false};
  mutable bool is_sigma_{false};
  mutable bool is_moe_{false};
  mutable bool is_distributed_{false};
  mutable bool is_fast_gelu_{false};
  mutable bool is_embedding_layer_{false};

  mutable bool embedding_layer_{false};
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_ENCODER_LAYER_FUSION_H_
