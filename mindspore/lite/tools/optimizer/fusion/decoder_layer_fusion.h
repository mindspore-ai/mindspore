// /**
//  * Copyright 2023 Huawei Technologies Co., Ltd
//  *
//  * Licensed under the Apache License, Version 2.0 (the "License");
//  * you may not use this file except in compliance with the License.
//  * You may obtain a copy of the License at
//  *
//  * http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  */
#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_DECODER_LAYER_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_DECODER_LAYER_FUSION_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "tools/optimizer/common/multiple_pattern_process_pass.h"
#include "include/common/utils/utils.h"
#include "include/errorcode.h"
#include "ops/decoder_layer.h"
#include "ops/fusion/layer_norm_fusion.h"
#include "ops/fusion/activation.h"
#include "tools/optimizer/fusion/multi_head_attention_fusion.h"

namespace mindspore {
namespace opt {
class DecoderLayerFusion : public MultiplePatternProcessPass {
 public:
  explicit DecoderLayerFusion(const std::string &name = "DecoderLayerFusion", bool multigraph = true)
      : MultiplePatternProcessPass(name, multigraph) {}

  ~DecoderLayerFusion() override = default;

  AnfNodePtr Process(const std::string &pattern_name, const FuncGraphPtr &, const AnfNodePtr &,
                     const EquivPtr &) const override;
  std::unordered_map<std::string, VectorRef> DefinePatterns() const override;

 protected:
  virtual bool Init() const;

 private:
  VectorRef DefinePatternDecoderLayer(bool post_layernorm, bool layernorm_fusion, bool is_position_bias, bool mask,
                                      bool is_layer_norm) const;
  VectorRef getTuple(bool post_layernorm, bool layernorm_fusion, bool is_position_bias) const;
  VectorRef DefineLayerNorm(VectorRef input, VarPtr gamma, VarPtr beta, VarPtr eps) const;
  CNodePtr CreateMaskedDecoderLayerFusionNode(const FuncGraphPtr &func_graph, const EquivPtr &equiv,
                                              const AnfNodePtr &node, bool post_layernorm, bool mask) const;
  std::shared_ptr<ops::DecoderLayer> CreatePrim(const FuncGraphPtr &func_graph, const EquivPtr &equiv,
                                                bool post_layernorm, int64_t ffn_hidden_size) const;
  lite::STATUS CheckPattern(const FuncGraphPtr &func_graph, const EquivPtr &equiv, int *head_num, int *head_size,
                            float *eps1, float *eps2, float *eps3, float *eps4, bool *is_position_bias1,
                            bool *is_position_bias2, float *scale1, float *scale2) const;
  AnfNodePtr GetAttribute(const FuncGraphPtr &func_graph, const EquivPtr &equiv, VarPtr node_name) const;
  bool IsActGELU(const FuncGraphPtr &func_graph, const EquivPtr &equiv) const;
  lite::STATUS GetEps(const EquivPtr &equiv, VarPtr node_name, float *eps) const;
  VectorRef DefineDecoderLayerNorm(VectorRef input, VarPtr gamma, VarPtr eps) const;

 protected:
  const std::string kPatternDecoderLayerPre = "PatternDecoderLayerPre";
  const std::string kPatternDecoderLayerPost = "PatternDecoderLayerPost";
  const std::string kPatternDecoderLayerNormPre = "PatternDecoderLayerNormPre";
  const std::string kPatternDecoderLayerNormPost = "PatternDecoderLayerNormPost";
  const std::string kPatternDecoderLayerNormT5Pre = "PatternDecoderLayerNormT5Pre";
  const std::string kPatternDecoderT5Pre = "PatternDecoderT5Pre";
  const std::string kPatternDecoderT5Post = "PatternDecoderT5Post";
  mutable VarPtr hidden_stats_{nullptr};
  mutable VarPtr encoder_output_{nullptr};
  mutable VarPtr position_bias_{nullptr};
  mutable VarPtr beta1_{nullptr};
  mutable VarPtr gamma1_{nullptr};
  mutable VarPtr beta2_{nullptr};
  mutable VarPtr gamma2_{nullptr};
  mutable VarPtr gamma3_{nullptr};
  mutable VarPtr gamma4_{nullptr};
  mutable VarPtr beta3_{nullptr};
  mutable VarPtr beta4_{nullptr};
  mutable VarPtr weight_attn_qkv_{nullptr};
  mutable VarPtr weight_attn_qkv_cross_{nullptr};
  mutable VarPtr weight_attn_o_{nullptr};
  mutable VarPtr weight_m_{nullptr};
  mutable VarPtr weight_p_{nullptr};
  mutable VarPtr bias_attn_qkv_{nullptr};
  mutable VarPtr bias_attn_o_{nullptr};
  mutable VarPtr bias_attn_cross_qkv_{nullptr};
  mutable VarPtr bias_attn_cross_o_{nullptr};
  mutable VarPtr bias_m_{nullptr};
  mutable VarPtr bias_p_{nullptr};
  mutable VarPtr mask_{nullptr};
  mutable VarPtr is_attention_{nullptr};
  mutable VarPtr is_attention_cross_{nullptr};
  mutable VarPtr weight_attn_q_{nullptr};
  mutable VarPtr weight_attn_kv_{nullptr};
  mutable VarPtr weight_attn_cross_o_{nullptr};
  mutable VarPtr position_bias_cross_{nullptr};
  mutable VarPtr cross_mask_{nullptr};
  mutable VarPtr reshape_k_{nullptr};
  mutable VarPtr reshape_v_{nullptr};
  mutable VarPtr is_layernorm1_{nullptr};
  mutable VarPtr is_layernorm2_{nullptr};
  mutable VarPtr is_layernorm3_{nullptr};
  mutable VarPtr is_act_{nullptr};
  mutable VarPtr eps1_{nullptr};
  mutable VarPtr eps2_{nullptr};
  mutable VarPtr eps3_{nullptr};
  mutable VarPtr eps4_{nullptr};
  mutable bool is_position_bias_{false};
  mutable bool is_layernorm_fusion_{false};
  mutable bool is_layernorm_{false};
  mutable ActType act_type_{ActType::ActType_No};
  mutable bool layer_norm_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_DECODER_LAYER_FUSION_H_
