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
  explicit EncoderLayerFusion(const std::string &name = "EncoderLayerFusion", bool multigraph = true)
      : MultiplePatternProcessPass(name, multigraph) {}

  ~EncoderLayerFusion() override = default;

  AnfNodePtr Process(const std::string &pattern_name, const FuncGraphPtr &, const AnfNodePtr &,
                     const EquivPtr &) const override;
  std::unordered_map<std::string, VectorRef> DefinePatterns() const override;

 protected:
  virtual bool Init() const;

 private:
  const std::string kPatternEncoderLayerPreNormAlpha = "PatternEncoderLayerPreNormAlpha";
  const std::string kPatternEncoderLayerPost = "PatternTEncoderLayerPost";
  const std::string kPatternEncoderLayerPre = "PatternTEncoderLayerPre";
  const std::string kPatternEncoderLayerPostNorm = "PatternTEncoderLayerPostNorm";
  const std::string kPatternEncoderLayerPreNorm = "PatternTEncoderLayerPreNorm";
  const std::string kPatternEncoderLayerT5Post = "PatternEncoderLayerT5Post";
  const std::string kPatternEncoderLayerT5Pre = "PatternEncoderLayerT5Pre";
  const std::string kPatternEncoderLayerNormT5Pre = "PatternEncoderLayerNormT5Pre";

  VectorRef DefinePatternEncoderLayer(bool post_layernorm, bool layernorm_fusion, bool is_position_bias_, bool mask,
                                      bool is_layer_norm, bool alpha) const;
  VectorRef getTuple(bool post_layernorm, bool layernorm_fusion, bool is_position_bias) const;
  VectorRef DefineLayerNorm(bool is_position_bias, VectorRef input, VarPtr gamma, VarPtr beta, VarPtr eps) const;
  CNodePtr CreateMaskedEncoderLayerFusionNode(const FuncGraphPtr &func_graph, const EquivPtr &equiv,
                                              const AnfNodePtr &node, bool post_layernorm = true,
                                              bool mask = true) const;
  AnfNodePtr GetAttribute(const FuncGraphPtr &func_graph, const EquivPtr &equiv, VarPtr node_name) const;
  bool IsActGELU(const FuncGraphPtr &func_graph, const EquivPtr &equiv, const VarPtr &input_prim) const;
  lite::STATUS GetEps(const EquivPtr &equiv, VarPtr node_name, float *eps) const;
  lite::STATUS CheckPattern(const FuncGraphPtr &func_graph, const EquivPtr &equiv, int *head_num, int *head_size,
                            float *eps1, float *eps2, float *eps3, float *scale) const;
  std::shared_ptr<ops::EncoderLayer> CreatePrim(const FuncGraphPtr &func_graph, const EquivPtr &equiv,
                                                bool post_layernorm, int64_t ffn_hidden_size) const;

 protected:
  mutable VarPtr input_{nullptr};
  mutable VarPtr position_bias_{nullptr};
  mutable VarPtr beta1_{nullptr};
  mutable VarPtr gamma1_{nullptr};
  mutable VarPtr beta2_{nullptr};
  mutable VarPtr gamma2_{nullptr};
  mutable VarPtr beta3_{nullptr};
  mutable VarPtr gamma3_{nullptr};
  mutable VarPtr weight_attn_qkv_{nullptr};
  mutable VarPtr weight_attn_qkv_cross_{nullptr};
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
  mutable bool is_position_bias_{false};
  mutable bool is_layernorm_fusion_{false};
  mutable ActType act_type_{ActType::ActType_No};
  mutable VarPtr is_act_{nullptr};
  mutable VarPtr eps1_{nullptr};
  mutable VarPtr eps2_{nullptr};
  mutable VarPtr eps3_{nullptr};
  mutable bool is_layernorm_{false};
  mutable bool alpha_{false};
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_ENCODER_LAYER_FUSION_H_
