/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_TFLITE_REL_POS_MULTI_HEAD_ATTENTION_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_TFLITE_REL_POS_MULTI_HEAD_ATTENTION_FUSION_H_

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include "include/backend/optimizer/optimizer.h"
#include "include/common/utils/utils.h"
#include "include/errorcode.h"
#include "tools/optimizer/fusion/multi_head_attention_fusion.h"

namespace mindspore::opt {
class TfliteRelPosMultiHeadAttentionFusion : public MultiHeadAttentionFusion {
 public:
  explicit TfliteRelPosMultiHeadAttentionFusion(const std::string &name = "TfliteRelPosMultiHeadAttentionFusion",
                                                bool multigraph = true)
      : MultiHeadAttentionFusion(name, multigraph) {}
  ~TfliteRelPosMultiHeadAttentionFusion() override = default;
  AnfNodePtr Process(const std::string &pattern_name, const FuncGraphPtr &, const AnfNodePtr &,
                     const EquivPtr &) const override;
  std::unordered_map<std::string, VectorRef> DefinePatterns() const override;

 private:
  bool Init() const override;

  std::shared_ptr<ops::Attention> BuildAttentionPrim(const EquivPtr &equiv) const override;

  const VectorRef DefineProcessInputPattern(const BaseRef &input, const BaseRef &weight, const BaseRef &bias,
                                            const std::vector<VarPtr> &stack_params, const VarPtr &full_connect_prim,
                                            bool transpose = false) const;
  const VectorRef DefineProcessOutputPattern(const BaseRef &input, const BaseRef &weight, const BaseRef &bias) const;

  CNodePtr CreateRelPosMultiHeadAttentionNode(const FuncGraphPtr &func_graph, const EquivPtr &equiv,
                                              const std::string &base_name) const;

  int SetQuantParamForAttentionNode(const PrimitivePtr &prim, const EquivPtr &equiv) const;

  const VectorRef DefineRelativeShiftPattern(const BaseRef &input) const;

 private:
  const std::string kRPMHAttentionPatternName = "RPMHAttentionPattern";
  mutable VarPtr query_u_{nullptr};
  mutable VarPtr query_v_{nullptr};
  mutable VarPtr query_prim_{nullptr};
  mutable VarPtr key_prim_{nullptr};
  mutable VarPtr value_prim_{nullptr};
  mutable VarPtr pos_prim_{nullptr};
  mutable VarPtr output_prim_{nullptr};
  mutable VarPtr input_p_{nullptr};
  mutable VarPtr weight_p_{nullptr};
  mutable std::vector<VarPtr> query_stack_params_;
  mutable std::vector<VarPtr> key_stack_params_;
  mutable std::vector<VarPtr> value_stack_params_;
  mutable std::vector<VarPtr> pos_stack_params_;
};

}  // namespace mindspore::opt
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_TFLITE_REL_POS_MULTI_HEAD_ATTENTION_FUSION_H_
