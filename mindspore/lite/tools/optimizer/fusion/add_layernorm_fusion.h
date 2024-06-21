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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_ADD_LAYERNORM_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_ADD_LAYERNORM_FUSION_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "include/common/utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/optimizer/common/multiple_pattern_process_pass.h"
#include "tools/optimizer/common/pattern_process_pass_extends.h"
#include "mindspore/core/ops/nn_optimizer_ops.h"

namespace mindspore {
namespace opt {
class LayerNormV3Fusion : public MultiplePatternProcessPass {
 public:
  explicit LayerNormV3Fusion(const std::string &name = "LayerNormV3Fusion", bool multigraph = true)
      : MultiplePatternProcessPass(name, multigraph) {}

  ~LayerNormV3Fusion() override = default;

  std::unordered_map<std::string, VectorRef> DefinePatterns() const override;

  AnfNodePtr Process(const std::string &pattern_name, const FuncGraphPtr &, const AnfNodePtr &,
                     const EquivPtr &) const override;

 private:
  bool Init() const;
  bool CheckPattern(const EquivPtr &equiv) const;

  AnfNodePtr CreateLayerNormV3Node(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &equiv) const;

  const VectorRef DefineLayerNormV3Pattern1() const;
  const VectorRef DefineLayerNormV3Pattern2() const;

  // LayerNormV3
  mutable std::vector<VarPtr> reduce_1_x_;
  mutable std::vector<VarPtr> reduce_1_axis_;  // -1
  mutable std::vector<VarPtr> sub_a_;
  mutable std::vector<VarPtr> pow_y_;
  mutable std::vector<VarPtr> reduce_2_axis_;  // -1
  mutable std::vector<VarPtr> add_2_b_;        // -0.00001
  mutable std::vector<VarPtr> mul_b_;
  mutable std::vector<VarPtr> add_3_b_;
  mutable std::vector<VarPtr> cast_to_;
  mutable int index_{0};
};

class FuseAddAndLayernorm : public opt::LitePatternProcessPass {
 public:
  explicit FuseAddAndLayernorm(bool multigraph = true)
      : opt::LitePatternProcessPass("FuseAddAndLayernorm", multigraph) {
    x1_ = std::make_shared<Var>();
    x2_ = std::make_shared<Var>();
    gamma_ = std::make_shared<Var>();
    beta_ = std::make_shared<Var>();
    begin_norm_axis_ = std::make_shared<Var>();
    begin_params_axis_ = std::make_shared<Var>();
    eps_ = std::make_shared<Var>();
    layer_norm_ = std::make_shared<Var>(std::make_shared<Primitive>(prim::kPrimLayerNormV3->name()));
  }
  ~FuseAddAndLayernorm() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  VarPtr x1_;
  VarPtr x2_;
  VarPtr layer_norm_;
  VarPtr gamma_;
  VarPtr beta_;
  VarPtr begin_norm_axis_;
  VarPtr begin_params_axis_;
  VarPtr eps_;
  CondVarPtr tuple_get_item_;
};

}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_ADD_LAYERNORM_FUSION_H_
