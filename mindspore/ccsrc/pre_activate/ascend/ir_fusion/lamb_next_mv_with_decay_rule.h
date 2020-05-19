/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_IR_FUSION_LAMB_NEXT_MV_WITH_DECAY_RULE_H_
#define MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_IR_FUSION_LAMB_NEXT_MV_WITH_DECAY_RULE_H_

#include <vector>
#include <memory>
#include <string>
#include "pre_activate/common/optimizer.h"
#include "pre_activate/common/helper.h"

namespace mindspore {
namespace opt {
class LambNextMVWithDecayRule : public PatternProcessPass {
 public:
  explicit LambNextMVWithDecayRule(const std::string &name = "lamb_next_mv_with_decay_rule_cond4",
                                   bool multigraph = true)
      : PatternProcessPass(name, multigraph),
        child_pattern_engine_(PatternEngine(std::make_shared<DefaultVisitor>(),
                                            std::function<bool(const BaseRef &, const BaseRef &)>(AnfEqual),
                                            std::function<bool(const BaseRef &, const BaseRef &)>(CNodeTypeEqual))),
        child_primitive_vars_(std::make_shared<PrimitiveVarMap>()) {
    for (size_t i = 0; i < kLambNextMVWithDecayInputNum; ++i) {
      input_vars_.push_back(std::make_shared<Var>());
    }
    for (size_t i = 0; i < kLambNextMVWithDecayConstantMulInputNum; ++i) {
      constant_mul_input_vars_.push_back(std::make_shared<Var>());
    }
    constant_add2_y_ = std::make_shared<Var>();
    mul4_var_ = std::make_shared<Var>(std::make_shared<Primitive>(prim::kPrimMul->name()));
    real_div0_var_ = std::make_shared<Var>(std::make_shared<Primitive>(kRealDivOpName));
    real_div1_var_ = std::make_shared<Var>(std::make_shared<Primitive>(kRealDivOpName));
    add0_var_ = std::make_shared<Var>(std::make_shared<Primitive>(prim::kPrimTensorAdd->name()));
    add1_var_ = std::make_shared<Var>(std::make_shared<Primitive>(prim::kPrimTensorAdd->name()));
  }

  ~LambNextMVWithDecayRule() override = default;
  const BaseRef DefinePattern() const override;
  virtual const BaseRef DefineAnotherPattern() const;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 protected:
  bool MatchAnotherPattern(const AnfNodePtr &node, const EquivPtr &equiv) const;
  // check two patterns whether share the same nodes or not
  bool IsShareNodes(const EquivPtr &equiv1, const EquivPtr &equiv2) const;

  AnfNodePtr GetLambNextMVWithDecayOutput(const FuncGraphPtr &func_graph, const AnfNodePtr &new_node,
                                          const AnfNodePtr &add3, const AnfNodePtr &add5, const EquivPtr &equiv) const;
  AnfNodePtr CreateLambNextMVWithDecayNode(const FuncGraphPtr &func_graph, const AnfNodePtr &add3,
                                           const AnfNodePtr &add5, const EquivPtr &equiv) const;
  PatternEngine child_pattern_engine_;
  PrimitiveVarMapPtr child_primitive_vars_;
  std::vector<VarPtr> input_vars_;
  std::vector<VarPtr> constant_mul_input_vars_;
  // nodes which two patterns share
  VarPtr constant_add2_y_;
  VarPtr mul4_var_;
  VarPtr real_div0_var_;
  VarPtr real_div1_var_;
  // part of output nodes
  VarPtr add0_var_;
  VarPtr add1_var_;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_IR_FUSION_LAMB_NEXT_MV_WITH_DECAY_RULE_H_
