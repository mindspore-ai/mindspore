/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_ADAM_APPLY_ONE_WITH_DECAY_RULE_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_ADAM_APPLY_ONE_WITH_DECAY_RULE_H_

#include <vector>
#include <memory>
#include <string>
#include "include/backend/optimizer/optimizer.h"
#include "include/common/utils/utils.h"
namespace mindspore {
namespace opt {
class AdamApplyOneWithDecayRule : public PatternProcessPass {
 public:
  explicit AdamApplyOneWithDecayRule(const std::string &name = "adam_apply_one_with_decay_rule", bool multigraph = true)
      : PatternProcessPass(name, multigraph) {
    input0_ = std::make_shared<Var>();
    input1_ = std::make_shared<Var>();
    input2_ = std::make_shared<Var>();
    input3_ = std::make_shared<Var>();
    input4_ = std::make_shared<Var>();
    mul0_x_ = std::make_shared<Var>();
    mul1_x_ = std::make_shared<Var>();
    mul2_x_ = std::make_shared<Var>();
    mul3_x_ = std::make_shared<Var>();
    mul4_x_ = std::make_shared<Var>();
    add2_y_ = std::make_shared<Var>();
    add0_var_ = std::make_shared<Var>(std::make_shared<Primitive>(prim::kPrimAdd->name()));
    add1_var_ = std::make_shared<Var>(std::make_shared<Primitive>(prim::kPrimAdd->name()));
    sub0_var_ = std::make_shared<Var>(std::make_shared<Primitive>(prim::kPrimSub->name()));
  }
  ~AdamApplyOneWithDecayRule() override = default;
  const BaseRef DefinePattern() const override = 0;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &equiv) const override;

 protected:
  std::vector<AnfNodePtr> GetFusionNodeInputs(const EquivPtr &equiv, const AnfNodePtr &final_node) const;
  VarPtr input0_;
  VarPtr input1_;
  VarPtr input2_;
  VarPtr input3_;
  VarPtr input4_;
  VarPtr mul0_x_;
  VarPtr mul1_x_;
  VarPtr mul2_x_;
  VarPtr mul3_x_;
  VarPtr mul4_x_;
  VarPtr add2_y_;
  VarPtr add0_var_;
  VarPtr add1_var_;
  VarPtr sub0_var_;
};

class AdamApplyOneWithDecayRuleCond1 : public AdamApplyOneWithDecayRule {
 public:
  explicit AdamApplyOneWithDecayRuleCond1(bool multigraph = true)
      : AdamApplyOneWithDecayRule("adam_apply_one_with_decay_rule_cond1", multigraph) {}

  ~AdamApplyOneWithDecayRuleCond1() override = default;
  const BaseRef DefinePattern() const override;
};

class AdamApplyOneWithDecayRuleCond2 : public AdamApplyOneWithDecayRule {
 public:
  explicit AdamApplyOneWithDecayRuleCond2(bool multigraph = true)
      : AdamApplyOneWithDecayRule("adam_apply_one_with_decay_rule_cond2", multigraph) {}

  ~AdamApplyOneWithDecayRuleCond2() override = default;
  const BaseRef DefinePattern() const override;
};

class AdamApplyOneWithDecayRuleCond3 : public AdamApplyOneWithDecayRule {
 public:
  explicit AdamApplyOneWithDecayRuleCond3(bool multigraph = true)
      : AdamApplyOneWithDecayRule("adam_apply_one_with_decay_rule_cond3", multigraph) {}

  ~AdamApplyOneWithDecayRuleCond3() override = default;
  const BaseRef DefinePattern() const override;
};

class AdamApplyOneWithDecayRuleCond4 : public AdamApplyOneWithDecayRule {
 public:
  explicit AdamApplyOneWithDecayRuleCond4(bool multigraph = true)
      : AdamApplyOneWithDecayRule("adam_apply_one_with_decay_rule_cond4", multigraph) {}

  ~AdamApplyOneWithDecayRuleCond4() override = default;
  const BaseRef DefinePattern() const override;
};

class AdamApplyOneWithDecayRuleCond5 : public AdamApplyOneWithDecayRule {
 public:
  explicit AdamApplyOneWithDecayRuleCond5(bool multigraph = true)
      : AdamApplyOneWithDecayRule("adam_apply_one_with_decay_rule_cond5", multigraph) {}

  ~AdamApplyOneWithDecayRuleCond5() override = default;
  const BaseRef DefinePattern() const override;
};

class AdamApplyOneWithDecayRuleCond6 : public AdamApplyOneWithDecayRule {
 public:
  explicit AdamApplyOneWithDecayRuleCond6(bool multigraph = true)
      : AdamApplyOneWithDecayRule("adam_apply_one_with_decay_rule_cond6", multigraph) {}

  ~AdamApplyOneWithDecayRuleCond6() override = default;
  const BaseRef DefinePattern() const override;
};

class AdamApplyOneWithDecayAssignRuleCond1 : public AdamApplyOneWithDecayRule {
 public:
  explicit AdamApplyOneWithDecayAssignRuleCond1(bool multigraph = true)
      : AdamApplyOneWithDecayRule("adam_apply_one_with_decay_assign_rule_cond1", multigraph) {}

  ~AdamApplyOneWithDecayAssignRuleCond1() override = default;
  const BaseRef DefinePattern() const override;
};

class AdamApplyOneWithDecayAssignRuleCond2 : public AdamApplyOneWithDecayRule {
 public:
  explicit AdamApplyOneWithDecayAssignRuleCond2(bool multigraph = true)
      : AdamApplyOneWithDecayRule("adam_apply_one_with_decay_assign_rule_cond2", multigraph) {}

  ~AdamApplyOneWithDecayAssignRuleCond2() override = default;
  const BaseRef DefinePattern() const override;
};

class AdamApplyOneWithDecayAssignRuleCond3 : public AdamApplyOneWithDecayRule {
 public:
  explicit AdamApplyOneWithDecayAssignRuleCond3(bool multigraph = true)
      : AdamApplyOneWithDecayRule("adam_apply_one_with_decay_assign_rule_cond3", multigraph) {}

  ~AdamApplyOneWithDecayAssignRuleCond3() override = default;
  const BaseRef DefinePattern() const override;
};

class AdamApplyOneWithDecayAssignRuleCond4 : public AdamApplyOneWithDecayRule {
 public:
  explicit AdamApplyOneWithDecayAssignRuleCond4(bool multigraph = true)
      : AdamApplyOneWithDecayRule("adam_apply_one_with_decay_assign_rule_cond4", multigraph) {}

  ~AdamApplyOneWithDecayAssignRuleCond4() override = default;
  const BaseRef DefinePattern() const override;
};

class AdamApplyOneWithDecayAssignRuleCond5 : public AdamApplyOneWithDecayRule {
 public:
  explicit AdamApplyOneWithDecayAssignRuleCond5(bool multigraph = true)
      : AdamApplyOneWithDecayRule("adam_apply_one_with_decay_assign_rule_cond5", multigraph) {}

  ~AdamApplyOneWithDecayAssignRuleCond5() override = default;
  const BaseRef DefinePattern() const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_ADAM_APPLY_ONE_WITH_DECAY_RULE_H_
