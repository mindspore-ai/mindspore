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
#ifndef MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_IR_FUSION_ADAM_APPLY_ONE_WITH_DECAY_RULE_H_
#define MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_IR_FUSION_ADAM_APPLY_ONE_WITH_DECAY_RULE_H_

#include <vector>
#include <memory>
#include "pre_activate/common/optimizer.h"
#include "utils/utils.h"
namespace mindspore {
namespace opt {
class AdamApplyOneWithDecayRule : public PatternProcessPass {
 public:
  explicit AdamApplyOneWithDecayRule(bool multigraph = true)
      : PatternProcessPass("adam_apply_one_with_decay_rule", multigraph) {
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
    add0_var_ = std::make_shared<Var>(std::make_shared<Primitive>(prim::kPrimTensorAdd->name()));
    add1_var_ = std::make_shared<Var>(std::make_shared<Primitive>(prim::kPrimTensorAdd->name()));
  }
  ~AdamApplyOneWithDecayRule() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  std::vector<AnfNodePtr> GetFusionNodeInputs(const EquivPtr &equiv) const;
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
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_IR_FUSION_ADAM_APPLY_ONE_WITH_DECAY_RULE_H_
