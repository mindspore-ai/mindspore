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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_LAMB_UPDATE_WITH_LR_RULE_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_LAMB_UPDATE_WITH_LR_RULE_FUSION_H_

#include <memory>
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class LambUpdateWithLRRuleFusion : public PatternProcessPass {
 public:
  explicit LambUpdateWithLRRuleFusion(bool multigraph = true)
      : PatternProcessPass("lamb_update_with_lr_rule_fusion", multigraph) {
    input0_ = std::make_shared<Var>();
    input1_ = std::make_shared<Var>();
    input2_ = std::make_shared<Var>();
    input3_ = std::make_shared<Var>();
    input4_ = std::make_shared<Var>();
    input5_ = std::make_shared<Var>();
    constant_greater_max_ = std::make_shared<Var>();
    constant_select_ = std::make_shared<Var>();
    constant_minimum_ = std::make_shared<Var>();
  }
  ~LambUpdateWithLRRuleFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &equiv) const override;

 private:
  VarPtr input0_;
  VarPtr input1_;
  VarPtr input2_;
  VarPtr input3_;
  VarPtr input4_;
  VarPtr input5_;
  VarPtr constant_greater_max_;
  VarPtr constant_select_;
  VarPtr constant_minimum_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_LAMB_UPDATE_WITH_LR_RULE_FUSION_H_
