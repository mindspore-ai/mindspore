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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_LAMB_NEXT_RIGHT_RULE_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_LAMB_NEXT_RIGHT_RULE_H_

#include <memory>
#include "include/backend/optimizer/optimizer.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace opt {
class LambNextRightRule : public PatternProcessPass {
 public:
  explicit LambNextRightRule(bool multigraph = true)
      : PatternProcessPass("lamb_next_right_rule", multigraph),
        input0_(std::make_shared<Var>()),
        input1_(std::make_shared<Var>()),
        mul2_x_(std::make_shared<Var>()),
        mul3_x_(std::make_shared<Var>()),
        true_div1_recip_(std::make_shared<Var>()),
        add2_y_(std::make_shared<Var>()),
        add1_var_(std::make_shared<Var>(std::make_shared<Primitive>(prim::kPrimAdd->name()))) {}

  ~LambNextRightRule() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                           const EquivPtr &equiv) const override;

 private:
  AnfNodePtr CreateLambNextRightNode(const FuncGraphPtr &func_graph, const EquivPtr &equiv) const;

  VarPtr input0_;
  VarPtr input1_;
  VarPtr mul2_x_;
  VarPtr mul3_x_;
  VarPtr true_div1_recip_;
  VarPtr add2_y_;
  VarPtr add1_var_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_LAMB_NEXT_RIGHT_RULE_H_
