/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_LAMB_NEXT_MV_RULE_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_LAMB_NEXT_MV_RULE_H_

#include <vector>
#include <string>
#include <memory>
#include "utils/hash_map.h"
#include "ir/anf.h"
#include "include/backend/optimizer/pattern_engine.h"
#include "include/backend/optimizer/helper.h"
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class LambNextMVRule : public MultipleOutputPatternProcessPass {
 public:
  explicit LambNextMVRule(const std::string &name = "", bool multigraph = true)
      : MultipleOutputPatternProcessPass(name, multigraph) {
    input0_ = std::make_shared<Var>();
    input1_ = std::make_shared<Var>();
    input2_ = std::make_shared<Var>();
    input3_ = std::make_shared<Var>();
    input4_ = std::make_shared<Var>();
    input5_ = std::make_shared<Var>();
    input6_ = std::make_shared<Var>();
    mul0_x_ = std::make_shared<Var>();
    mul1_sub_ = std::make_shared<Var>();
    mul2_x_ = std::make_shared<Var>();
    mul3_sub1_ = std::make_shared<Var>();
    mul4_x_ = std::make_shared<Var>();
    add2_y_ = std::make_shared<Var>();
    real_div0_var_ = std::make_shared<Var>(std::make_shared<Primitive>(kRealDivOpName));
    real_div1_var_ = std::make_shared<Var>(std::make_shared<Primitive>(kRealDivOpName));
    real_div2_var_ = std::make_shared<Var>(std::make_shared<Primitive>(prim::kPrimMul->name()));
    add0_var_ = std::make_shared<Var>(std::make_shared<Primitive>(prim::kPrimAdd->name()));
    add1_var_ = std::make_shared<Var>(std::make_shared<Primitive>(prim::kPrimAdd->name()));
  }
  ~LambNextMVRule() override = default;
  const BaseRef DefinePattern() const override = 0;
  BaseRef DefineAnotherPattern() const override = 0;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                           const EquivPtr &equiv) const override;
  bool IsShareNodes(const EquivPtr &equiv1, const EquivPtr &equiv2) const override;

 protected:
  bool IsRuleMatched(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &equiv,
                     std::vector<AnfNodePtr> *const old_pattern_outputs) const;
  AnfNodePtr CreateLambNextMVNode(const FuncGraphPtr &func_graph, const std::vector<AnfNodePtr> &old_pattern_outputs,
                                  const EquivPtr &equiv) const;

  VarPtr input0_;
  VarPtr input1_;
  VarPtr input2_;
  VarPtr input3_;
  VarPtr input4_;
  VarPtr input5_;
  VarPtr input6_;
  VarPtr mul0_x_;
  VarPtr mul1_sub_;
  VarPtr mul2_x_;
  VarPtr mul3_sub1_;
  VarPtr mul4_x_;
  VarPtr add2_y_;
  // nodes which two patterns share, and add2_y_ also.
  VarPtr real_div0_var_;
  VarPtr real_div1_var_;
  // part of output nodes
  VarPtr add0_var_;
  VarPtr add1_var_;
  // other node
  VarPtr real_div2_var_;
};

class LambNextMVRuleCond1 : public LambNextMVRule {
 public:
  explicit LambNextMVRuleCond1(bool multigraph = true) : LambNextMVRule("lamb_next_mv_rule_cond1", multigraph) {}

  ~LambNextMVRuleCond1() override = default;
  const BaseRef DefinePattern() const override;
  BaseRef DefineAnotherPattern() const override;
};

class LambNextMVRuleCond2 : public LambNextMVRule {
 public:
  explicit LambNextMVRuleCond2(bool multigraph = true) : LambNextMVRule("lamb_next_mv_rule_cond2", multigraph) {}

  ~LambNextMVRuleCond2() override = default;
  const BaseRef DefinePattern() const override;
  BaseRef DefineAnotherPattern() const override;
};

class LambNextMVRuleCond3 : public LambNextMVRule {
 public:
  explicit LambNextMVRuleCond3(bool multigraph = true) : LambNextMVRule("lamb_next_mv_rule_cond3", multigraph) {}

  ~LambNextMVRuleCond3() override = default;
  const BaseRef DefinePattern() const override;
  BaseRef DefineAnotherPattern() const override;
};

class LambNextMVRuleCond4 : public LambNextMVRule {
 public:
  explicit LambNextMVRuleCond4(bool multigraph = true) : LambNextMVRule("lamb_next_mv_rule_cond4", multigraph) {}

  ~LambNextMVRuleCond4() override = default;
  const BaseRef DefinePattern() const override;
  BaseRef DefineAnotherPattern() const override;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_LAMB_NEXT_MV_RULE_H_
