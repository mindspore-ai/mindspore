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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_LAMB_NEXT_MV_WITH_DECAY_V1_RULE_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_LAMB_NEXT_MV_WITH_DECAY_V1_RULE_H_

#include <vector>
#include <memory>
#include "include/backend/optimizer/optimizer.h"
#include "include/backend/optimizer/helper.h"

namespace mindspore {
namespace opt {
class LambNextMVWithDecayV1Rule : public PatternProcessPass {
 public:
  explicit LambNextMVWithDecayV1Rule(bool multigraph = true)
      : PatternProcessPass("lamb_next_mv_with_decay_v1_rule", multigraph) {
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
  }

  ~LambNextMVWithDecayV1Rule() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                           const EquivPtr &equiv) const override;

 private:
  std::vector<AnfNodePtr> GetFusionNodeInputs(const EquivPtr &equiv) const;
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
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_LAMB_NEXT_MV_WITH_DECAY_V1_RULE_H_
