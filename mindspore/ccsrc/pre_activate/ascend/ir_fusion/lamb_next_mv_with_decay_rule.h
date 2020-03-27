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
#include "pre_activate/common/optimizer.h"
#include "pre_activate/common/helper.h"

namespace mindspore {
namespace opt {
class LambNextMVWithDecayRule : public PatternProcessPass {
 public:
  explicit LambNextMVWithDecayRule(bool multigraph = true)
      : PatternProcessPass("lamb_next_mv_with_decay_rule", multigraph) {
    for (size_t i = 0; i < kLambNextMVWithDecayInputNum; ++i) {
      input_vars_.push_back(std::make_shared<Var>());
    }
    for (size_t i = 0; i < kLambNextMVWithDecayConstantMulInputNum; ++i) {
      constant_mul_input_vars_.push_back(std::make_shared<Var>());
    }
    constant_add2_y_ = std::make_shared<Var>();
  }

  ~LambNextMVWithDecayRule() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  AnfNodePtr CreateLambNextMVWithDecayNode(const FuncGraphPtr &func_graph, const AnfNodePtr &add3,
                                           const AnfNodePtr &add5, const AnfNodePtr &real_div0,
                                           const AnfNodePtr &real_div1, const EquivPtr &equiv) const;

  std::vector<VarPtr> input_vars_;
  std::vector<VarPtr> constant_mul_input_vars_;
  VarPtr constant_add2_y_;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_IR_FUSION_LAMB_NEXT_MV_WITH_DECAY_RULE_H_
