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

#ifndef MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_IR_FUSION_LAMB_NEXT_MV_RULE_H_
#define MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_IR_FUSION_LAMB_NEXT_MV_RULE_H_

#include <vector>
#include <string>
#include <utility>
#include <unordered_map>
#include <memory>
#include "ir/anf.h"
#include "pre_activate/common/pattern_engine.h"
#include "pre_activate/common/helper.h"
#include "pre_activate/common/optimizer.h"

namespace mindspore {
namespace opt {
class LambNextMVRule : public PatternProcessPass {
 public:
  explicit LambNextMVRule(bool multigraph = true) : PatternProcessPass("lamb_next_mv_rule", multigraph) {
    for (size_t i = 0; i < kLambNextMVRuleInputNum - 1; ++i) {
      input_varptr_.push_back(std::make_shared<Var>());
    }
  }
  ~LambNextMVRule() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  std::vector<VarPtr> input_varptr_;
  bool IsRuleMatched(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                     std::vector<AnfNodePtr> *old_pattern_outputs) const;
  AnfNodePtr CreateLambNextMVNode(const FuncGraphPtr &func_graph, const std::vector<AnfNodePtr> &old_pattern_outputs,
                                  const EquivPtr &equiv) const;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_IR_FUSION_LAMB_NEXT_MV_RULE_H_
