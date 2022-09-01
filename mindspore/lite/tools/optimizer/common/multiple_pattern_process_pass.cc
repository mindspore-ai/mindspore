/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "tools/optimizer/common/multiple_pattern_process_pass.h"
#include "tools/optimizer/common/helper.h"
#include "nnacl/op_base.h"

namespace mindspore::opt {
AnfNodePtr MultiplePatternProcessPass::Run(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  if (patterns_.empty()) {
    VarPtr fg = std::make_shared<Var>("RootG");
    MS_CHECK_TRUE_RET(fg != nullptr, nullptr);
    auto patterns = std::move(DefinePatterns());
    for (const auto &pattern : patterns) {
      auto primitive_var = std::make_shared<PrimitiveVarMap>();
      MS_CHECK_TRUE_RET(primitive_var != nullptr, nullptr);
      this->patterns_[pattern.first] = (Helper::SexpToNode(pattern.second, fg, primitive_var.get(), multigraph_));
      this->primitive_var_maps_[pattern.first] = primitive_var;
    }
  }

  auto empty_equiv = std::make_shared<Equiv>();
  MS_CHECK_TRUE_RET(empty_equiv != nullptr, nullptr);
  MS_ASSERT(primitive_var_maps_.size() == patterns_.size());
  for (const auto &iter : primitive_var_maps_) {
    auto name = iter.first;
    auto primitive_var = iter.second;
    auto pattern = this->patterns_[name];
    MS_CHECK_TRUE_RET(primitive_var != nullptr, nullptr);
    MS_CHECK_TRUE_RET(pattern != nullptr, nullptr);
    EquivPtr equiv = pattern_engine_.Match(pattern, node, *primitive_var, empty_equiv);
    if (equiv != nullptr && !equiv->empty()) {
      return Process(name, func_graph, node, equiv);
    }
  }
  return nullptr;
}
}  // namespace mindspore::opt
