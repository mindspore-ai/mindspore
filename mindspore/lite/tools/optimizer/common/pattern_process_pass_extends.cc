/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/common/pattern_process_pass_extends.h"

#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "backend/common/optimizer/pass_manager.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/manager.h"
#include "tools/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
void LitePatternProcessPass::Build() {
  VarPtr fg = std::make_shared<Var>("RootG");
  pattern_ = Helper::SexpToNode(DefinePattern(), fg, primitive_vars_.get(), multigraph_);
}

AnfNodePtr LitePatternProcessPass::Run(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  if (pattern_ == nullptr) {
    Build();
  }
  auto primitive = GetCNodePrimitive(pattern_);
  if (primitive_vars_ == nullptr || equiv_ == nullptr) {
    return nullptr;
  }
  if (IsPrimitiveCNode(node, primitive)) {
    equiv_->clear();
    EquivPtr equiv = pattern_engine_.Match(pattern_, node, *primitive_vars_, equiv_);
    if (equiv != nullptr && !equiv->empty()) {
      return Process(func_graph, node, equiv);
    }
  }
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
