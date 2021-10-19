/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "tools/optimizer/fisson/iter_node_outputs.h"
#include "tools/optimizer/parallel/spliter.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace opt {
AnfNodePtr IterNodeOutputs::Run(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, nullptr, "input func_graph is nullptr");
  MS_CHECK_TRUE_MSG(node != nullptr, nullptr, "input node is nullptr");
  if (!utils::isa<CNodePtr>(node)) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  auto inputs = cnode->inputs();

  for (const auto &input_node : inputs) {
    MS_ASSERT(input_node != nullptr);
    if (!utils::isa<CNodePtr>(input_node)) {
      continue;
    }
    auto input_cnode = input_node->cast<CNodePtr>();
    auto input_name = input_cnode->fullname_with_scope();
    Spliter::GetInstance()->UpdateNodeOutputs(input_name, node);
  }
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
