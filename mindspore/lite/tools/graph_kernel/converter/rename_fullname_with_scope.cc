/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "tools/graph_kernel/converter/rename_fullname_with_scope.h"
#include <string>
#include <unordered_map>

namespace mindspore::graphkernel {
bool RenameFullnameWithScope::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  bool changed = false;
  std::unordered_map<std::string, int> names;
  auto nodes = TopoSort(func_graph->output());
  for (auto &node : nodes) {
    if (node->isa<CNode>()) {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      auto name_scope = cnode->fullname_with_scope();
      if (name_scope.empty()) {
        continue;
      }
      if (names.find(name_scope) == names.end()) {
        names[name_scope] = 1;
      } else {
        // node with same name
        names[name_scope]++;
        auto new_name_scope = name_scope + "-" + std::to_string(names[name_scope]);
        cnode->set_fullname_with_scope(new_name_scope);
        changed = true;
      }
    }
  }
  return changed;
}
}  // namespace mindspore::graphkernel
