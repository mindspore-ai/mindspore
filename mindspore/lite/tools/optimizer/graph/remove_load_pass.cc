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
#include "tools/optimizer/graph/remove_load_pass.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "ops/nn_optimizer_ops.h"

namespace mindspore::opt {
bool RemoveLoadPass::Run(const mindspore::FuncGraphPtr &func_graph) {
  // Remove Load Node
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }

  auto manager = func_graph->manager();
  if (manager == nullptr) {
    return false;
  }

  for (auto &cnode : func_graph->GetOrderedCnodes()) {
    if (opt::CheckPrimitiveType(cnode, prim::kPrimLoad)) {
      auto node_users = manager->node_users()[cnode];
      if (node_users.empty()) {
        MS_LOG(WARNING) << cnode->fullname_with_scope() << " cnode is isolated.";
        continue;
      }
      bool has_assign = false;
      for (const auto &user : node_users) {
        if (opt::CheckPrimitiveType(user.first, prim::kPrimAssign)) {
          has_assign = true;
          break;
        }
      }
      if (!has_assign) {
        MS_LOG(INFO) << cnode->fullname_with_scope() << " `Load` node is removed.";
        bool ret = manager->Replace(cnode, cnode->input(1));
        if (!ret) {
          MS_LOG(ERROR) << cnode->fullname_with_scope() << " replace redundant op failed.";
          return ret;
        }
      }
    }
  }
  return true;
}
}  // namespace mindspore::opt
