/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "backend/common/pass/common/optimizer_utils.h"
#include <vector>
#include "primitive/other_ops.h"
#include "primitive/framework_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"

namespace mindspore {
namespace opt {
auto const kSplitConcatDepend = "split_concat_depend";
void OptimizerUtils::MoveContrlDepend(const FuncGraphPtr &func_graph, const AnfNodePtr &from_node,
                                      const AnfNodePtr &to_node) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(from_node);
  MS_EXCEPTION_IF_NULL(to_node);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto node_users = manager->node_users()[from_node];
  constexpr size_t kDependControlIdx = 2;
  for (auto &node_user : node_users) {
    if (!IsPrimitiveCNode(node_user.first, prim::kPrimDepend)) {
      continue;
    }
    auto cnode_user = node_user.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode_user);
    if (cnode_user->input(kDependControlIdx) == from_node) {
      if (node_users.size() < kDependControlIdx) {
        MS_LOG(WARNING) << "The from_node only referenced by control edge.";
        return;
      }
      manager->SetEdge(cnode_user, kDependControlIdx, to_node);
    }
  }
}

std::vector<CNodePtr> OptimizerUtils::MoveDataDepend(const FuncGraphPtr &func_graph, const AnfNodePtr &from_node,
                                                     const CNodePtr &to_node) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(from_node);
  MS_EXCEPTION_IF_NULL(to_node);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto node_users = manager->node_users()[from_node];
  constexpr size_t kDependDataIdx = 1;
  std::vector<CNodePtr> data_depend_nodes;
  for (auto &node_user : node_users) {
    if (!IsPrimitiveCNode(node_user.first, prim::kPrimDepend)) {
      continue;
    }
    auto cnode_user = node_user.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode_user);
    if (cnode_user->input(kDependDataIdx) != from_node) {
      continue;
    }

    if (!cnode_user->HasAttr(kSplitConcatDepend)) {
      continue;
    }
    data_depend_nodes.emplace_back(cnode_user);
  }

  for (auto &depend_node : data_depend_nodes) {
    manager->SetEdge(depend_node, kDependDataIdx, to_node);
    depend_node->set_abstract(to_node->abstract());
  }
  return data_depend_nodes;
}

void OptimizerUtils::ReplaceDataDepend(const FuncGraphPtr &func_graph, const std::vector<CNodePtr> &old_nodes,
                                       const AnfNodePtr &new_node) {
  if (old_nodes.empty()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(new_node);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (auto &old_node : old_nodes) {
    manager->Replace(old_node, new_node);
  }
}
}  // namespace opt
}  // namespace mindspore
