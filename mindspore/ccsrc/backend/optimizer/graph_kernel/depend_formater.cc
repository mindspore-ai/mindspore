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
#include "backend/optimizer/graph_kernel/depend_formater.h"
#include <tuple>
#include <utility>
#include <vector>
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/kernel_compiler/common_utils.h"
#include "backend/optimizer/graph_kernel/graph_kernel_helper.h"

namespace mindspore {
namespace opt {
namespace {
bool RemoveRedundantDepend(const AnfNodePtr &node, const FuncGraphManagerPtr &mng) {
  const auto &users = mng->node_users()[node];
  std::vector<std::pair<AnfNodePtr, int>> sons;
  for (const auto &[user, index] : users) {
    if (!IsPrimitiveCNode(user, prim::kPrimTupleGetItem)) {
      sons.emplace_back(user, index);
      continue;
    }
    auto &[fake_first_grad_son, grad_index] = *((mng->node_users()[user]).begin());
    sons.emplace_back(fake_first_grad_son, grad_index);
  }

  AnfNodePtrList latter_to_delete;
  for (const auto &[son, index] : sons) {
    if (!IsPrimitiveCNode(son, prim::kPrimDepend) || index != kDependAttachNodeIndex) {
      continue;
    }

    latter_to_delete.push_back(son);
  }

  if (latter_to_delete.empty()) {
    return false;
  }

  std::vector<AnfNodePtr>::iterator delete_begin = latter_to_delete.begin();
  if (latter_to_delete.size() == sons.size()) {
    // Left one Depend node relation and delete others!
    ++delete_begin;
  }
  for (; delete_begin != latter_to_delete.end(); ++delete_begin) {
    auto depend_anfnode = *delete_begin;
    auto depend_cnode = depend_anfnode->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(depend_cnode);
    auto depend_prior_node = depend_cnode->input(kRealInputIndexInDepend);
    mng->Replace(depend_anfnode, depend_prior_node);
  }
  return true;
}

AnfNodePtr FindPatronNode(const FuncGraphPtr &main_graph, const FuncGraphManagerPtr &mng) {
  AnfNodePtr patron_node;

  auto return_cnode = main_graph->get_return()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(return_cnode);
  auto output_node = return_cnode->input(kFirstDataInputIndex);
  if (IsPrimitiveCNode(output_node, prim::kPrimMakeTuple)) {
    auto output_cnode = output_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(output_cnode);
    patron_node = output_cnode->input(kFirstDataInputIndex);
  } else {
    patron_node = output_node;
  }

  return patron_node;
}

void AddDepends(const AnfNodePtr &stable_node, const AnfNodePtrList &free_nodes, const FuncGraphPtr &main_graph,
                const FuncGraphManagerPtr &mng) {
  AnfNodePtr modified_node = stable_node;
  for (const auto &free_node : free_nodes) {
    AnfNodePtrList d_inputs = {NewValueNode(prim::kPrimDepend), modified_node, free_node};
    auto depend_cnode = main_graph->NewCNode(d_inputs);
    depend_cnode->set_abstract(modified_node->abstract());
    main_graph->AddNode(depend_cnode);
    modified_node = depend_cnode;
  }

  if (!free_nodes.empty()) {
    mng->Replace(stable_node, modified_node);
  }
}
}  // namespace

bool DependFormater::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }

  // 1. Try to remove redundant depend.
  bool changed = false;
  auto nodes = TopoSort(func_graph->get_return());
  std::for_each(nodes.rbegin(), nodes.rend(), [&changed, &mng](const AnfNodePtr &node) -> void {
    if (HasAbstractMonad(node)) {
      return;
    }
    if (RemoveRedundantDepend(node, mng)) {
      changed = true;
    }
  });

  // Should re-toposort for changed graph.
  if (changed) {
    nodes = TopoSort(func_graph->get_return());
  }

  // 2. Move depend to tail of graph.
  AnfNodePtrList old_depends;
  AnfNodePtrList free_nodes;

  // Find depend and its free nodes.
  for (const auto &node : nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimDepend) ||
        HasAbstractMonad(node->cast<CNodePtr>()->input(kDependAttachNodeIndex))) {
      continue;
    }

    old_depends.push_back(node);
    auto cnode = node->cast<CNodePtr>();
    for (size_t id = kDependAttachNodeIndex; id < cnode->inputs().size(); ++id) {
      auto attach_node = cnode->input(id);
      if (!IsPrimitiveCNode(attach_node, prim::kPrimDepend)) {
        continue;
      }
      free_nodes.push_back(attach_node);
    }
  }

  if (old_depends.empty()) {
    return changed;
  }

  // Delete old depend.
  for (const auto &depend_anfnode : old_depends) {
    auto depend_cnode = depend_anfnode->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(depend_cnode);
    auto depend_prior_node = depend_cnode->input(kControlDependPriorIndex);
    mng->Replace(depend_anfnode, depend_prior_node);
  }

  // Add new depend node in tail.
  AnfNodePtr patron_node = FindPatronNode(func_graph, mng);
  AddDepends(patron_node, free_nodes, func_graph, mng);
  return true;
}
}  // namespace opt
}  // namespace mindspore
