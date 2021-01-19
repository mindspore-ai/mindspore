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
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/kernel_compiler/common_utils.h"
#include "backend/optimizer/graph_kernel/graph_kernel_helper.h"

namespace mindspore {
namespace opt {
namespace {
std::tuple<AnfNodePtr, AnfNodePtr, int> FindPatronNode(const FuncGraphPtr &main_graph, const FuncGraphManagerPtr &mng) {
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

  auto &user_nodes = mng->node_users()[patron_node];
  auto user = user_nodes.begin();
  return std::make_tuple(patron_node, user->first, user->second);
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

  auto nodes = TopoSort(func_graph->get_return());
  AnfNodePtrList old_depends;
  AnfNodePtrList free_nodes;

  // Find depend and its free nodes.
  for (const auto &node : nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimDepend)) {
      continue;
    }

    old_depends.push_back(node);
    free_nodes.push_back(node->cast<CNodePtr>()->input(kDependAttachNodeIndex));
  }

  if (old_depends.empty()) {
    return false;
  }

  // Delete old depend.
  for (const auto &depend_anfnode : old_depends) {
    auto depend_cnode = depend_anfnode->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(depend_cnode);
    auto depend_prior_node = depend_cnode->input(kControlDependPriorIndex);
    mng->Replace(depend_anfnode, depend_prior_node);
  }

  // Add new depend node in tail.
  AnfNodePtr patron_node;
  std::tie(patron_node, std::ignore, std::ignore) = FindPatronNode(func_graph, mng);
  AddDepends(patron_node, free_nodes, func_graph, mng);

  return true;
}
}  // namespace opt
}  // namespace mindspore
