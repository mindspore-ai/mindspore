/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "backend/optimizer/graph_kernel/shape_ops_splitter.h"
#include <algorithm>
#include <vector>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <queue>
#include <map>
#include <unordered_map>
#include "frontend/optimizer/irpass.h"
#include "pipeline/jit/parse/python_adapter.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/kernel_compiler/common_utils.h"
#include "backend/kernel_compiler/akg/akg_kernel_json_decoder.h"
#include "backend/optimizer/graph_kernel/graph_kernel_helper.h"
#include "debug/anf_ir_dump.h"

namespace mindspore {
namespace opt {
namespace {
AnfNodePtr CloneCNode(const AnfNodePtr &anf_node) {
  auto func_graph = anf_node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  TraceGuard guard(std::make_shared<TraceOpt>(cnode->debug_info()));
  CNodePtr node = func_graph->NewCNode(cnode->inputs());
  node->set_abstract(cnode->abstract());
  node->set_forward(cnode->forward().first, cnode->forward().second);
  node->set_inputs_value(cnode->inputs_value());
  ScopePtr scope = (anf_node->scope() != kDefaultScope) ? anf_node->scope() : kDefaultScope;
  node->set_scope(scope);
  node->set_kernel_info(cnode->kernel_info_ptr());
  return node;
}

void SplitNode(const AnfNodePtr &node, const FuncGraphManagerPtr &mng) {
  const auto &index_set = mng->node_users()[node];
  std::map<AnfNodePtr, std::vector<int>> users_info;
  std::for_each(index_set.cbegin(), index_set.cend(), [&users_info](const std::pair<AnfNodePtr, int> &iter) {
    users_info[iter.first].push_back(iter.second);
  });

  AnfNodePtrList split_nodes;
  for (size_t i = 0; i < users_info.size(); ++i) {
    split_nodes.push_back(CloneCNode(node));
  }

  int i = 0;
  for (auto [user, indices] : users_info) {
    auto user_node = user->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(user_node);
    for (auto index : indices) {
      user_node->set_input(index, split_nodes[i]);
    }
    i++;
  }
}
}  // namespace

bool ShapeOpsSplitter::IsMultiUserShapeOps(const AnfNodePtr &node, const FuncGraphManagerPtr &mng) {
  auto &users = mng->node_users();
  std::set<AnfNodePtr> user_set;
  std::transform(users[node].cbegin(), users[node].cend(), std::inserter(user_set, user_set.end()),
                 [](const std::pair<AnfNodePtr, int> &iter) { return iter.first; });
  return user_set.size() > 1 && std::any_of(shape_ops_.begin(), shape_ops_.end(),
                                            [&node](const PrimitivePtr &prim) { return IsPrimitiveCNode(node, prim); });
}

bool ShapeOpsSplitter::Process(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }
  bool changed = false;
  auto todos = TopoSort(func_graph->get_return());
  for (const auto &anf_node : todos) {
    auto node = anf_node->cast<CNodePtr>();
    if (node != nullptr && IsMultiUserShapeOps(node, mng)) {
      SplitNode(node, mng);
      changed = true;
    }
  }
  if (changed) {
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }
  return changed;
}

bool ShapeOpsSplitter::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }

  auto todos = TopoSort(func_graph->get_return());
  bool result = false;
  for (const auto &anf_node : todos) {
    if (AnfAlgo::IsGraphKernel(anf_node)) {
      auto sub_graph = AnfAlgo::GetCNodeFuncGraphPtr(anf_node);
      bool changed = false;
      do {
        changed = Process(sub_graph);
        result = result || changed;
      } while (changed);
    }
  }

  if (result) {
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }
  return result;
}
}  // namespace opt
}  // namespace mindspore
