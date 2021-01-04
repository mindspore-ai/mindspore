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
#include "backend/optimizer/graph_kernel/shape_ops_splitter.h"
#include <algorithm>
#include <vector>
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
  auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(anf_node->func_graph());
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  TraceGuard guard(std::make_shared<TraceOpt>(cnode->debug_info()));
  CNodePtr node = kernel_graph->NewCNode(cnode->inputs());
  node->set_abstract(cnode->abstract());
  node->set_forward(cnode->forward().first, cnode->forward().second);
  node->set_inputs_value(cnode->inputs_value());
  ScopePtr scope = (anf_node->scope() != kDefaultScope) ? anf_node->scope() : kDefaultScope;
  node->set_scope(scope);
  node->set_kernel_info(cnode->kernel_info_ptr());
  return node;
}

void SplitNode(const AnfNodePtr &node, const FuncGraphManagerPtr &mng) {
  auto &users = mng->node_users();
  AnfNodePtrList splitted_nodes;
  for (size_t i = 0; i < users[node].size(); ++i) {
    splitted_nodes.push_back(CloneCNode(node));
  }

  const auto &index_set = users[node];
  int i = 0;
  for (auto [user, index] : index_set) {
    auto user_node = user->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(user_node);
    user_node->set_input(index, splitted_nodes[i]);
    i++;
  }
}
}  // namespace

bool ShapeOpsSplitter::IsMultiUserShapeOps(const AnfNodePtr &node, const FuncGraphManagerPtr &mng) {
  auto &users = mng->node_users();
  return users[node].size() > 1 && std::any_of(shape_ops_.begin(), shape_ops_.end(), [&node](const PrimitivePtr &prim) {
           return IsPrimitiveCNode(node, prim);
         });
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

  mng->RemoveRoots();
  mng->KeepRoots({func_graph});
  return changed;
}

bool ShapeOpsSplitter::Run(const FuncGraphPtr &func_graph) {
  bool result = false;
  bool changed;
  do {
    changed = Process(func_graph);
    result |= changed;
  } while (changed);
  return result;
}
}  // namespace opt
}  // namespace mindspore
