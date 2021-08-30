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
#include "backend/optimizer/graph_kernel/eliminate_redundant_complex.h"

#include <algorithm>
#include <vector>
#include <string>
#include <utility>

#include "frontend/optimizer/irpass.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/kernel_compiler/common_utils.h"
#include "debug/anf_ir_dump.h"

namespace mindspore {
namespace opt {
namespace {
bool EliminateRedudantComplexInGraphkernel(const FuncGraphPtr &func_graph) {
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  auto todos = TopoSort(func_graph->get_return());
  bool changed = false;
  for (const auto &node : todos) {
    auto cnode = node->cast<CNodePtr>();
    // Find all Complex node in graphkernel sub_graph
    if (cnode != nullptr && IsPrimitiveCNode(cnode, std::make_shared<Primitive>("Complex"))) {
      auto original_users = mng->node_users()[cnode];
      for (const auto &getitem_iter : original_users) {
        auto getitem = getitem_iter.first;
        auto getitem_cnode = getitem->cast<CNodePtr>();
        // Find all complex users which are CReal or CImag, then use Complex inputs replace them.
        if (IsPrimitiveCNode(getitem_cnode, std::make_shared<Primitive>("CReal"))) {
          (void)mng->Replace(getitem, cnode->inputs()[1]);
          changed = true;
        } else if (IsPrimitiveCNode(getitem_cnode, std::make_shared<Primitive>("CImag"))) {
          (void)mng->Replace(getitem, cnode->inputs()[2]);
          changed = true;
        }
      }
    }
  }
  return changed;
}
}  // namespace

bool EliminateRedundantComplex::Run(const FuncGraphPtr &func_graph) {
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  bool changed = false;
  auto todos = TopoSort(func_graph->get_return());
  std::reverse(todos.begin(), todos.end());
  for (const auto &node : todos) {
    auto cnode = node->cast<CNodePtr>();
    // Check whether graph_kernel node
    if (cnode != nullptr && AnfAlgo::IsGraphKernel(cnode)) {
      auto graph_kernel_fg = AnfAlgo::GetCNodeFuncGraphPtr(cnode);
      MS_EXCEPTION_IF_NULL(graph_kernel_fg);
      changed = EliminateRedudantComplexInGraphkernel(graph_kernel_fg) || changed;
    }
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
