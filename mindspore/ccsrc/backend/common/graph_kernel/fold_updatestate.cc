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
#include "backend/common/graph_kernel/fold_updatestate.h"
#include <vector>
#include <utility>
#include "mindspore/core/ops/nn_optimizer_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "abstract/utils.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore::graphkernel {
constexpr auto USER_NUM = 1;
constexpr auto REAL_INPUT_START_IDX = 2;

std::pair<AnfNodePtr, AnfNodePtrList> IsTargetUpdateState(const AnfNodePtr &node, const FuncGraphManagerPtr &mng) {
  auto users = mng->node_users()[node];
  if (users.size() != USER_NUM || (!IsPrimitiveCNode(users.front().first, prim::kPrimUpdateState) &&
                                   !IsPrimitiveCNode(users.front().first, prim::kPrimDepend))) {
    return std::make_pair(nullptr, AnfNodePtrList{});
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto inputs = cnode->inputs();
  AnfNodePtrList graph_kernel_inputs;
  for (size_t i = REAL_INPUT_START_IDX; i < cnode->size(); i++) {
    if (common::AnfAlgo::IsGraphKernel(inputs[i])) {
      graph_kernel_inputs.push_back(inputs[i]);
    } else if (IsPrimitiveCNode(inputs[i], prim::kPrimTupleGetItem)) {
      auto real_input = inputs[i]->cast<CNodePtr>()->input(1);
      if (common::AnfAlgo::IsGraphKernel(real_input)) {
        graph_kernel_inputs.push_back(real_input);
      }
    }
  }
  if (graph_kernel_inputs.size() == 0) {
    return std::make_pair(nullptr, graph_kernel_inputs);
  } else {
    return std::make_pair(node, graph_kernel_inputs);
  }
}

bool Parallelize(const AnfNodePtrList &updatestates, const FuncGraphManagerPtr &mng) {
  auto last_updatestate = updatestates.back()->cast<CNodePtr>();
  auto first_updatestate = updatestates.front()->cast<CNodePtr>()->input(1)->cast<CNodePtr>();
  AnfNodePtrList additional_inputs;
  for (size_t i = 0; i < updatestates.size() - 1; i++) {
    auto updatestate = updatestates[i]->cast<CNodePtr>();
    for (size_t input_idx = REAL_INPUT_START_IDX; input_idx < updatestate->size(); input_idx++) {
      additional_inputs.push_back(updatestate->input(input_idx));
    }
  }
  if (first_updatestate != nullptr) {
    last_updatestate->set_input(1, first_updatestate);
  } else {
    auto u = NewValueNode(kUMonad);
    u->set_abstract(kUMonad->ToAbstract());
    last_updatestate->set_input(1, u);
  }
  AnfNodePtrList final_inputs = last_updatestate->inputs();
  final_inputs.insert(final_inputs.cend(), additional_inputs.cbegin(), additional_inputs.cend());
  last_updatestate->set_inputs(final_inputs);
  return true;
}

bool FoldUpdateState::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(func_graph->get_return());
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  bool graph_change = false;

  bool changed = false;
  std::vector<AnfNodePtr> updatestates;
  auto todos = TopoSort(func_graph->get_return());

  for (auto &node : todos) {
    if (IsPrimitiveCNode(node, prim::kPrimUpdateState)) {
      auto res = IsTargetUpdateState(node, mng);
      if (res.first == nullptr) {
        if (updatestates.size() != 0) {
          changed = Parallelize(updatestates, mng);
          updatestates.clear();
        }
        continue;
      }
      updatestates.push_back(node);
    }
  }
  if (updatestates.size() != 0) {
    changed = Parallelize(updatestates, mng);
  }
  if (changed) {
    graph_change = true;
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }
  return graph_change;
}
}  // namespace mindspore::graphkernel
