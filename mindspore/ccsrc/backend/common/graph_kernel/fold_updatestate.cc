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
#include <stack>
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

AnfNodePtr GetTailDepend(const FuncGraphPtr &func_graph) {
  const auto &return_node = func_graph->get_return();
  auto node = return_node->input(kIndex1);
  while (IsPrimitiveCNode(node, prim::kPrimMakeTuple) && node->cast<CNodePtr>()->inputs().size() == kSizeTwo) {
    node = node->cast<CNodePtr>()->input(kIndex1);
  }
  return IsPrimitiveCNode(node, prim::kPrimDepend) ? node : nullptr;
}

AnfNodePtr FindMakeTupleRealInput(const AnfNodePtr &node) {
  const auto &maketuple = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(maketuple);
  const auto &inputs = maketuple->inputs();
  AnfNodePtr getitem_input{nullptr};
  for (size_t i = kIndex1; i < inputs.size(); i++) {
    if (!IsPrimitiveCNode(inputs[i], prim::kPrimTupleGetItem)) {
      return nullptr;
    }
    auto tuplegetitem = inputs[i]->cast<CNodePtr>();
    if (getitem_input == nullptr) {
      getitem_input = tuplegetitem->input(kIndex1);
    }
    if (getitem_input != tuplegetitem->input(kIndex1)) {
      return nullptr;
    }
  }
  return getitem_input;
}

bool TailDependProcess(const AnfNodePtr &tail_depend) {
  const auto &depend_node = tail_depend->cast<CNodePtr>();
  const auto &input_node = depend_node->input(kIndex1);
  const auto &attach_node = depend_node->input(kIndex2);
  if (!IsPrimitiveCNode(attach_node, prim::kPrimUpdateState)) {
    MS_LOG(DEBUG) << "For node " << tail_depend->fullname_with_scope()
                  << ", attach node is not UpdateState, match pattern failed";
    return false;
  }
  const auto &updatestate = attach_node->cast<CNodePtr>();
  auto &updatestate_inputs = updatestate->inputs();
  if (!std::all_of(updatestate_inputs.begin() + kIndex2, updatestate_inputs.end(),
                   [](const AnfNodePtr &n) { return IsPrimitiveCNode(n, prim::kPrimTupleGetItem); })) {
    MS_LOG(DEBUG) << "For node " << tail_depend->fullname_with_scope()
                  << ", inputs of UpdateState are not all TupleGetItems, match pattern failed";
    return false;
  }
  if (!IsPrimitiveCNode(input_node, prim::kPrimMakeTuple)) {
    MS_LOG(DEBUG) << "For node " << tail_depend->fullname_with_scope()
                  << ", input node is not MakeTuple, match pattern failed";
    return false;
  }
  HashSet<AnfNodePtr> updatestate_real_inputs;
  for (size_t i = kIndex2; i < updatestate_inputs.size(); i++) {
    updatestate_real_inputs.insert(updatestate_inputs[i]->cast<CNodePtr>()->input(kIndex1));
  }

  const auto &maketuple = input_node->cast<CNodePtr>();
  HashSet<AnfNodePtr> maketuple_real_inputs;
  for (size_t i = kIndex1; i < maketuple->inputs().size(); i++) {
    if (!IsPrimitiveCNode(maketuple->input(i), prim::kPrimMakeTuple)) {
      MS_LOG(DEBUG) << "For node " << tail_depend->fullname_with_scope()
                    << ", inputs node of MakeTuple are not all MakeTuples, match pattern failed";
      return false;
    }
    auto input_maketuple = maketuple->input(i)->cast<CNodePtr>();
    auto real_input = FindMakeTupleRealInput(input_maketuple);
    if (real_input == nullptr) {
      MS_LOG(DEBUG) << "For node " << tail_depend->fullname_with_scope() << ", find real input of MakeTuple failed";
      return false;
    }
    maketuple_real_inputs.insert(real_input);
  }
  if (updatestate_real_inputs != maketuple_real_inputs) {
    return false;
  }
  AnfNodePtrList new_inputs;
  new_inputs.emplace_back(updatestate_inputs[kIndex0]);
  new_inputs.emplace_back(updatestate_inputs[kIndex1]);
  updatestate->set_inputs(new_inputs);
  return true;
}

AnfNodePtrList FindAssignedParams(const CNodePtr &gk_node) {
  const auto &args = gk_node->inputs();
  const auto &fg = GetCNodeFuncGraph(gk_node);
  const auto &params = fg->parameters();
  const auto &nodes = TopoSort(fg->get_return());
  AnfNodePtrList assigned_params;
  for (auto &node : nodes) {
    if (IsPrimitiveCNode(node, prim::kPrimAssign)) {
      const auto &param = node->cast<CNodePtr>()->input(kIndex1);
      for (size_t i = 0; i < params.size(); i++) {
        if (params[i] == param) {
          assigned_params.push_back(args[i + 1]);
        }
        break;
      }
    }
  }
  return assigned_params;
}

bool RemoveRedundantDepends(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &mng) {
  const auto &todos = TopoSort(func_graph->get_return());
  auto &users = mng->node_users();
  for (auto &node : todos) {
    if (IsPrimitiveCNode(node, prim::kPrimDepend)) {
      const auto &depend_cnode = node->cast<CNodePtr>();
      if (depend_cnode->size() != kSizeThree) {
        continue;
      }
      if (users[depend_cnode].size() != 1 || !common::AnfAlgo::IsGraphKernel(users[depend_cnode].front().first)) {
        continue;
      }
      const auto &attach_node = depend_cnode->input(kIndex2);
      if (attach_node->isa<ValueNode>() || attach_node->isa<Parameter>()) {
        mng->Replace(depend_cnode, depend_cnode->input(kIndex1));
        continue;
      }
      const auto &attach_cnode = attach_node->cast<CNodePtr>();
      if (users[attach_cnode].size() == 1) {
        continue;
      }
      MS_EXCEPTION_IF_NULL(attach_cnode);
      const auto &gk_node = users[depend_cnode].front().first->cast<CNodePtr>();
      const auto &assigned_params = FindAssignedParams(gk_node);
      HashSet<AnfNodePtr> used_params;
      std::stack<CNodePtr> st;
      HashSet<AnfNodePtr> visited;
      st.push(attach_cnode);
      visited.insert(attach_cnode);
      while (!st.empty()) {
        auto n = st.top();
        st.pop();
        size_t size = n->size();
        for (size_t i = 1; i < size; i++) {
          auto input = n->input(i);
          if (input->isa<CNode>() && visited.find(input) == visited.end()) {
            st.push(input->cast<CNodePtr>());
            visited.insert(input);
          } else if (input->isa<Parameter>()) {
            used_params.insert(input);
          }
        }
      }
      bool is_redundant_depend{true};
      for (auto assigned_param : assigned_params) {
        if (used_params.find(assigned_param) != used_params.end()) {
          is_redundant_depend = false;
          break;
        }
      }
      if (is_redundant_depend) {
        MS_LOG(DEBUG) << depend_cnode->fullname_with_scope()
                      << "is redundant, it can be replaced by its input directly";
        mng->Replace(depend_cnode, depend_cnode->input(kIndex1));
      }
    }
  }
  return true;
}

bool FoldUpdateState::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(func_graph->get_return());
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  bool changed = false;
  std::vector<AnfNodePtr> updatestates;
  auto todos = TopoSort(func_graph->get_return());
  changed = RemoveRedundantDepends(func_graph, mng);
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
  auto tail_depend = GetTailDepend(func_graph);
  if (tail_depend) {
    changed = TailDependProcess(tail_depend) || changed;
  }
  if (changed) {
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }
  return changed;
}
}  // namespace mindspore::graphkernel
