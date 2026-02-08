/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "plugin/ascend/graph_optimizer/pass/heterogeneous/insert_pre_fetch_depend.h"

#include <queue>
#include <utility>
#include <stack>

#include "include/utils/anfalgo.h"
#include "include/backend/common/pass_manager/helper.h"
#include "plugin/ascend/graph_optimizer/pass/heterogeneous/move_to_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/framework_ops.h"

namespace mindspore {
namespace opt {
void InsertPreFetchDepend::Init(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  func_graph_ = graph;
  kernel_graph_ = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph_);
  manager_ = graph->manager();
  MS_EXCEPTION_IF_NULL(manager_);

  kernel_graph_->SetExecOrderByDefault();
  MakeExecOrderCache();
}

void InsertPreFetchDepend::MakeExecOrderCache() {
  const auto &execution_order = kernel_graph_->execution_order();
  size_t j = 0;
  for (size_t i = 0; i < execution_order.size(); ++i) {
    exec_order_cache_[execution_order[i]] = i;
    if (!IsPrimitiveCNode(execution_order[i], prim::kPrimMoveTo)) {
      exec_order_cache_without_moveto_[execution_order[i]] = j++;
      exec_order_without_moveto_.emplace_back(execution_order[i]);
    }
  }
}

size_t InsertPreFetchDepend::CalPreFetchCeiling(const CNodePtr &move_to_node) {
  size_t ceiling = 0;
  const auto &move_to_input =
    common::AnfAlgo::VisitKernelWithReturnType(move_to_node->input(kIndex1), kIndex0, true, {prim::kPrimDepend});
  const auto &move_to_input_node = move_to_input.first;
  if (move_to_input_node == nullptr || !move_to_input_node->isa<CNode>()) {
    return ceiling;
  }

  std::queue<CNodePtr> to_visit;
  to_visit.push(move_to_input_node->cast<CNodePtr>());
  while (!to_visit.empty()) {
    const auto &node = to_visit.front();
    to_visit.pop();
    if (node == nullptr) {
      continue;
    }
    const auto is_depend = IsPrimitiveCNode(node, prim::kPrimDepend);
    if (is_depend) {
      for (size_t i = kIndex1; i <= kIndex2; i += 1) {
        const auto &input =
          common::AnfAlgo::VisitKernelWithReturnType(node->input(i), kIndex0, true, {prim::kPrimDepend});
        const auto &input_node = input.first;
        if (input_node == nullptr || !input_node->isa<CNode>()) {
          continue;
        }
        to_visit.push(input_node->cast<CNodePtr>());
      }
    } else {
      const auto &exec_order_iter = exec_order_cache_.find(node);
      if (exec_order_iter != exec_order_cache_.end()) {
        ceiling = exec_order_iter->second > ceiling ? exec_order_iter->second : ceiling;
      }
    }
  }
  return ceiling;
}

size_t InsertPreFetchDepend::GetFirstUserExecOrder(const mindspore::CNodePtr &move_to_node) {
  const auto users = manager_->node_users()[move_to_node];
  if (users.empty()) {
    return false;
  }
  size_t bw_node_exec_order = SIZE_MAX;
  std::stack<std::pair<AnfNodePtr, int>> to_visit;
  std::stack<int> make_tuple_stack;
  for (const auto &user : users) {
    to_visit.push(user);
  }
  while (!to_visit.empty()) {
    auto user = to_visit.top();
    to_visit.pop();
    if (!user.first->isa<CNode>()) {
      continue;
    }
    auto user_node = user.first->cast<CNodePtr>();
    if (IsPrimitiveCNode(user_node, prim::kPrimDepend) && user.second == kIndex1) {
      for (const auto &depend_user : manager_->node_users()[user_node]) {
        to_visit.push(depend_user);
      }
      continue;
    }
    if (IsPrimitiveCNode(user_node, prim::kPrimMakeTuple)) {
      make_tuple_stack.push(user.second);
      for (const auto &make_tuple_user : manager_->node_users()[user_node]) {
        to_visit.push(make_tuple_user);
      }
      continue;
    }
    if (IsPrimitiveCNode(user_node, prim::kPrimTupleGetItem)) {
      const auto get_item_idx = GetGetitemIndex(user_node);
      if (get_item_idx != make_tuple_stack.top()) {
        continue;
      }
      for (const auto &get_item_user : manager_->node_users()[user_node]) {
        to_visit.push(get_item_user);
      }
      make_tuple_stack.pop();
      continue;
    }

    const auto &exec_order_iter = exec_order_cache_.find(user_node);
    if (exec_order_iter == exec_order_cache_.end()) {
      continue;
    }
    const auto exec_order = exec_order_iter->second;
    if (exec_order < bw_node_exec_order) {
      bw_node_exec_order = exec_order;
    }
  }
  return bw_node_exec_order;
}

bool InsertPreFetchDepend::CalExecutionOrder(const CNodePtr &move_to_node, int64_t prefetch, size_t *pre_exec_order,
                                             size_t *post_exec_order) {
  const auto bw_node_exec_order = GetFirstUserExecOrder(move_to_node);
  if (prefetch + 1 > SizeToInt(bw_node_exec_order)) {
    return false;
  }
  if (prefetch + 1 == SizeToInt(bw_node_exec_order) && exec_order_cache_[move_to_node] == 0) {
    return false;
  }
  const auto execution_order = kernel_graph_->execution_order();
  const auto &bw_node = execution_order[bw_node_exec_order];
  MS_EXCEPTION_IF_NULL(bw_node);
  const auto prefetch_ceiling = CalPreFetchCeiling(move_to_node);
  const auto bw_node_without_moveto = exec_order_cache_without_moveto_[bw_node];
  auto pre_order_without_moveto = bw_node_without_moveto - 1 - prefetch;
  auto pre_node = exec_order_without_moveto_[pre_order_without_moveto];
  auto pre_order = exec_order_cache_[pre_node];
  if (pre_order < prefetch_ceiling) {
    pre_order = prefetch_ceiling;
    while (IsPrimitiveCNode(execution_order[pre_order], prim::kPrimMoveTo)) {
      pre_order += 1;
    }
    pre_node = execution_order[pre_order];
    pre_order_without_moveto = exec_order_cache_without_moveto_[pre_node];
  }
  auto post_order_without_move_to = pre_order_without_moveto + 1;

  *pre_exec_order = pre_order_without_moveto;
  *post_exec_order = post_order_without_move_to;
  return true;
}

void InsertPreFetchDepend::InsertDepend(const CNodePtr &move_to_node, int64_t prefetch) {
  MS_EXCEPTION_IF_NULL(move_to_node);
  size_t pre_exec_order;
  size_t post_exec_order;
  if (!CalExecutionOrder(move_to_node, prefetch, &pre_exec_order, &post_exec_order)) {
    return;
  }
  const auto &pre_node = exec_order_without_moveto_[pre_exec_order];
  MS_EXCEPTION_IF_NULL(pre_node);
  const auto &post_node = exec_order_without_moveto_[post_exec_order];
  MS_EXCEPTION_IF_NULL(post_node);
  if (MoveToUtils::InsertDependNode(kernel_graph_, pre_node, move_to_node) == nullptr) {
    MS_LOG(WARNING) << "Insert depend for pre fetch " << move_to_node->fullname_with_scope()
                    << " failed. Prefetch: " << prefetch << ", pre node: " << pre_node->DebugString();
  }
  if (MoveToUtils::InsertDependNode(kernel_graph_, move_to_node, post_node) == nullptr) {
    MS_LOG(WARNING) << "Insert depend for pre fetch " << move_to_node->fullname_with_scope()
                    << " failed. Prefetch: " << prefetch << ", post node: " << post_node->DebugString();
  }
}

bool InsertPreFetchDepend::Run(const FuncGraphPtr &graph) {
  Init(graph);
  bool changed = false;
  const auto &execution_order = kernel_graph_->execution_order();
  for (const auto &cnode : execution_order) {
    if (!IsPrimitiveCNode(cnode, prim::kPrimMoveTo) || common::AnfAlgo::GetMoveToDstStr(cnode) != kToNpu) {
      continue;
    }
    const auto &prim = GetCNodePrimitive(cnode);
    MS_EXCEPTION_IF_NULL(prim);
    const auto &attr_iter = prim->attrs().find(kAttrBackwardPrefetch);
    if (attr_iter == prim->attrs().end()) {
      continue;
    }
    const auto &pre_fetch_value = attr_iter->second;
    if (!pre_fetch_value->isa<Int64Imm>()) {
      continue;
    }
    const auto pre_fetch = GetValue<int64_t>(pre_fetch_value);
    InsertDepend(cnode, pre_fetch);
    changed = true;
  }
  return changed;
}

}  // namespace opt
}  // namespace mindspore
