/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "plugin/ascend/graph_optimizer/pass/heterogeneous/insert_move_to.h"

#include <utility>
#include <stack>
#include <memory>
#include <algorithm>

#include "plugin/ascend/graph_optimizer/pass/heterogeneous/move_to_utils.h"
#include "primitive/framework_ops.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "include/utils/offload_context.h"
#include "include/utils/tensor_py.h"

namespace mindspore {
namespace opt {
namespace {
std::vector<std::pair<CNodePtr, size_t>> GetAllUserNode(const AnfNodePtr &node, int index,
                                                        const NodeUsersMap &node_users,
                                                        const mindspore::HashMap<CNodePtr, size_t> &node_exec_order) {
  std::vector<std::pair<CNodePtr, size_t>> ret;
  std::stack<std::pair<AnfNodePtr, int>> to_visit;
  std::stack<int> make_tuple_idx;
  to_visit.emplace(node, index);
  while (!to_visit.empty()) {
    auto [user, idx] = to_visit.top();
    to_visit.pop();
    if (IsPrimitiveCNode(user, prim::kPrimMakeTuple)) {
      const auto &iter = node_users.find(user);
      if (iter == node_users.end()) {
        continue;
      }
      for (const auto &node_idx : iter->second) {
        to_visit.push(node_idx);
      }
      make_tuple_idx.push(idx);
    } else if (IsPrimitiveCNode(user, prim::kPrimTupleGetItem)) {
      const auto get_item_idx = common::AnfAlgo::GetTupleGetItemOutIndex(user->cast<CNodePtr>());
      if (SizeToInt(get_item_idx) != make_tuple_idx.top()) {
        continue;
      }
      make_tuple_idx.pop();
      const auto &iter = node_users.find(user);
      if (iter == node_users.end()) {
        continue;
      }
      for (const auto &node_idx : iter->second) {
        to_visit.push(node_idx);
      }
    } else if (IsPrimitiveCNode(user, prim::kPrimDepend) || IsPrimitiveCNode(user, prim::kPrimLoad)) {
      if (idx != kIndex1) {
        continue;
      }
      const auto &iter = node_users.find(user);
      if (iter == node_users.end()) {
        continue;
      }
      for (const auto &node_idx : iter->second) {
        to_visit.push(node_idx);
      }
    } else {
      const auto &exec_idx_iter = node_exec_order.find(user->cast<CNodePtr>());
      if (exec_idx_iter == node_exec_order.end()) {
        continue;
      }
      ret.emplace_back(exec_idx_iter->first, exec_idx_iter->second);
    }
  }
  return ret;
}
}  // namespace

bool InsertMoveTo::Run(const FuncGraphPtr &graph) {
  Init(graph);
  // 1. Insert MoveTo and MoveAssign for offloaded parameter.
  bool changed = HandleParameter();

  // 2. Execution order by default
  kernel_graph_->SetExecOrderByDefault();
  return changed;
}

void InsertMoveTo::Init(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  func_graph_ = graph;
  kernel_graph_ = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph_);
  kernel_graph_->SetExecOrderByDefault();
  manager_ = kernel_graph_->manager();
  MS_EXCEPTION_IF_NULL(manager_);
}

bool InsertMoveTo::BackendInlineNode(const CNodePtr &node) {
  return common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimPartial) ||
         common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimPartialInline) ||
         common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimCall) ||
         common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimCallInline);
}

void InsertMoveTo::CollectOffloadedParameter() {
  const std::vector<CNodePtr> &exec_order = kernel_graph_->execution_order();
  mindspore::HashMap<CNodePtr, size_t> node_exec_order;
  for (size_t idx = 0; idx < exec_order.size(); idx += 1) {
    node_exec_order[exec_order[idx]] = idx;
  }
  const auto &node_users = manager_->node_users();
  const auto &parameters = kernel_graph_->parameters();
  for (const auto &node : parameters) {
    const auto &parameter = node->cast<ParameterPtr>();
    auto device_str = AnfAlgo::GetParameterDeviceStr(parameter);
    if (device_str.empty() || device_str != kToCpu) {
      continue;
    }
    const auto &users_iter = node_users.find(parameter);
    if (users_iter == node_users.end()) {
      continue;
    }
    for (const auto &[user_node, user_idx] : users_iter->second) {
      const auto &user_cnode = user_node->cast<CNodePtr>();
      if (user_cnode == nullptr) {
        continue;
      }
      if (BackendInlineNode(user_cnode)) {
        MS_LOG(WARNING) << "Skip backend inline node: " << user_cnode->DebugString();
        continue;
      }
      const auto &exec_iter = node_exec_order.find(user_node->cast<CNodePtr>());
      if (exec_iter != node_exec_order.end()) {
        const auto is_side_effect = common::AnfAlgo::HasNodeAttr(GRAPH_FLAG_SIDE_EFFECT_MEM, user_cnode) &&
                                    common::AnfAlgo::GetNodeAttr<bool>(user_cnode, GRAPH_FLAG_SIDE_EFFECT_MEM);
        OffloadParamInfo info{user_cnode,        IntToSize(user_idx), exec_iter->second,
                              exec_iter->second, is_side_effect,      device_str};
        MS_LOG(INFO) << "Offloaded parameter is used by " << user_cnode->fullname_with_scope()
                     << ", input index: " << user_idx << ", kernel execution order: " << exec_iter->second
                     << ", side effect: " << is_side_effect;
        offloaded_parameters_[parameter].emplace_back(info);
      } else {
        if (IsPrimitiveCNode(user_node, prim::kPrimLoad)) {
          auto depend_prim = NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name()));
          manager_->SetEdge(user_node, kIndex0, depend_prim);
          user_node->cast<CNodePtr>()->AddAttr("changed_from_load", MakeValue(true));
        }
        const auto &all_exec_user = GetAllUserNode(user_node, user_idx, node_users, node_exec_order);
        if (all_exec_user.empty()) {
          continue;
        }
        size_t first_execution_order = exec_order.size();
        size_t last_side_effect_execution_order = kIndex0;
        bool side_effect = false;
        for (const auto &[n, i] : all_exec_user) {
          if (i < first_execution_order) {
            first_execution_order = i;
          }
          const auto is_side_effect = common::AnfAlgo::HasNodeAttr(GRAPH_FLAG_SIDE_EFFECT_MEM, n) &&
                                      common::AnfAlgo::GetNodeAttr<bool>(n, GRAPH_FLAG_SIDE_EFFECT_MEM);
          if (is_side_effect && i > last_side_effect_execution_order) {
            last_side_effect_execution_order = i;
            side_effect = true;
          }
        }
        OffloadParamInfo info{user_cnode,  IntToSize(user_idx), first_execution_order, last_side_effect_execution_order,
                              side_effect, device_str};
        MS_LOG(INFO) << "Offloaded parameter is used by " << user_cnode->fullname_with_scope()
                     << ", input index: " << user_idx
                     << ", first user kernel execution order: " << first_execution_order << "["
                     << exec_order[first_execution_order] << "], side effect: " << side_effect
                     << (side_effect ? ", last side effect user: " +
                                         exec_order[last_side_effect_execution_order]->fullname_with_scope()
                                     : ".");
        offloaded_parameters_[parameter].emplace_back(info);
      }
    }
  }
}

CNodePtr InsertMoveTo::InsertParamMoveTo(const ParameterPtr &parameter, const OffloadParamInfo &info) const {
  MS_EXCEPTION_IF_NULL(parameter);
  // Get control previous and following node.
  const auto pre_load_execution_order_l =
    info.first_execution_order_ > load_lead_dh_ ? info.first_execution_order_ - load_lead_dh_ : 0;
  auto pre_node = kernel_graph_->execution_order()[pre_load_execution_order_l];
  if (pre_node == info.user_node_) {
    pre_node = nullptr;
  }
  const auto pre_load_execution_order_r = pre_load_execution_order_l + 1;
  const auto following_node = kernel_graph_->execution_order()[pre_load_execution_order_r];
  MS_EXCEPTION_IF_NULL(following_node);

  const MoveToInfo to_d_info{kToNpu, parameter, info.user_node_, info.input_index_, pre_node, following_node};

  auto move_to_d_node = MoveToUtils::InsertMoveTo(kernel_graph_, to_d_info);
  MS_LOG(INFO) << "Add MoveTo node[" << move_to_d_node->DebugString() << "] for " << info.input_index_ << "th input of "
               << info.user_node_->fullname_with_scope() << ".";

  if (info.offload_device_ == kToDisk) {
    const auto load_lead = load_lead_dh_ + load_lead_hf_;
    const auto l = info.first_execution_order_ > load_lead ? info.first_execution_order_ - load_lead : 0;
    const auto l_node = kernel_graph_->execution_order()[l];
    MS_EXCEPTION_IF_NULL(l_node);
    const auto r = l + 1;
    const auto r_node = kernel_graph_->execution_order()[r];
    MS_EXCEPTION_IF_NULL(r_node);

    const MoveToInfo to_h_info{kToCpu, parameter, move_to_d_node, 1, l_node, r_node};

    const auto move_to_h_node = MoveToUtils::InsertMoveTo(kernel_graph_, to_h_info);
    MS_LOG(INFO) << "Add MoveTo node[" << move_to_h_node->DebugString() << "] for " << info.input_index_
                 << "th input of " << info.user_node_->fullname_with_scope() << ".";
  }
  return move_to_d_node;
}

void InsertMoveTo::InsertParamMoveAssign(const ParameterPtr &parameter, const OffloadParamInfo &info,
                                         const CNodePtr &move_to) const {
  MS_EXCEPTION_IF_NULL(parameter);
  const auto &execution_order = kernel_graph_->execution_order();
  auto pre_node = execution_order[info.last_side_effect_execution_order_];
  MS_EXCEPTION_IF_NULL(pre_node);
  auto next_node = kernel_graph_->get_return();
  if (info.last_side_effect_execution_order_ + 1 < execution_order.size()) {
    next_node = execution_order[info.last_side_effect_execution_order_ + 1];
  }
  MS_EXCEPTION_IF_NULL(next_node);

  const MoveAssignInfo move_assign_info{info.offload_device_.c_str(), parameter, move_to, pre_node, next_node};
  const auto &move_assign_node = MoveToUtils::InsertMoveAssign(kernel_graph_, move_assign_info);
  MS_EXCEPTION_IF_NULL(move_assign_node);

  MS_LOG(INFO) << "Add MoveAssign node[" << move_assign_node->DebugString() << "] for " << info.input_index_
               << "th input of " << info.user_node_->fullname_with_scope() << ".";
}

bool InsertMoveTo::HandleParameter() {
  constexpr size_t kReuseThreshold = 100;
  CollectOffloadedParameter();
  if (offloaded_parameters_.empty()) {
    return false;
  }
  OffloadContext::GetInstance()->set_specific_param_offload(true);

  bool changed = false;
  struct MoveToInfo {
    OffloadParamInfo user_;
    CNodePtr move_to_;
    ParameterPtr parameter_;
  };
  std::vector<MoveToInfo> move_assign_to_insert;
  for (const auto &iter : offloaded_parameters_) {
    auto parameter = iter.first;
    MS_EXCEPTION_IF_NULL(parameter);
    CNodePtr move_to = nullptr;
    size_t pre_user_idx = kIndex0;
    OffloadParamInfo last_size_effect_user{nullptr, kIndex0, kIndex0, kIndex0, false, ""};
    auto offload_info = iter.second;
    const auto &compare = [](const OffloadParamInfo &l, const OffloadParamInfo &r) {
      return l.first_execution_order_ < r.first_execution_order_;
    };
    std::sort(offload_info.begin(), offload_info.end(), compare);
    for (const auto &user : offload_info) {
      if (move_to == nullptr || user.first_execution_order_ - pre_user_idx > kReuseThreshold) {
        move_to = InsertParamMoveTo(parameter, user);
      } else {
        manager_->SetEdge(user.user_node_, SizeToInt(user.input_index_), move_to);
      }
      pre_user_idx = user.first_execution_order_;
      if (user.side_effect_ &&
          user.last_side_effect_execution_order_ > last_size_effect_user.last_side_effect_execution_order_) {
        last_size_effect_user = user;
      }
      changed = true;
    }
    if (last_size_effect_user.user_node_ != nullptr) {
      kernel_graph_->ReplaceRefPair({parameter, 0}, {move_to, 0});
      MoveToInfo move_to_info{last_size_effect_user, move_to, parameter};
      move_assign_to_insert.emplace_back(move_to_info);
    }
  }
  for (const auto &item : move_assign_to_insert) {
    InsertParamMoveAssign(item.parameter_, item.user_, item.move_to_);
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
