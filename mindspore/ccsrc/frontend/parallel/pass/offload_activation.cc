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

#include "frontend/parallel/pass/offload_activation.h"

#include <vector>

#include "frontend/parallel/step_parallel_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "mindspore/core/include/ir/core_ops_primitive.h"
#include "primitive/framework_ops.h"
#include "ir/func_graph_flag.h"
#include "ir/graph_utils.h"

namespace mindspore {
namespace parallel {
constexpr size_t kReturnIndex = 1;
constexpr int64_t kDefaultPrefetch = 1;
constexpr char kAttrOffloadActivationHandled[] = "offload_activation_handled";

bool OffloadActivation(const FuncGraphPtr &func_graph) {
  OffloadActivationOptimizer optimizer;
  return optimizer.Optimize(func_graph);
}

FuncGraphPtr OffloadActivationOptimizer::GetBackwardGraph(const FuncGraphPtr &func_graph) {
  // %make_tuple = MakeTuple(%x, @Backward_Graph)
  // Return(%make_tuple)
  const auto &return_node = func_graph->get_return();
  if (return_node == nullptr) {
    return nullptr;
  }
  const auto &return_input = return_node->input(kReturnIndex);
  if (return_input == nullptr || !return_input->isa<CNode>()) {
    return nullptr;
  }
  const auto &make_tuple = return_input->cast<CNodePtr>();
  if (make_tuple == nullptr || !IsPrimitiveCNode(make_tuple, prim::kPrimMakeTuple)) {
    return nullptr;
  }
  const auto &tuple_last_input = make_tuple->input(make_tuple->inputs().size() - 1);
  if (tuple_last_input == nullptr || !IsValueNode<FuncGraph>(tuple_last_input)) {
    return nullptr;
  }
  return GetValueNode<FuncGraphPtr>(tuple_last_input);
}

void OffloadActivationOptimizer::GetFwBwGraphs() {
  for (const auto &child_graph : manager_->func_graphs()) {
    if (child_graph->has_flag(FUNC_GRAPH_FLAG_CELL_REUSE)) {
      const auto &all_nodes = TopoSort(child_graph->get_return(), SuccDeeperSimple);
      for (const auto &node : all_nodes) {
        if (!node->isa<CNode>()) {
          continue;
        }
        const auto &fw_cnode = node->cast<CNodePtr>();
        const auto &fw_primitive = GetCNodePrimitive(fw_cnode);
        if (!GetPrimitiveFlag(fw_primitive, kAttrOffload)) {
          continue;
        }
        offload_in_lazy_inline_.insert(fw_cnode);
      }
    }
    const auto &backward_graph = GetBackwardGraph(child_graph);
    if (backward_graph == nullptr || backward_graph->has_flag(kAttrOffloadActivationHandled)) {
      continue;
    }
    fw_bw_graphs_.emplace_back(child_graph, backward_graph);
  }
}

void OffloadActivationOptimizer::AddOffloadForCommUser(const FuncGraphPtr &fw_graph) {
  const auto &cnodes = fw_graph->GetOrderedCnodes();
  for (const auto &fw_cnode : cnodes) {
    const auto &fw_primitive = GetCNodePrimitive(fw_cnode);
    if (!GetPrimitiveFlag(fw_primitive, kAttrOffload)) {
      continue;
    }
    const auto &users_iter = manager_->node_users().find(fw_cnode);
    if (users_iter == manager_->node_users().end()) {
      continue;
    }
    for (const auto &user : users_iter->second) {
      const auto &user_node = user.first;
      if (user_node == nullptr || user_node->func_graph() != fw_graph) {
        continue;
      }
      const auto &user_prim = GetCNodePrimitive(user_node);
      if (user_prim == nullptr || !IsCommunicationOp(user_prim)) {
        continue;
      }
      user_prim->AddAttr(kAttrOffload, MakeValue(true));
      if (fw_primitive->HasAttr(kAttrBackwardPrefetch)) {
        user_prim->AddAttr(kAttrBackwardPrefetch, fw_primitive->GetAttr(kAttrBackwardPrefetch));
      }
    }
  }
}

void OffloadActivationOptimizer::GetActivationOffloadInfo(const FuncGraphPtr &fw_graph, const FuncGraphPtr &bw_grpah) {
  const auto &cnodes = bw_grpah->GetOrderedCnodes();
  for (const auto &bw_cnode : cnodes) {
    for (size_t idx = 0; idx < bw_cnode->inputs().size(); ++idx) {
      const auto &input = bw_cnode->input(idx);
      if (!input->isa<CNode>() || input->func_graph() != fw_graph) {
        continue;
      }
      const auto &fw_cnode = input->cast<CNodePtr>();
      const auto &fw_primitive = GetCNodePrimitive(fw_cnode);
      if (!GetPrimitiveFlag(fw_primitive, kAttrOffload)) {
        continue;
      }
      if (GetPrimitiveFlag(fw_primitive, kAttrRecompute) || fw_graph->has_flag(FUNC_GRAPH_RECOMPUTE_K_GRAPH)) {
        MS_LOG(WARNING) << "Node can not be recomputed and offloaded at the same time, ignore offload fot it. Node: "
                        << trace::GetDebugInfoStr(fw_cnode->debug_info());
        continue;
      }
      int64_t prefetch = kDefaultPrefetch;
      const auto prefetch_value = fw_primitive->GetAttr(kAttrBackwardPrefetch);
      if (prefetch_value != nullptr && prefetch_value->isa<Int64Imm>()) {
        prefetch = GetValue<int64_t>(prefetch_value);
      }
      auto offload_info = std::make_shared<OffloadInfo>(fw_graph, fw_cnode, bw_grpah, bw_cnode, idx, prefetch);
      offload_infos_.emplace_back(offload_info);
    }
  }
}

void OffloadActivationOptimizer::AddDependForMoveOut(const FuncGraphPtr &fw_graph, const CNodePtr &fw_node,
                                                     const CNodePtr &move_out) {
  /* Start copy out as sonn as possible.
  forward node o   o to ValueNode
              / \ /
             |   o MoveTo(Out)
              \/
       depend o
               \
                o user
  */
  MS_EXCEPTION_IF_NULL(fw_graph);
  MS_EXCEPTION_IF_NULL(fw_node);
  MS_EXCEPTION_IF_NULL(move_out);
  const auto user_iter = manager_->node_users().find(fw_node);
  if (user_iter == manager_->node_users().end() || user_iter->second.empty()) {
    return;
  }
  for (const auto &user : user_iter->second) {
    const auto &user_node = user.first;
    if (user_node == nullptr || !user_node->isa<CNode>() || user_node == move_out ||
        user_node->func_graph() != fw_node->func_graph()) {
      continue;
    }
    const std::vector<AnfNodePtr> depend_input = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                                  fw_node, move_out};
    auto depend_node = fw_graph->NewCNode(depend_input);
    MS_EXCEPTION_IF_NULL(depend_node);
    depend_node->set_abstract(fw_node->abstract());
    depend_node->set_scope(fw_node->scope());
    manager_->SetEdge(user_node, user.second, depend_node);
    MS_LOG(DEBUG) << "Insert Depend for MoveTo(Out)" << move_out->DebugString() << " before "
                  << user_node->DebugString();
    return;
  }
}

void OffloadActivationOptimizer::InsertMoveToForOffloadActivation(const OffloadInfoPtr &offload_info) {
  MS_EXCEPTION_IF_NULL(offload_info);

  const bool is_tuple_get_item = IsPrimitiveCNode(offload_info->bw_node_, prim::kPrimTupleGetItem);
  if (is_tuple_get_item) {
    const auto get_item_idx = common::AnfAlgo::GetTupleGetItemOutIndex(offload_info->bw_node_);
    const std::vector<AnfNodePtr> get_item_inputs = {NewValueNode(prim::kPrimTupleGetItem), offload_info->fw_node_,
                                                     NewValueNode(SizeToLong(get_item_idx))};
    auto get_item_node = offload_info->fw_graph_->NewCNode(get_item_inputs);
    get_item_node->set_scope(offload_info->fw_node_->scope());
    get_item_node->set_abstract(offload_info->bw_node_->abstract());
    offload_info->fw_node_ = get_item_node;
    MS_LOG(DEBUG) << "Backward node is TupleGetItem, move it to forward graph.";
  }

  auto move_in_node = move_to_node_cache_[offload_info->fw_node_][offload_info->bw_graph_];
  if (move_in_node == nullptr) {
    // CreateMoveTo CPU in forward graph.
    const auto &to_cpu_value = MakeValue(kToCpu);
    auto to_cpu_value_node = NewValueNode(to_cpu_value);
    to_cpu_value_node->set_abstract(to_cpu_value->ToAbstract());
    const auto &blocking_value = MakeValue(false);
    auto blocking_value_node = NewValueNode(blocking_value);
    blocking_value_node->set_abstract(blocking_value->ToAbstract());
    const auto move_out_primitive = std::make_shared<Primitive>(prim::kPrimMoveTo->name());
    move_out_primitive->AddAttr(kAttrBackwardPrefetch, MakeValue(offload_info->bw_prefetch_));
    const std::vector<AnfNodePtr> move_out_inputs = {NewValueNode(move_out_primitive), offload_info->fw_node_,
                                                     to_cpu_value_node, blocking_value_node};
    auto move_out_node = offload_info->fw_graph_->NewCNode(move_out_inputs);
    MS_EXCEPTION_IF_NULL(move_out_node);
    move_out_node->set_scope(offload_info->fw_node_->scope());
    move_out_node->set_abstract(offload_info->fw_node_->abstract());
    AddDependForMoveOut(offload_info->fw_graph_, offload_info->fw_node_, move_out_node);

    // Create MoveTo NPU in backward graph.
    const auto &to_npu_value = MakeValue(kToNpu);
    auto to_npu_value_node = std::make_shared<ValueNode>(to_npu_value);
    to_npu_value_node->set_abstract(to_npu_value->ToAbstract());
    const auto move_in_primitive = std::make_shared<Primitive>(prim::kPrimMoveTo->name());
    // Add backward_prefetch attr for MoveTo(In) node.
    move_in_primitive->AddAttr(kAttrBackwardPrefetch, MakeValue(offload_info->bw_prefetch_));
    const std::vector<AnfNodePtr> move_in_inputs = {NewValueNode(move_in_primitive), move_out_node, to_npu_value_node,
                                                    blocking_value_node};
    move_in_node = offload_info->bw_graph_->NewCNode(move_in_inputs);
    MS_EXCEPTION_IF_NULL(move_in_node);
    move_in_node->set_scope(offload_info->bw_node_->scope());
    move_in_node->set_abstract(offload_info->fw_node_->abstract());
    MS_LOG(DEBUG) << "Offload forward output " << offload_info->fw_node_->DebugString() << " with "
                  << move_out_node->DebugString() << " and copy it back with " << move_in_node->DebugString() << " for "
                  << offload_info->bw_node_->DebugString() << ":" << offload_info->bw_input_idx_;
  }

  // Set MoveTo(Npu) as the input of backward node.
  if (is_tuple_get_item) {
    manager_->Replace(offload_info->bw_node_, move_in_node);
  } else {
    manager_->SetEdge(offload_info->bw_node_, offload_info->bw_input_idx_, move_in_node);
    move_to_node_cache_[offload_info->fw_node_][offload_info->bw_graph_] = move_in_node;
  }
}

void OffloadActivationOptimizer::WarningOffloadOutsideLazyInline() {
  for (const auto &child_graph : manager_->func_graphs()) {
    if (child_graph->has_flag(FUNC_GRAPH_FLAG_CELL_REUSE)) {
      continue;
    }
    const auto &all_cnodes = child_graph->GetOrderedCnodes();
    for (const auto &cnode : all_cnodes) {
      const auto &primitive = GetCNodePrimitive(cnode);
      if (!GetPrimitiveFlag(primitive, kAttrOffload)) {
        continue;
      }
      if (offload_in_lazy_inline_.count(cnode) == 0) {
        MS_LOG(WARNING) << "Offload does not take effect outside the scope of the lazy_inline Cell. NOde: "
                        << cnode->DebugString() << ", trace info: " << trace::GetDebugInfoStr(cnode->debug_info());
      }
    }
  }
}

bool OffloadActivationOptimizer::Optimize(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  bool changed = false;
  manager_ = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager_);
  GetFwBwGraphs();
  WarningOffloadOutsideLazyInline();
  // Add offload flag for communication user node of offloaded nodes.
  for (const auto &fw_bw : fw_bw_graphs_) {
    AddOffloadForCommUser(fw_bw.first);
  }
  for (const auto &fw_bw : fw_bw_graphs_) {
    GetActivationOffloadInfo(fw_bw.first, fw_bw.second);
    fw_bw.second->set_flag(kAttrOffloadActivationHandled, true);
  }
  changed = !offload_infos_.empty();
  // Insert MoveTo node
  for (const auto &offload_info : offload_infos_) {
    InsertMoveToForOffloadActivation(offload_info);
  }
  return changed;
}
}  // namespace parallel
}  // namespace mindspore
