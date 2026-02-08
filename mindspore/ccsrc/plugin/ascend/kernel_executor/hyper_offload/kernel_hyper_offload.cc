/**
 * Copyright 2025-2026 Huawei Technologies Co., Ltd
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

#include "plugin/ascend/kernel_executor/hyper_offload/kernel_hyper_offload.h"

#include <vector>
#include <memory>
#include "utils/compile_config.h"
#include "include/utils/anfalgo.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "primitive/framework_ops.h"
#include "primitive/other_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "plugin/ascend/kernel_executor/kernel_select_ascend.h"
#include "plugin/ascend/kernel_executor/rts/rt_kernel_build.h"
#include "plugin/ascend/graph_optimizer/stream_assign/acl_stream_assign.h"
#include "mindspore/ops/ops_utils/op_constants.h"
#include "utils/hyper_offload/hyper_offload.h"
#include "utils/hyper_offload/hyper_offload_utils.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace hyper_offload {
namespace {
using ConditionSwitchInfoPtr = std::shared_ptr<ConditionSwitchInfo>;
constexpr auto kSwitchTrueBranchNum = "TrueBranchNodeNum";
constexpr auto kSwitchFalseBranchNum = "FalseBranchNodeNum";

// Set branch information attributes for the Switch node
void SetBranchInfoForSwitchNode(const ConditionSwitchInfoPtr &switch_info) {
  MS_EXCEPTION_IF_NULL(switch_info);
  auto switch_cnode = switch_info->switch_cnode;
  MS_EXCEPTION_IF_NULL(switch_cnode);
  switch_cnode->AddAttr(kSwitchTrueBranchNum, MakeValue(switch_info->true_branch_node_num));
  switch_cnode->AddAttr(kSwitchFalseBranchNum, MakeValue(switch_info->false_branch_node_num));
}

// Handle the ConditionGather node, reorder the control flow nodes for the condition gather node
CNodePtrList ReorderControlFlowNodesForConditionGather(std::vector<ConditionSwitchInfoPtr> *switch_index,
                                                       CNodePtrList *true_branch_cnodes,
                                                       CNodePtrList *false_branch_cnodes, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(switch_index);
  MS_EXCEPTION_IF_NULL(true_branch_cnodes);
  MS_EXCEPTION_IF_NULL(false_branch_cnodes);
  MS_EXCEPTION_IF_NULL(cnode);
  auto lastest_switch_info = switch_index->back();
  CNodePtrList switch_kernel_cnodes;
  MS_EXCEPTION_IF_NULL(lastest_switch_info);
  auto true_branch_size = lastest_switch_info->true_branch_node_num;
  // Mark true branch range
  if (true_branch_size > true_branch_cnodes->size()) {
    MS_LOG(EXCEPTION) << "Invalid true branch node num: " << true_branch_size << " , "
                      << lastest_switch_info->true_branch_node_num;
  }
  switch_kernel_cnodes.insert(switch_kernel_cnodes.end(), true_branch_cnodes->end() - true_branch_size,
                              true_branch_cnodes->end());
  true_branch_cnodes->resize(true_branch_cnodes->size() - true_branch_size);
  // Mark false branch range
  auto false_branch_size = lastest_switch_info->false_branch_node_num;
  if (false_branch_size > false_branch_cnodes->size()) {
    MS_LOG(EXCEPTION) << "Invalid false branch node num: " << false_branch_size << " , "
                      << lastest_switch_info->false_branch_node_num;
  }
  switch_kernel_cnodes.insert(switch_kernel_cnodes.end(), false_branch_cnodes->end() - false_branch_size,
                              false_branch_cnodes->end());
  false_branch_cnodes->resize(false_branch_cnodes->size() - false_branch_size);
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "Switch scope, true branch size: " << true_branch_size
                                 << ", false branch size: " << false_branch_size;
  switch_kernel_cnodes.emplace_back(cnode);
  SetBranchInfoForSwitchNode(lastest_switch_info);
  switch_index->pop_back();

  return switch_kernel_cnodes;
}

void UpdateOuterSwitchInfo(const std::vector<ConditionSwitchInfoPtr> &switch_index, CNodePtrList *true_branch_cnodes,
                           CNodePtrList *false_branch_cnodes, const CNodePtrList &switch_kernel_cnodes,
                           const std::string &subgraph_name) {
  if (switch_index.empty()) {
    return;
  }
  // Nested control flow, update outer switch info
  auto &outer_switch_info = switch_index.back();
  bool is_true_branch = subgraph_name == outer_switch_info->true_subgraph;
  if (is_true_branch) {
    MS_EXCEPTION_IF_NULL(true_branch_cnodes);
    true_branch_cnodes->insert(true_branch_cnodes->end(), switch_kernel_cnodes.begin(), switch_kernel_cnodes.end());
    outer_switch_info->true_branch_node_num += switch_kernel_cnodes.size();
  } else {
    MS_EXCEPTION_IF_NULL(false_branch_cnodes);
    false_branch_cnodes->insert(false_branch_cnodes->end(), switch_kernel_cnodes.begin(), switch_kernel_cnodes.end());
    outer_switch_info->false_branch_node_num += switch_kernel_cnodes.size();
  }
}

// Create a ConditionSwitchInfo object for the ConditionSwitch cnode
ConditionSwitchInfoPtr CreateConditionSwitchInfo(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &inline_sub_graph_names = cnode->GetAttr(kInlineSubGraphName);
  if (inline_sub_graph_names == nullptr || !inline_sub_graph_names->isa<ValueTuple>()) {
    MS_LOG(EXCEPTION) << "Invalid condition switch node: " << cnode->DebugString();
  }
  const auto &tuple_name = inline_sub_graph_names->cast<ValueTuplePtr>()->value();
  constexpr size_t tuple_name_size = 2;
  if (tuple_name.size() != tuple_name_size) {
    MS_LOG(EXCEPTION) << "Invalid condition switch node: " << cnode->DebugString()
                      << ", tuple name size: " << tuple_name.size();
  }
  MS_LOG(INFO) << "Condition switch cnode: " << cnode->DebugString()
               << ", sub graph name: " << inline_sub_graph_names->ToString();
  constexpr size_t true_branch_index = 1;
  constexpr size_t false_branch_index = 0;
  auto true_branch_name = GetValue<std::string>(tuple_name[true_branch_index]);
  auto false_branch_name = GetValue<std::string>(tuple_name[false_branch_index]);
  auto cur_switch_info = std::make_shared<ConditionSwitchInfo>(cnode, true_branch_name, false_branch_name);
  return cur_switch_info;
}

void UpdateInlineInfo(const KernelGraphPtr &kernel_graph, const CNodePtr &node,
                      const mindspore::HashMap<AnfNodePtr, std::string>::const_iterator &iter) {
  const auto &inline_sub_graph_kernels = kernel_graph->inline_sub_graph_kernels();
  if (iter == inline_sub_graph_kernels.end()) {
    return;
  }
  kernel_graph->AddInlineSubgraphKernel(node, iter->second);
}

void AdjustFrontendHyperOffloadOpsExecutionOrder(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  CNodePtrList new_execution_order{};
  CNodePtrList remain_h2d_nodes{};
  for (const auto node : graph->execution_order()) {
    if (IsH2DNode(node)) {
      (void)remain_h2d_nodes.emplace_back(node);
      continue;
    }
    // ToRemote node should execute right after data is created.
    if (IsD2HNode(node)) {
      constexpr size_t data_index = 1;
      auto data_node = node->input(data_index);
      if (data_node->isa<CNode>()) {
        auto data_node_iter = std::find(new_execution_order.begin(), new_execution_order.end(), data_node);
        MS_EXCEPTION_IF_CHECK_FAIL(data_node_iter != new_execution_order.end(), "Find data_node failed.");
        new_execution_order.insert(data_node_iter + 1, node);
      } else {
        (void)new_execution_order.emplace_back(node);
      }
      continue;
    }
    const auto &inputs = node->inputs();
    if (IsViewOrInplaceNode(node)) {
      auto new_end = std::remove_if(remain_h2d_nodes.begin(), remain_h2d_nodes.end(), [&](const CNodePtr &h2d_node) {
        constexpr size_t h2d_data_idx = 1;
        const auto &h2d_data = h2d_node->input(h2d_data_idx);
        bool found =
          std::any_of(inputs.begin(), inputs.end(), [&h2d_data](const auto &input) { return input == h2d_data; });
        if (found) {
          (void)new_execution_order.emplace_back(h2d_node);
        }
        return found;
      });
      remain_h2d_nodes.erase(new_end, remain_h2d_nodes.end());
      new_execution_order.emplace_back(node);
      continue;
    }

    for (const auto &input : inputs) {
      if (!IsH2DNode(input)) {
        continue;
      }
      auto remain_iter = std::find(remain_h2d_nodes.begin(), remain_h2d_nodes.end(), input);
      if (remain_iter != remain_h2d_nodes.end()) {
        remain_h2d_nodes.erase(remain_iter);
        (void)new_execution_order.emplace_back(input->cast<CNodePtr>());
      }
    }
    (void)new_execution_order.emplace_back(node);
  }
  for (const auto &node : remain_h2d_nodes) {
    (void)new_execution_order.emplace_back(node);
  }
  graph->set_execution_order(new_execution_order);
}
}  // namespace

bool KernelHyperOffloadOptimizer::EnableHyperOffloadParameterForGraph(const KernelGraphPtr &kernel_graph) {
  constexpr auto offload_parameter_attr = "offload_parameter";
  return kernel_graph->has_user_data(offload_parameter_attr);
}

bool KernelHyperOffloadOptimizer::EnableHyperOffloadActivationForGraph(const KernelGraphPtr &kernel_graph) {
  constexpr auto offload_activation_attr = "offload_activation";
  return kernel_graph->has_user_data(offload_activation_attr);
}

bool KernelHyperOffloadOptimizer::EnableHyperOffloadForGraph(const KernelGraphPtr &kernel_graph) {
  if (!HyperOffloadConfig::GetInstance().enable_hyper_offload()) {
    return false;
  }
  return EnableHyperOffloadParameterForGraph(kernel_graph) || EnableHyperOffloadActivationForGraph(kernel_graph);
}

HyperOffloadInput KernelHyperOffloadOptimizer::GenerateHyperOffloadInput(const KernelGraphPtr &kernel_graph) {
  auto callback = [&kernel_graph](const AnfNodePtr &new_node, const AnfNodePtr &context_node) {
    auto cnode = new_node->cast<CNodePtr>();
    if (cnode == nullptr) {
      return;
    }

    if (context_node != nullptr) {
      cnode->set_scope(context_node->scope());
      const auto &inline_sub_graph_kernels = kernel_graph->inline_sub_graph_kernels();
      auto iter = inline_sub_graph_kernels.find(context_node);
      if (iter != inline_sub_graph_kernels.end()) {
        UpdateInlineInfo(kernel_graph, cnode, iter);
      }
    }

    for (size_t i = 0; i < cnode->inputs().size(); ++i) {
      auto input = cnode->input(i);
      if (input->isa<ValueNode>()) {
        auto value = GetValueNode(input);
        auto new_value_node = kernel_graph->NewValueNode(value);
        new_value_node->set_abstract(input->abstract());
        cnode->set_input(i, new_value_node);
      }
    }

    device::ascend::GenerateKernelBuildInfo(cnode, RT_KERNEL);
    auto kernel_mod_ptr = kernel::RtOpBuild(cnode);
    MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
    AnfAlgo::SetKernelMod(kernel_mod_ptr, cnode.get());
  };

  HyperOffloadInput input;
  input.graph = kernel_graph;
  input.exec_order = kernel_graph->execution_order();
  input.enable_offload_parameter = EnableHyperOffloadParameterForGraph(kernel_graph);
  input.enable_offload_activation = EnableHyperOffloadActivationForGraph(kernel_graph);
  input.callback = callback;
  return input;
}

void KernelHyperOffloadOptimizer::UpdateRefMapForHyperOffloadInput(const KernelGraphPtr &kernel_graph) {
  const auto &execution_order = kernel_graph->execution_order();
  const auto &user_info_list = CollectAllNodeUsers(execution_order);
  for (const auto &user_info : user_info_list) {
    if (user_info.node.second != 0) {
      continue;
    }
    const auto &data_node = user_info.node.first;
    if (!IsD2HNode(data_node) && !IsH2DNode(data_node)) {
      continue;
    }
    const auto &user_nodes = user_info.users;
    for (const auto &user_node : user_nodes) {
      kernel_graph->ReplaceRefMapInput({user_node.first, 0}, {data_node, 0});
    }
  }
}

std::pair<CNodePtrList, std::vector<EventPlan>> KernelHyperOffloadOptimizer::GenerateExecutionOrderAndEventPlan(
  const CNodePtrList &exec_order) {
  std::map<uint32_t, EventPlan> event_map;
  CNodePtrList new_execution_order;
  CNodePtr cur_real_node = nullptr;

  // Collect all send/recv nodes with type kSendRecvAfterD2H and filter other types of EventNodes.
  for (const auto &node : exec_order) {
    if (!node->has_user_data(utils::hyper_offload::kHyperOffloadEventNodeType)) {
      (void)new_execution_order.emplace_back(node);
      cur_real_node = node;
      continue;
    }
    auto event_type =
      node->user_data<utils::hyper_offload::EventNodeType>(utils::hyper_offload::kHyperOffloadEventNodeType);
    MS_EXCEPTION_IF_NULL(event_type);
    if (*event_type != utils::hyper_offload::kSendRecvAfterD2H) {
      continue;
    }
    MS_EXCEPTION_IF_CHECK_FAIL(node->has_user_data(utils::hyper_offload::kHyperOffloadEventId),
                               "Event node has no event_id attribute.");
    auto event_id = *(node->user_data<uint32_t>(utils::hyper_offload::kHyperOffloadEventId));
    MS_EXCEPTION_IF_NULL(cur_real_node);
    if (IsPrimitiveCNode(node, prim::kPrimSend)) {
      event_map[event_id].send_insert_after = cur_real_node;
    } else if (IsPrimitiveCNode(node, prim::kPrimReceive)) {
      event_map[event_id].recv_insert_after = cur_real_node;
    }
  }

  // Generate event_plan according to event_map.
  std::vector<EventPlan> events;
  for (const auto &pair : event_map) {
    MS_EXCEPTION_IF_NULL(pair.second.send_insert_after);
    MS_EXCEPTION_IF_NULL(pair.second.recv_insert_after);
    (void)events.emplace_back(pair.second);
  }
  return std::pair<CNodePtrList, std::vector<EventPlan>>(new_execution_order, events);
}

void KernelHyperOffloadOptimizer::ApplyHyperOffloadOptimizer(const KernelGraphPtr &kernel_graph) {
  const auto &input = GenerateHyperOffloadInput(kernel_graph);
  utils::hyper_offload::HyperOffloadOptimizer optimizer(input);
  const auto &plan = optimizer.Run();
  const auto &[new_execution_order, events] = GenerateExecutionOrderAndEventPlan(plan.exec_order);
  kernel_graph->set_execution_order(new_execution_order);
  events_ = events;
}

void KernelHyperOffloadOptimizer::Run(const KernelGraphPtr &kernel_graph) {
  if (!EnableHyperOffloadForGraph(kernel_graph)) {
    return;
  }
  AdjustFrontendHyperOffloadOpsExecutionOrder(kernel_graph);
  ApplyHyperOffloadOptimizer(kernel_graph);
  UpdateRefMapForHyperOffloadInput(kernel_graph);
}

void KernelHyperOffloadOptimizer::AddEventToHyperOffloadOps(const KernelGraphPtr &kernel_graph) {
  if (!EnableHyperOffloadForGraph(kernel_graph) || events_.empty()) {
    return;
  }
  CNodePtrList new_execution_order{};
  const auto &nodes = kernel_graph->execution_order();
  const auto &inline_sub_graph_kernels = kernel_graph->inline_sub_graph_kernels();
  std::unordered_map<AnfNodePtr, AnfNodePtrList> event_map;

  for (const auto &event_hint : events_) {
    auto process_stream_id = AnfAlgo::GetStreamId(event_hint.send_insert_after);
    constexpr size_t wait_stream_id = 0;
    const auto &send_recv_pair =
      AclStreamAssign::GetInstance().CreateSendReceive(NOT_NULL(kernel_graph), process_stream_id, wait_stream_id);
    const auto &send_node = send_recv_pair.first;
    const auto &recv_node = send_recv_pair.second;

    auto iter = event_map.find(event_hint.send_insert_after);
    if (iter == event_map.end()) {
      event_map[event_hint.send_insert_after] = {send_node};
    } else {
      (void)event_map[event_hint.send_insert_after].emplace_back(send_node);
    }
    iter = event_map.find(event_hint.recv_insert_after);
    if (iter == event_map.end()) {
      event_map[event_hint.recv_insert_after] = {recv_node};
    } else {
      (void)event_map[event_hint.recv_insert_after].emplace_back(recv_node);
    }
  }

  for (const auto &node : nodes) {
    (void)new_execution_order.emplace_back(node);
    auto event_iter = event_map.find(node);
    if (event_iter == event_map.end()) {
      continue;
    }
    auto inline_iter = inline_sub_graph_kernels.find(node);
    for (const auto &insert_node : event_iter->second) {
      auto insert_cnode = insert_node->cast<CNodePtr>();
      new_execution_order.emplace_back(insert_cnode);
      if (inline_iter != inline_sub_graph_kernels.end()) {
        UpdateInlineInfo(kernel_graph, insert_cnode, inline_iter);
      }
    }
  }
  kernel_graph->set_execution_order(new_execution_order);
}

// Reorder the control flow nodes in the execution order, keep true branch and false branch nodes together
void ReorderControlFlowNodes(const KernelGraphPtr &kernel_graph) {
  static const bool use_hyper_offload_with_sliding_window =
    common::GetEnv("MS_DEV_HYPER_OFFLOAD") == "1" && common::GetCompileConfig("ENABLE_HYPER_OFFLOAD_SLIDE") == "1";
  if (!use_hyper_offload_with_sliding_window) {
    return;
  }
  MS_EXCEPTION_IF_NULL(kernel_graph);
  const auto &cnodes = kernel_graph->execution_order();
  std::vector<ConditionSwitchInfoPtr> switch_index;
  CNodePtrList true_branch_cnodes;
  CNodePtrList false_branch_cnodes;
  CNodePtrList new_execution_order{};
  const auto &cnode_branch_map = kernel_graph->inline_sub_graph_kernels();
  for (const auto &cnode : cnodes) {
    MS_EXCEPTION_IF_NULL(cnode);
    // Not in control flow node
    if (switch_index.empty()) {
      if (!IsPrimitiveCNode(cnode, prim::kPrimConditionSwitch)) {
        new_execution_order.emplace_back(cnode);
      } else {
        auto cur_switch_info = CreateConditionSwitchInfo(cnode);
        switch_index.push_back(cur_switch_info);
        new_execution_order.emplace_back(cnode);
      }
      continue;
    }

    const auto &branch_subgraph_name_iter = cnode_branch_map.find(cnode);
    std::string branch_subgraph_name = "";
    if (branch_subgraph_name_iter != cnode_branch_map.end()) {
      branch_subgraph_name = branch_subgraph_name_iter->second;
    } else if (!IsPrimitiveCNode(cnode, prim::kPrimConditionGather) || switch_index.size() != 1) {
      MS_LOG(EXCEPTION) << "Invalid node in control flow: " << cnode->DebugString();
    }

    // ConditionGather, process the whole control flow
    if (IsPrimitiveCNode(cnode, prim::kPrimConditionGather)) {
      auto switch_kernel_cnodes =
        ReorderControlFlowNodesForConditionGather(&switch_index, &true_branch_cnodes, &false_branch_cnodes, cnode);
      // Single control flow, just append to the execution order
      if (switch_index.empty()) {
        new_execution_order.insert(new_execution_order.end(), switch_kernel_cnodes.begin(), switch_kernel_cnodes.end());
      } else {
        UpdateOuterSwitchInfo(switch_index, &true_branch_cnodes, &false_branch_cnodes, switch_kernel_cnodes,
                              branch_subgraph_name);
      }
      continue;
    }

    auto &lastest_switch_info = switch_index.back();
    MS_EXCEPTION_IF_NULL(lastest_switch_info);
    if (lastest_switch_info->true_subgraph == branch_subgraph_name) {
      true_branch_cnodes.emplace_back(cnode);
      lastest_switch_info->true_branch_node_num++;
    } else {
      false_branch_cnodes.emplace_back(cnode);
      lastest_switch_info->false_branch_node_num++;
    }

    // Nested control flow scene
    if (IsPrimitiveCNode(cnode, prim::kPrimConditionSwitch)) {
      auto cur_switch_info = CreateConditionSwitchInfo(cnode);
      switch_index.push_back(cur_switch_info);
    }
  }
  kernel_graph->set_execution_order(new_execution_order);
}
}  // namespace hyper_offload
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
