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

#include "plugin/ascend/kernel_executor/hyper_offload/hyper_offload.h"
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
#include "plugin/ascend/kernel_executor/hyper_offload/hyper_offload_utils.h"
#include "plugin/ascend/kernel_executor/hyper_offload/hyper_offload_strategy.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace hyper_offload {
namespace {
using ConditionSwitchInfoPtr = std::shared_ptr<ConditionSwitchInfo>;
constexpr auto kSwitchTrueBranchNum = "TrueBranchNodeNum";
constexpr auto kSwitchFalseBranchNum = "FalseBranchNodeNum";

void InsertHyperOffloadNodes(const OffloadInfo &offload_info, const std::vector<KernelWithIndex> &outputs,
                             const KernelGraphPtr &kernel_graph, CNodePtrList &new_execution_order) {
  const auto &data_node_info = offload_info.data_node;
  auto data_node = data_node_info.first;
  auto data_node_abstract = data_node->abstract();
  MS_EXCEPTION_IF_NULL(data_node_abstract);
  if (!data_node_abstract->isa<abstract::AbstractTensor>()) {
    if (!data_node_abstract->isa<abstract::AbstractTuple>()) {
      MS_LOG(EXCEPTION) << "Found non-except abstract for node " << data_node->fullname_with_scope()
                        << " with abstract: " << data_node_abstract->ToString();
    }
    auto index_node = kernel_graph->NewValueNode(MakeValue(int64_t(data_node_info.second)));
    index_node->set_abstract(std::make_shared<abstract::AbstractScalar>(int64_t(data_node_info.second)));
    data_node = kernel_graph->NewCNode(
      {NewValueNode(std::make_shared<Primitive>(prim::kPrimTupleGetItem->name())), data_node, index_node});
    data_node->set_abstract(data_node_abstract->cast<abstract::AbstractTuplePtr>()->elements()[data_node_info.second]);
  }
  auto to_remote_node = BuildToHostNode(kernel_graph, data_node);
  for (const auto &cur_replace_info : offload_info.replace_info_list) {
    auto to_device_node = BuildToDeviceNode(kernel_graph, to_remote_node);
    auto upper_bound_node = cur_replace_info.replace_start_node.first;
    const auto &change_nodes_infos = cur_replace_info.replace_rest_nodes;
    MS_EXCEPTION_IF_CHECK_FAIL(!change_nodes_infos.empty(), "change_nodes should not be empty");
    for (const auto &change_node_info : change_nodes_infos) {
      auto change_node = change_node_info.first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(change_node);
      int index = change_node_info.second;
      MS_EXCEPTION_IF_CHECK_FAIL(index >= 0, "fail to found index.");
      UpdateNodeInput(kernel_graph, change_node, index, to_device_node);
    }
    auto position = std::find(new_execution_order.begin(), new_execution_order.end(), change_nodes_infos[0].first);
    new_execution_order.insert(position, to_device_node);
  }
  bool is_output =
    std::any_of(outputs.begin(), outputs.end(), [&data_node](const auto &e) { return e.first == data_node; });
  if (is_output) {
    auto output = kernel_graph->output()->cast<CNodePtr>();
    MS_EXCEPTION_IF_CHECK_FAIL(IsPrimitiveCNode(output, prim::kPrimMakeTuple), "output not make tuple");
    int index = -1;
    for (size_t i = 1; i < output->inputs().size(); ++i) {
      if (output->input(i) == data_node) {
        index = SizeToInt(i);
        break;
      }
    }
    MS_EXCEPTION_IF_CHECK_FAIL(index > 0, "Fail to find index");
    auto to_device_node = BuildToDeviceNode(kernel_graph, to_remote_node);
    UpdateNodeInput(kernel_graph, output, IntToSize(index), to_device_node);
    new_execution_order.emplace_back(to_device_node);
  }
  auto data_iter = std::find(new_execution_order.begin(), new_execution_order.end(), data_node_info.first);
  MS_EXCEPTION_IF_CHECK_FAIL(data_iter != new_execution_order.end(), "Fail to find node in execution order.");
  new_execution_order.insert(data_iter + 1, to_remote_node);
}

std::optional<AnfNodePtr> FindLastRealUser(const AnfNodePtr &node, const UserInfoList &user_info_list) {
  int64_t output_index = 0;
  if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
    auto cnode = node->cast<CNodePtr>();
    constexpr size_t index_pos = 2;
    auto index_value_node = cnode->input(index_pos)->cast<ValueNodePtr>();
    output_index = GetValue<int64_t>(index_value_node->value());
  }
  auto node_iter = std::find_if(user_info_list.begin(), user_info_list.end(), [&node, output_index](const auto &e) {
    if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
      auto real_data = node->cast<CNodePtr>()->input(1);
      return e.node.first == real_data && (int64_t)e.node.second == output_index;
    }
    return e.node.first == node && (int64_t)e.node.second == output_index;
  });
  if (node_iter == user_info_list.end()) {
    return std::nullopt;
  }
  const auto &node_users = node_iter->users;
  auto last_user = node_users[node_users.size() - 1].first;
  // ToRemote will be placed right after the data is changed.
  if (IsD2HNode(last_user)) {
    return std::nullopt;
  }
  return last_user;
}

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

void AdjustFrontendHyperOffloadOps(const KernelGraphPtr &kernel_graph) {
  CNodePtrList new_execution_order{};
  CNodePtrList remain_h2d_nodes{};
  for (const auto node : kernel_graph->execution_order()) {
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
  kernel_graph->set_execution_order(new_execution_order);
}

void UpdateInlineInfo(const KernelGraphPtr &kernel_graph, const CNodePtr &node,
                      const mindspore::HashMap<AnfNodePtr, std::string>::const_iterator &iter) {
  const auto &inline_sub_graph_kernels = kernel_graph->inline_sub_graph_kernels();
  if (iter == inline_sub_graph_kernels.end()) {
    return;
  }
  kernel_graph->AddInlineSubgraphKernel(node, iter->second);
}

void InsertHyperOffloadOpsForInputParameters(const KernelGraphPtr &kernel_graph, const CNodePtr &node,
                                             CNodePtrList &execution_order) {
  if (IsPrimitiveCNode(node, prim::kPrimConditionSwitch)) {
    (void)execution_order.emplace_back(node);
    return;
  }
  CNodePtrList to_remote_nodes{};
  const auto &inline_sub_graph_kernels = kernel_graph->inline_sub_graph_kernels();
  const auto &inline_info_iter = inline_sub_graph_kernels.find(node);
  for (size_t i = 0; i < node->inputs().size(); ++i) {
    const auto &input = node->input(i);
    auto real_input = common::AnfAlgo::VisitKernelWithReturnType(input, 0, false, {prim::kPrimLoad}).first;
    if (IsPrimitiveCNode(real_input, prim::kPrimLoad)) {
      constexpr size_t data_index = 1;
      auto data_node = real_input->cast<CNodePtr>()->input(data_index);
      if (!GetRemoteParameter(kernel_graph, data_node).has_value()) {
        continue;
      }
      auto to_device_node = BuildToDeviceNode(kernel_graph, data_node);
      to_device_node->set_abstract(real_input->abstract());
      UpdateNodeInput(kernel_graph, node, i, to_device_node);
      UpdateInlineInfo(kernel_graph, to_device_node, inline_info_iter);
      (void)execution_order.emplace_back(to_device_node);
      continue;
    }
    const auto &param_res = GetRemoteParameter(kernel_graph, input);
    if (!param_res.has_value()) {
      continue;
    }
    auto to_device_node = BuildToDeviceNode(kernel_graph, input);
    to_device_node->set_abstract(dyn_cast<abstract::AbstractRefTensor>(input->abstract())->CloneAsTensor());
    UpdateNodeInput(kernel_graph, node, i, to_device_node);
    UpdateInlineInfo(kernel_graph, to_device_node, inline_info_iter);
    (void)execution_order.emplace_back(to_device_node);
    auto to_remote_node = BuildInplaceToHostNode(kernel_graph, to_device_node, param_res.value(), node);
    UpdateInlineInfo(kernel_graph, to_remote_node, inline_info_iter);
    (void)to_remote_nodes.emplace_back(to_remote_node);
  }
  (void)execution_order.emplace_back(node);
  for (const auto &to_remote_node : to_remote_nodes) {
    (void)execution_order.emplace_back(to_remote_node);
  }
}

void PrefetchH2DNode(const CNodePtr &node, CNodePtrList &execution_order) {
  if (execution_order.empty()) {
    (void)execution_order.emplace_back(node);
    return;
  }
  static size_t prefetch_distance = HyperOffloadConfig::GetInstance().prefetch_distance();
  constexpr size_t data_index = 1;
  auto data_node = node->input(data_index);
  size_t upper_bound_index = 0;
  if (data_node->isa<CNode>()) {
    // todo: This is caused by condition switch. Need to fix later.
    if (!IsD2HNode(data_node)) {
      (void)execution_order.emplace_back(node);
      return;
    }
    auto upper_bound_iter = std::find(execution_order.begin(), execution_order.end(), data_node);
    MS_EXCEPTION_IF_CHECK_FAIL(upper_bound_iter != execution_order.end(), "Fail to find data node");
    upper_bound_index = std::distance(execution_order.begin(), upper_bound_iter) + 1;
  }
  MS_LOG(INFO) << "For node: " << node->fullname_with_scope() << ", upper bound index: " << upper_bound_index
               << ", lower bound index: " << (execution_order.size() - 1);
  if (size_t(execution_order.size()) > prefetch_distance) {
    (void)execution_order.insert(execution_order.begin() + execution_order.size() - prefetch_distance, node);
  } else {
    MS_LOG(INFO) << "Execution order size: " << execution_order.size()
                 << " is smaller than prefetch distance: " << prefetch_distance;
    (void)execution_order.insert(execution_order.begin(), node);
  }
}

void GenerateEventForD2HNode(const KernelGraphPtr &kernel_graph, const CNodePtr &node,
                             const UserInfoList &all_node_users, CNodePtrList &execution_order,
                             std::unordered_map<AnfNodePtr, AnfNodePtrList> &event_map) {
  static size_t distance = HyperOffloadConfig::GetInstance().release_distance();
  const auto &nodes = kernel_graph->execution_order();
  auto process_stream_id = AnfAlgo::GetStreamId(node);
  constexpr size_t wait_stream_id = 0;
  constexpr size_t data_index = 1;
  auto data_node = node->input(data_index);
  const auto &send_recv_pair =
    AclStreamAssign::GetInstance().CreateSendReceive(NOT_NULL(kernel_graph), process_stream_id, wait_stream_id);
  (void)execution_order.emplace_back(send_recv_pair.first);
  auto data_node_last_user = FindLastRealUser(data_node, all_node_users);
  AnfNodePtr insert_node = nullptr;
  if (data_node_last_user.has_value()) {
    auto last_user_node = data_node_last_user.value();
    auto last_user_node_iter = std::find(nodes.begin(), nodes.end(), last_user_node);
    MS_EXCEPTION_IF_CHECK_FAIL(last_user_node_iter != nodes.end(), "Fail to find last user node.");
    insert_node = *(last_user_node_iter + distance);
  } else {
    auto node_iter = std::find(nodes.begin(), nodes.end(), node);
    MS_EXCEPTION_IF_CHECK_FAIL(node_iter != nodes.end(), "Fail to find node");
    auto insert_node_iter = std::next(node_iter, distance);
    MS_EXCEPTION_IF_CHECK_FAIL(insert_node_iter != nodes.end(), "Failed to find insert node position");
    insert_node = *(insert_node_iter);
  }
  MS_EXCEPTION_IF_NULL(insert_node);
  auto copy_to_remote_event_map_iter = event_map.find(insert_node);
  if (copy_to_remote_event_map_iter == event_map.end()) {
    event_map[insert_node] = {send_recv_pair.second};
  } else {
    (void)event_map[insert_node].emplace_back(send_recv_pair.second);
  }
}
}  // namespace

bool HyperOffloadOptimizer::EnableHyperOffloadParameterForGraph(const KernelGraphPtr &kernel_graph) {
  constexpr auto offload_parameter_attr = "offload_parameter";
  return kernel_graph->has_user_data(offload_parameter_attr);
}

bool HyperOffloadOptimizer::EnableHyperOffloadActivationForGraph(const KernelGraphPtr &kernel_graph) {
  constexpr auto offload_activation_attr = "offload_activation";
  return kernel_graph->has_user_data(offload_activation_attr);
}

bool HyperOffloadOptimizer::EnableHyperOffloadForGraph(const KernelGraphPtr &kernel_graph) {
  if (!HyperOffloadConfig::GetInstance().enable_hyper_offload()) {
    return false;
  }
  return EnableHyperOffloadParameterForGraph(kernel_graph) || EnableHyperOffloadActivationForGraph(kernel_graph);
}

void HyperOffloadOptimizer::Run(const KernelGraphPtr &kernel_graph) {
  if (!EnableHyperOffloadForGraph(kernel_graph)) {
    return;
  }
  AdjustFrontendHyperOffloadOps(kernel_graph);
  if (EnableHyperOffloadParameterForGraph(kernel_graph)) {
    OffloadParameters(kernel_graph);
  }
  if (EnableHyperOffloadActivationForGraph(kernel_graph)) {
    OffloadActivation(kernel_graph);
  }
  AdjustHyperOffloadNodePosition(kernel_graph);
}

void HyperOffloadOptimizer::OffloadActivation(const KernelGraphPtr &kernel_graph) {
  auto hyper_offload_strategy = std::make_unique<DistanceBaseHyperOffloadStrategy>();
  const auto &offload_info_list = hyper_offload_strategy->Run(kernel_graph);
  if (offload_info_list.empty()) {
    return;
  }
  auto offload_info_filter = std::make_unique<OffloadInfoFilterByNumber>();
  const auto &filtered_offload_info_list = offload_info_filter->Filter(offload_info_list);

  const auto &outputs = common::AnfAlgo::GetAllOutputWithIndex(kernel_graph->output());
  CNodePtrList new_execution_order = kernel_graph->execution_order();
  for (const auto &offload_info : filtered_offload_info_list) {
    InsertHyperOffloadNodes(offload_info, outputs, kernel_graph, new_execution_order);
  }
  kernel_graph->set_execution_order(new_execution_order);
}

void HyperOffloadOptimizer::OffloadParameters(const KernelGraphPtr &kernel_graph) {
  const auto &nodes = kernel_graph->execution_order();
  CNodePtrList new_execution_order{};
  for (const auto &node : nodes) {
    InsertHyperOffloadOpsForInputParameters(kernel_graph, node, new_execution_order);
  }
  kernel_graph->set_execution_order(new_execution_order);
}

void HyperOffloadOptimizer::AdjustHyperOffloadNodePosition(const KernelGraphPtr &kernel_graph) {
  const auto &nodes = kernel_graph->execution_order();
  CNodePtrList new_execution_order{};
  for (const auto &node : nodes) {
    if (IsH2DNode(node)) {
      PrefetchH2DNode(node, new_execution_order);
      continue;
    }
    (void)new_execution_order.emplace_back(node);
  }
  kernel_graph->set_execution_order(new_execution_order);
}

void HyperOffloadOptimizer::AddEventToHyperOffloadOps(const KernelGraphPtr &kernel_graph) {
  if (!EnableHyperOffloadForGraph(kernel_graph)) {
    return;
  }
  CNodePtrList new_execution_order{};
  const auto &nodes = kernel_graph->execution_order();
  const auto &all_node_users = CollectAllNodeUsers(nodes);
  std::unordered_map<AnfNodePtr, AnfNodePtrList> event_map;
  for (size_t i = 0; i < nodes.size(); ++i) {
    const auto &node = nodes[i];
    (void)new_execution_order.emplace_back(node);
    if (IsD2HNode(node)) {
      GenerateEventForD2HNode(kernel_graph, node, all_node_users, new_execution_order, event_map);
    }
    auto iter = event_map.find(node);
    if (iter != event_map.end()) {
      const auto &all_recv_nodes = iter->second;
      for (const auto recv_node : all_recv_nodes) {
        (void)new_execution_order.emplace_back(recv_node->cast<CNodePtr>());
      }
      event_map.erase(iter);
    }
  }
  MS_EXCEPTION_IF_CHECK_FAIL(event_map.empty(), "Remain recv node to insert.");
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
