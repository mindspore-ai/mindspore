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

#include "utils/hyper_offload/hyper_offload.h"

#include <atomic>
#include <unordered_map>
#include "include/utils/anfalgo.h"
#include "primitive/framework_ops.h"
#include "primitive/other_ops.h"
#include "utils/hyper_offload/hyper_offload_operation.h"

namespace mindspore {
namespace utils {
namespace hyper_offload {
namespace {
void AddSingleParameterOperations(const FuncGraphPtr &graph, const CNodePtr &node, const CNodePtrList &exec_order,
                                  const HandleNewNodeCallback &callback, HyperOffloadOperations &operations) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  if (IsPrimitiveCNode(node, prim::kPrimConditionSwitch)) {
    return;
  }
  auto node_iter = std::find(exec_order.begin(), exec_order.end(), node);
  if (node_iter == exec_order.end()) {
    return;
  }
  size_t node_index = std::distance(exec_order.begin(), node_iter);
  CNodePtrList to_remote_nodes{};
  for (size_t i = 0; i < node->inputs().size(); ++i) {
    const auto &input = node->input(i);
    auto real_input = common::AnfAlgo::VisitKernelWithReturnType(input, 0, false, {prim::kPrimLoad}).first;
    if (IsPrimitiveCNode(real_input, prim::kPrimLoad)) {
      constexpr size_t data_index = 1;
      auto data_node = real_input->cast<CNodePtr>()->input(data_index);
      if (!GetRemoteParameter(graph, data_node).has_value()) {
        continue;
      }
      auto to_device_node = BuildToDeviceNode(graph, data_node, callback);
      to_device_node->set_abstract(real_input->abstract());
      UpdateNodeInput(node, i, to_device_node);
      operations.AddOperationBefore(node_index, to_device_node);
      continue;
    }
    const auto &param_res = GetRemoteParameter(graph, input);
    if (!param_res.has_value()) {
      continue;
    }
    auto to_device_node = BuildToDeviceNode(graph, input, callback);
    to_device_node->set_abstract(dyn_cast<abstract::AbstractRefTensor>(input->abstract())->CloneAsTensor());
    UpdateNodeInput(node, i, to_device_node);
    operations.AddOperationBefore(node_index, to_device_node);
    auto to_remote_node = BuildInplaceToHostNode(graph, to_device_node, param_res.value(), node, callback);
    to_remote_nodes.push_back(to_remote_node);
  }

  for (const auto &to_remote_node : to_remote_nodes) {
    operations.AddOperationAfter(node_index, to_remote_node);
  }
  return;
}

void AddSingleActivationOperations(const OffloadInfo &offload_info, const std::vector<KernelWithIndex> &outputs,
                                   const FuncGraphPtr &graph, const CNodePtrList &exec_order,
                                   const HandleNewNodeCallback &callback, HyperOffloadOperations &operations) {
  MS_EXCEPTION_IF_NULL(graph);
  const auto &data_node_info = offload_info.data_node;
  auto data_node = data_node_info.first;
  auto data_node_abstract = data_node->abstract();
  MS_EXCEPTION_IF_NULL(data_node_abstract);

  if (!data_node_abstract->isa<abstract::AbstractTensor>()) {
    if (!data_node_abstract->isa<abstract::AbstractTuple>()) {
      MS_LOG(EXCEPTION) << "Found non-except abstract for node " << data_node->fullname_with_scope()
                        << " with abstract: " << data_node_abstract->ToString();
    }
    auto index_node = NewValueNode(MakeValue(int64_t(data_node_info.second)));
    index_node->set_abstract(std::make_shared<abstract::AbstractScalar>(int64_t(data_node_info.second)));
    data_node = graph->NewCNode(
      {NewValueNode(std::make_shared<Primitive>(prim::kPrimTupleGetItem->name())), data_node, index_node});
    data_node->set_abstract(data_node_abstract->cast<abstract::AbstractTuplePtr>()->elements()[data_node_info.second]);
    // Also notify callback about the tuplegetitem node
    if (callback != nullptr) {
      callback(data_node->cast<CNodePtr>(), data_node_info.first);
    }
  }

  auto to_remote_node = BuildToHostNode(graph, data_node, callback);

  auto data_iter = std::find(exec_order.begin(), exec_order.end(), data_node_info.first);
  MS_EXCEPTION_IF_CHECK_FAIL(data_iter != exec_order.end(), "Fail to find node in execution order.");
  size_t data_node_index = std::distance(exec_order.begin(), data_iter);
  operations.AddOperationAfter(data_node_index, to_remote_node);

  for (const auto &cur_replace_info : offload_info.replace_info_list) {
    auto to_device_node = BuildToDeviceNode(graph, to_remote_node, callback);
    const auto &change_nodes_infos = cur_replace_info.replace_rest_nodes;
    MS_EXCEPTION_IF_CHECK_FAIL(!change_nodes_infos.empty(), "change_nodes should not be empty");
    for (const auto &change_node_info : change_nodes_infos) {
      auto change_node = change_node_info.first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(change_node);
      int index = change_node_info.second;
      MS_EXCEPTION_IF_CHECK_FAIL(index >= 0, "fail to found index.");
      UpdateNodeInput(change_node, index, to_device_node);
    }
    auto change_node_iter = std::find(exec_order.begin(), exec_order.end(), change_nodes_infos[0].first);
    if (change_node_iter != exec_order.end()) {
      size_t change_node_index = std::distance(exec_order.begin(), change_node_iter);
      operations.AddOperationBefore(change_node_index, to_device_node);
    }
  }

  bool is_output =
    std::any_of(outputs.begin(), outputs.end(), [&data_node](const auto &e) { return e.first == data_node; });
  if (is_output) {
    auto output = graph->output()->cast<CNodePtr>();
    MS_EXCEPTION_IF_CHECK_FAIL(IsPrimitiveCNode(output, prim::kPrimMakeTuple), "output not make tuple");
    int index = -1;
    for (size_t i = 1; i < output->inputs().size(); ++i) {
      if (output->input(i) == data_node) {
        index = SizeToInt(i);
        break;
      }
    }
    MS_EXCEPTION_IF_CHECK_FAIL(index > 0, "Fail to find index");
    auto to_device_node = BuildToDeviceNode(graph, to_remote_node, callback);
    UpdateNodeInput(output, IntToSize(index), to_device_node);
    if (!exec_order.empty()) {
      size_t last_index = exec_order.size() - 1;
      operations.AddOperationAfter(last_index, to_device_node);
    }
  }
  return;
}

void AddEventNodesForSingleD2HNode(const FuncGraphPtr &graph, const CNodePtr &d2h_node, const CNodePtrList &exec_order,
                                   HyperOffloadOperations &ops, EventNodeFactory &event_factory, size_t node_index) {
  constexpr size_t data_index = 1;
  auto data_node = d2h_node->input(data_index);

  // Creation events: Send (Calc->Transfer) after A, Recv (Transfer) before D2H.
  auto [send_creation, recv_creation] =
    event_factory.CreateSendRecvPair(kCalculateStreamId, kTransferStreamId, kSendRecvBeforeD2H, data_node);
  // Completion events: Send (Transfer->Calc) after D2H, Recv (Calc) later.
  auto [send_completion, recv_completion] =
    event_factory.CreateSendRecvPair(kTransferStreamId, kCalculateStreamId, kSendRecvAfterD2H, data_node);

  // Insert Send_Creation, Recv_Creation, D2H, Send_Completion at node_index.
  // Note: AddOperationAfter appends to the list.
  // We want order: [Send_Creation, Recv_Creation, D2H, Send_Completion]
  // Since D2H is already in the list, we remove it and re-add everything in order.
  ops.RemoveNode(d2h_node);
  ops.AddOperationAfter(node_index, send_creation);
  ops.AddOperationAfter(node_index, recv_creation);
  ops.AddOperationAfter(node_index, d2h_node);
  ops.AddOperationAfter(node_index, send_completion);

  // Handle Recv_Completion insertion
  static size_t release_distance = HyperOffloadConfig::GetInstance().release_distance();
  size_t target_index = node_index + release_distance;
  if (target_index >= exec_order.size()) {
    target_index = exec_order.size() - 1;
  }
  // Insert Recv_Completion after the target node.
  ops.AddOperationAfter(target_index, recv_completion);

  auto free_node = BuildFreeDataNode(graph, data_node, kDeviceType);
  auto assign_node = BuildAssignDataNode(graph, data_node, d2h_node, kHostType);
  ops.AddOperationAfter(target_index, free_node->cast<CNodePtr>());
  ops.AddOperationAfter(target_index, assign_node->cast<CNodePtr>());
}

void AddEventNodesForD2HNodes(const FuncGraphPtr &graph, const CNodePtrList &exec_order, HyperOffloadOperations &ops,
                              EventNodeFactory &event_factory) {
  // Iterate over the original execution order
  // We make a copy of the operations list for each index because we modify the ops object
  // which might affect the underlying storage or iterators if we were iterating them directly.
  const auto &all_ops = ops.GetOperations();
  MS_EXCEPTION_IF_CHECK_FAIL(all_ops.size() == exec_order.size(),
                             "The size of operations is not equal to the size of original order.");
  for (size_t i = 0; i < exec_order.size(); ++i) {
    // Copy the list of operations at this index
    const auto &current_ops = all_ops[i].operations;
    for (const auto &node : current_ops) {
      if (IsD2HNode(node)) {
        AddEventNodesForSingleD2HNode(graph, node, exec_order, ops, event_factory, i);
      }
    }
  }
}

void AddEventNodesForNonLastH2DNode(const FuncGraphPtr &graph, const CNodePtr &h2d_node,
                                    const AnfNodePtr &original_data, size_t first_user_idx, size_t last_user_idx,
                                    const CNodePtr &recv_node, HyperOffloadOperations &ops) {
  ops.AddOperationBefore(first_user_idx, recv_node);
  // Before First User: AssignData(A, A2, Device)
  auto assign_dev = BuildAssignDataNode(graph, original_data, h2d_node, kDeviceType);
  ops.AddOperationBefore(first_user_idx, assign_dev->cast<CNodePtr>());

  // After Last User: FreeData(A, Device) -> AssignData(A, A1, Host)
  constexpr size_t data_index = 1;
  auto input_node = h2d_node->input(data_index);
  auto free_dev = BuildFreeDataNode(graph, original_data, kDeviceType);
  auto assign_host = BuildAssignDataNode(graph, original_data, input_node, kHostType);

  ops.AddOperationAfter(last_user_idx, free_dev->cast<CNodePtr>());
  ops.AddOperationAfter(last_user_idx, assign_host->cast<CNodePtr>());
}

void AddEventNodesForLastH2DNode(const FuncGraphPtr &graph, const CNodePtr &h2d_node, const AnfNodePtr &original_data,
                                 size_t first_user_idx, const CNodePtr &recv_node, HyperOffloadOperations &ops) {
  ops.AddOperationBefore(first_user_idx, recv_node);
  // Before First User: FreeData(A, Host) -> AssignData(A, A3, Device)
  auto free_host = BuildFreeDataNode(graph, original_data, kHostType);
  auto assign_dev = BuildAssignDataNode(graph, original_data, h2d_node, kDeviceType);

  ops.AddOperationBefore(first_user_idx, free_host->cast<CNodePtr>());
  ops.AddOperationBefore(first_user_idx, assign_dev->cast<CNodePtr>());
}

void AddEventNodesForH2DNodes(const FuncGraphPtr &graph, const CNodePtrList &exec_order, HyperOffloadOperations &ops,
                              EventNodeFactory &event_factory) {
  // 1. Collect all H2D nodes and group them by input (D2H node or Host node)
  // Map: InputNode -> List of (Index in exec_order, H2D Node)
  std::unordered_map<AnfNodePtr, std::vector<std::pair<size_t, CNodePtr>>> h2d_groups;

  constexpr size_t data_index = 1;
  const auto &all_ops = ops.GetOperations();
  for (size_t i = 0; i < all_ops.size(); ++i) {
    for (const auto &node : all_ops[i].operations) {
      if (IsH2DNode(node)) {
        auto input_node = node->input(data_index);
        (void)h2d_groups[input_node].emplace_back(i, node);
      }
    }
  }

  // 2. Process each group
  for (const auto &group_entry : h2d_groups) {
    const auto &h2d_list = group_entry.second;
    auto input_node = group_entry.first;
    AnfNodePtr original_data = nullptr;
    if (IsD2HNode(input_node)) {
      original_data = input_node->cast<CNodePtr>()->input(data_index);
    } else {
      // If input is not D2H, assume it is the data itself (e.g. host parameter)
      original_data = input_node;
    }

    size_t count = h2d_list.size();
    for (size_t i = 0; i < count; ++i) {
      auto [node_index, h2d_node] = h2d_list[i];
      bool is_last = (i == count - 1);

      // Create Send/Recv pair: Send (Transfer -> Calc), Recv (Calc)
      auto [send_node, recv_node] =
        event_factory.CreateSendRecvPair(kTransferStreamId, kCalculateStreamId, kSendRecvAfterH2D, original_data);

      // A. Insert Send node after H2D node
      ops.RemoveNode(h2d_node);
      ops.AddOperationAfter(node_index, h2d_node);
      ops.AddOperationAfter(node_index, send_node);

      // Find first and last user in exec_order
      size_t first_user_idx = exec_order.size();
      size_t last_user_idx = 0;
      bool found_user = false;

      for (size_t k = 0; k < exec_order.size(); ++k) {
        auto node = exec_order[k];
        for (const auto &input : node->inputs()) {
          if (input == h2d_node) {
            if (!found_user) {
              first_user_idx = k;
              found_user = true;
            }
            last_user_idx = k;
          }
        }
      }

      if (!found_user) {
        MS_LOG(EXCEPTION) << "H2D node " << h2d_node->fullname_with_scope() << " has no users in execution order.";
      }

      if (!is_last) {
        AddEventNodesForNonLastH2DNode(graph, h2d_node, original_data, first_user_idx, last_user_idx, recv_node, ops);
      } else {
        AddEventNodesForLastH2DNode(graph, h2d_node, original_data, first_user_idx, recv_node, ops);
      }
    }
  }
}

void AddEventNodes(const FuncGraphPtr &graph, const CNodePtrList &exec_order, HyperOffloadOperations &ops) {
  MS_EXCEPTION_IF_NULL(graph);
  EventNodeFactory event_factory(graph);
  AddEventNodesForD2HNodes(graph, exec_order, ops, event_factory);
  AddEventNodesForH2DNodes(graph, exec_order, ops, event_factory);
}
}  // namespace

bool HyperOffloadOptimizer::SkipHyperOffloadOptimizer(const HyperOffloadInput &input) {
  if (!HyperOffloadConfig::GetInstance().enable_hyper_offload()) {
    return true;
  }
  if (!input.enable_offload_parameter && !input.enable_offload_activation) {
    return true;
  }
  if (input.graph == nullptr || input.exec_order.empty()) {
    return true;
  }
  return false;
}

HyperOffloadPlan HyperOffloadOptimizer::Run(const HyperOffloadInput &input) {
  HyperOffloadPlan plan;
  plan.graph = input.graph;
  plan.exec_order = input.exec_order;
  if (SkipHyperOffloadOptimizer(input)) {
    return plan;
  }

  HyperOffloadOperations hyper_offload_operations;
  hyper_offload_operations.Init(plan.exec_order.size());
  if (input.enable_offload_activation) {
    auto act_ops = GenerateActivationOperations(plan.graph, plan.exec_order, input.callback);
    hyper_offload_operations.MergeFrom(act_ops);
  }
  if (input.enable_offload_parameter) {
    auto param_ops = GenerateParameterOperations(plan.graph, plan.exec_order, input.callback);
    hyper_offload_operations.MergeFrom(param_ops);
  }
  AdjustHyperOffloadNodePosition(hyper_offload_operations);
  AddEventNodes(plan.graph, plan.exec_order, hyper_offload_operations);

  plan.exec_order = BuildExecutionOrder(plan.exec_order, hyper_offload_operations);
  plan.hyper_offload_operations = hyper_offload_operations;
  AssignHyperOffloadIds(plan.exec_order);
  MS_LOG(INFO) << plan.hyper_offload_operations.ToString(input.exec_order);
  return plan;
}

HyperOffloadOperations HyperOffloadOptimizer::GenerateParameterOperations(const FuncGraphPtr &graph,
                                                                          const CNodePtrList &exec_order,
                                                                          const HandleNewNodeCallback &callback) {
  MS_EXCEPTION_IF_NULL(graph);
  HyperOffloadOperations params_operation;
  params_operation.Init(exec_order.size());
  for (const auto &node : exec_order) {
    AddSingleParameterOperations(graph, node, exec_order, callback, params_operation);
  }
  return params_operation;
}

HyperOffloadOperations HyperOffloadOptimizer::GenerateActivationOperations(const FuncGraphPtr &graph,
                                                                           const CNodePtrList &exec_order,
                                                                           const HandleNewNodeCallback &callback) {
  MS_EXCEPTION_IF_NULL(graph);
  HyperOffloadOperations activation_operation;
  activation_operation.Init(exec_order.size());

  const auto &user_info_list = CollectAllNodeUsers(exec_order);
  DistanceBaseHyperOffloadStrategy strategy;
  auto offload_info_list = strategy.Run(exec_order, user_info_list);
  OffloadInfoFilterByNumber filter;
  offload_info_list = filter.Filter(offload_info_list);

  if (offload_info_list.empty()) {
    return activation_operation;
  }

  const auto &outputs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
  for (const auto &offload_info : offload_info_list) {
    AddSingleActivationOperations(offload_info, outputs, graph, exec_order, callback, activation_operation);
  }
  return activation_operation;
}

CNodePtrList HyperOffloadOptimizer::BuildExecutionOrder(const CNodePtrList &execution_order,
                                                        const HyperOffloadOperations &hyper_offload_operations) {
  CNodePtrList new_execution_order;
  const auto &pre_operations = hyper_offload_operations.GetPreOperations().operations;
  std::copy(pre_operations.begin(), pre_operations.end(), std::back_inserter(new_execution_order));

  MS_EXCEPTION_IF_CHECK_FAIL(hyper_offload_operations.GetOperations().size() == execution_order.size(),
                             "The size of operations is not equal to the size of original order.");

  for (size_t i = 0; i < execution_order.size(); ++i) {
    new_execution_order.push_back(execution_order[i]);
    const auto &cur_operations = hyper_offload_operations.GetOperations()[i].operations;
    std::copy(cur_operations.begin(), cur_operations.end(), std::back_inserter(new_execution_order));
  }
  return new_execution_order;
}

void HyperOffloadOptimizer::AssignHyperOffloadIds(const CNodePtrList &exec_order) {
  std::unordered_map<AnfNodePtr, std::string> node_to_key_cache;

  std::function<std::string(const AnfNodePtr &)> get_key = [&](const AnfNodePtr &node) -> std::string {
    if (node == nullptr) {
      return "";
    }
    if (node_to_key_cache.count(node)) {
      return node_to_key_cache[node];
    }

    std::string key;
    if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_CHECK_FAIL(cnode->inputs().size() > 1, "Node inputs size check failed.");
      auto index = common::AnfAlgo::GetTupleGetItemOutIndex(cnode);
      key = get_key(cnode->input(1)) + "_idx_" + std::to_string(index);
    } else if (IsD2HNode(node) || IsH2DNode(node) ||
               IsOneOfPrimitiveCNode(node,
                                     {kFreeDataPrimitive, kAssignDataPrimitive, prim::kPrimSend, prim::kPrimReceive})) {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_CHECK_FAIL(cnode->inputs().size() > 1, "Node inputs size check failed.");
      key = get_key(cnode->input(1));
    } else {
      key = node->fullname_with_scope();
    }

    node_to_key_cache[node] = key;
    return key;
  };

  for (const auto &node : exec_order) {
    auto key = get_key(node);
    if (key.empty()) {
      continue;
    }
    size_t id = std::hash<std::string>{}(key);
    node->set_user_data(kHyperOffloadId, std::make_shared<size_t>(id));
  }
}

void HyperOffloadOptimizer::AdjustHyperOffloadNodePosition(HyperOffloadOperations &hyper_offload_operations) {
  const auto &h2d_nodes = hyper_offload_operations.GetAllH2DNodes();
  static size_t prefetch_distance = HyperOffloadConfig::GetInstance().prefetch_distance();
  for (const auto &h2d_node : h2d_nodes) {
    hyper_offload_operations.MoveAhead(h2d_node, prefetch_distance);
  }
}
}  // namespace hyper_offload
}  // namespace utils
}  // namespace mindspore
