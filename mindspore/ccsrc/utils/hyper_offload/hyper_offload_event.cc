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

#include "utils/hyper_offload/hyper_offload_event.h"
#include <unordered_map>

namespace mindspore {
namespace utils {
namespace hyper_offload {
HyperOffloadEventInjector::HyperOffloadEventInjector(const FuncGraphPtr &graph, const CNodePtrList &exec_order,
                                                     HyperOffloadOperations *operations)
    : graph_(graph), exec_order_(exec_order), operations_(operations) {}

void HyperOffloadEventInjector::Run() {
  MS_EXCEPTION_IF_NULL(graph_);
  MS_EXCEPTION_IF_NULL(operations_);
  EventNodeFactory event_factory(graph_);
  AddEventNodesForD2HNodes(event_factory);
  AddEventNodesForH2DNodes(event_factory);
}

void HyperOffloadEventInjector::AddEventNodesForSingleD2HNode(const CNodePtr &d2h_node, EventNodeFactory &event_factory,
                                                              size_t node_index) {
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
  operations_->RemoveNode(d2h_node);
  operations_->AddOperationAfter(node_index, send_creation);
  operations_->AddOperationAfter(node_index, recv_creation);
  operations_->AddOperationAfter(node_index, d2h_node);
  operations_->AddOperationAfter(node_index, send_completion);

  // Handle Recv_Completion insertion
  const size_t release_distance = HyperOffloadConfig::GetInstance().release_distance();
  size_t target_index = node_index + release_distance;
  if (target_index >= exec_order_.size()) {
    target_index = exec_order_.size() - 1;
  }
  // Insert Recv_Completion after the target node.
  operations_->AddOperationAfter(target_index, recv_completion);

  auto free_node = BuildFreeDataNode(graph_, data_node, kDeviceType);
  auto assign_node = BuildAssignDataNode(graph_, data_node, d2h_node, kHostType);
  operations_->AddOperationAfter(target_index, free_node->cast<CNodePtr>());
  operations_->AddOperationAfter(target_index, assign_node->cast<CNodePtr>());
}

void HyperOffloadEventInjector::AddEventNodesForD2HNodes(EventNodeFactory &event_factory) {
  // Iterate over the original execution order
  // We make a copy of the operations list for each index because we modify the ops object
  // which might affect the underlying storage or iterators if we were iterating them directly.
  const auto &all_ops = operations_->GetOperations();
  MS_EXCEPTION_IF_CHECK_FAIL(all_ops.size() == exec_order_.size(),
                             "The size of operations is not equal to the size of original order.");
  for (size_t i = 0; i < exec_order_.size(); ++i) {
    // Copy the list of operations at this index
    auto current_ops = all_ops[i].operations;
    for (const auto &node : current_ops) {
      if (IsD2HNode(node)) {
        AddEventNodesForSingleD2HNode(node, event_factory, i);
      }
    }
  }
}

void HyperOffloadEventInjector::AddEventNodesForNonLastH2DNode(const CNodePtr &h2d_node,
                                                               const AnfNodePtr &original_data, size_t first_user_idx,
                                                               size_t last_user_idx, const CNodePtr &recv_node) {
  operations_->AddOperationBefore(first_user_idx, recv_node);
  // Before First User: AssignData(A, A2, Device)
  auto assign_dev = BuildAssignDataNode(graph_, original_data, h2d_node, kDeviceType);
  operations_->AddOperationBefore(first_user_idx, assign_dev->cast<CNodePtr>());

  // After Last User: FreeData(A, Device) -> AssignData(A, A1, Host)
  constexpr size_t data_index = 1;
  auto input_node = h2d_node->input(data_index);
  auto free_dev = BuildFreeDataNode(graph_, original_data, kDeviceType);
  auto assign_host = BuildAssignDataNode(graph_, original_data, input_node, kHostType);

  operations_->AddOperationAfter(last_user_idx, free_dev->cast<CNodePtr>());
  operations_->AddOperationAfter(last_user_idx, assign_host->cast<CNodePtr>());
}

void HyperOffloadEventInjector::AddEventNodesForLastH2DNode(const CNodePtr &h2d_node, const AnfNodePtr &original_data,
                                                            size_t first_user_idx, const CNodePtr &recv_node) {
  operations_->AddOperationBefore(first_user_idx, recv_node);
  // Before First User: FreeData(A, Host) -> AssignData(A, A3, Device)
  auto free_host = BuildFreeDataNode(graph_, original_data, kHostType);
  auto assign_dev = BuildAssignDataNode(graph_, original_data, h2d_node, kDeviceType);

  operations_->AddOperationBefore(first_user_idx, free_host->cast<CNodePtr>());
  operations_->AddOperationBefore(first_user_idx, assign_dev->cast<CNodePtr>());
}

HyperOffloadEventInjector::H2DGroupMap HyperOffloadEventInjector::GroupH2DNodes() const {
  H2DGroupMap h2d_groups;
  constexpr size_t data_index = 1;
  const auto &all_ops = operations_->GetOperations();
  for (size_t i = 0; i < all_ops.size(); ++i) {
    for (const auto &node : all_ops[i].operations) {
      if (IsH2DNode(node)) {
        auto input_node = node->input(data_index);
        (void)h2d_groups[input_node].emplace_back(i, node);
      }
    }
  }
  return h2d_groups;
}

std::pair<size_t, size_t> HyperOffloadEventInjector::FindH2DUserIndices(const CNodePtr &h2d_node) const {
  size_t first_user_idx = exec_order_.size();
  size_t last_user_idx = 0;
  bool found_user = false;

  for (size_t k = 0; k < exec_order_.size(); ++k) {
    auto node = exec_order_[k];
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
  return {first_user_idx, last_user_idx};
}

void HyperOffloadEventInjector::ProcessH2DGroup(const AnfNodePtr &input_node,
                                                const std::vector<std::pair<size_t, CNodePtr>> &h2d_list,
                                                EventNodeFactory &event_factory) {
  AnfNodePtr original_data = nullptr;
  constexpr size_t data_index = 1;
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
    operations_->RemoveNode(h2d_node);
    operations_->AddOperationAfter(node_index, h2d_node);
    operations_->AddOperationAfter(node_index, send_node);

    // Find first and last user in exec_order
    auto [first_user_idx, last_user_idx] = FindH2DUserIndices(h2d_node);

    if (!is_last) {
      AddEventNodesForNonLastH2DNode(h2d_node, original_data, first_user_idx, last_user_idx, recv_node);
    } else {
      AddEventNodesForLastH2DNode(h2d_node, original_data, first_user_idx, recv_node);
    }
  }
}

void HyperOffloadEventInjector::AddEventNodesForH2DNodes(EventNodeFactory &event_factory) {
  auto h2d_groups = GroupH2DNodes();

  for (const auto &group_entry : h2d_groups) {
    ProcessH2DGroup(group_entry.first, group_entry.second, event_factory);
  }
}
}  // namespace hyper_offload
}  // namespace utils
}  // namespace mindspore
