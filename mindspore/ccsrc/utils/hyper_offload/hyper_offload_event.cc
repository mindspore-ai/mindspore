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

#include <algorithm>
#include <limits>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>
#include "utils/log_adapter.h"

namespace mindspore {
namespace utils {
namespace hyper_offload {
namespace {
constexpr size_t kPrimaryDataInputIndex = 1;

size_t GetReleaseTargetIndex(const CNodePtrList &exec_order, size_t node_index, size_t last_user_index,
                             size_t release_distance) {
  if (exec_order.empty()) {
    return 0;
  }
  size_t target_index = last_user_index;
  if (release_distance > std::numeric_limits<size_t>::max() - target_index) {
    target_index = exec_order.size() - 1;
  } else {
    target_index += release_distance;
    if (target_index >= exec_order.size()) {
      target_index = exec_order.size() - 1;
    }
  }
  if (target_index < node_index) {
    return node_index;
  }
  return target_index;
}
}  // namespace

HyperOffloadEventInjector::HyperOffloadEventInjector(const FuncGraphPtr &graph, const CNodePtrList &exec_order,
                                                     HyperOffloadOperations *operations)
    : graph_(graph), exec_order_(exec_order), operations_(operations) {}

void HyperOffloadEventInjector::Run() {
  MS_EXCEPTION_IF_NULL(graph_);
  MS_EXCEPTION_IF_NULL(operations_);
  MS_LOG(INFO) << "HyperOffloadEventInjector::Run started";
  EventNodeFactory event_factory(graph_);
  AddEventNodesForD2HNodes(event_factory);
  AddEventNodesForH2DNodes(event_factory);
  MS_LOG(INFO) << "HyperOffloadEventInjector::Run completed";
}

std::map<KernelWithIndex, size_t> HyperOffloadEventInjector::BuildDataNodeLastUserIndexMap() const {
  std::map<KernelWithIndex, size_t> data_node_last_user_index_map;
  for (size_t i = 0; i < exec_order_.size(); ++i) {
    const auto &node = exec_order_[i];
    for (const auto &input : node->inputs()) {
      const auto &[cur_node, cur_index] = GetRealDataNodeWithIndex(input);
      MS_EXCEPTION_IF_NULL(cur_node);
      if (!cur_node->isa<CNode>()) {
        continue;
      }
      data_node_last_user_index_map[{cur_node, cur_index}] = i;
    }
  }
  return data_node_last_user_index_map;
}

void HyperOffloadEventInjector::AddEventNodesForSingleD2HNode(
  const CNodePtr &d2h_node, EventNodeFactory &event_factory, size_t node_index,
  const std::map<KernelWithIndex, size_t> &data_node_last_user_index_map) {
  MS_LOG(DEBUG) << "Adding event nodes for single D2H node at index " << node_index;
  MS_EXCEPTION_IF_CHECK_FAIL(d2h_node->inputs().size() > kPrimaryDataInputIndex, "D2H node input size invalid.");
  auto data_node = d2h_node->input(kPrimaryDataInputIndex);

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

  auto free_node = BuildFreeDataNode(graph_, data_node, kDeviceType);
  auto assign_node = BuildAssignDataNode(graph_, data_node, d2h_node, kHostType);
  auto free_cnode = free_node->cast<CNodePtr>();
  auto assign_cnode = assign_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(free_cnode);
  MS_EXCEPTION_IF_NULL(assign_cnode);

  bool use_immediate_release = (HyperOffloadConfig::GetInstance().max_device_memory() > 0);
  if (use_immediate_release) {
    // Peak strategy: Recv_Completion, FreeData, AssignData right after D2H.
    operations_->AddOperationAfter(node_index, recv_completion);
    operations_->AddOperationAfter(node_index, free_cnode);
    operations_->AddOperationAfter(node_index, assign_cnode);
    MS_LOG(DEBUG) << "Added event nodes for D2H, immediate release at index " << node_index;
  } else {
    // Distance strategy: use pre-computed map for O(1) lookup.
    const auto &[target_node, target_output_index] = GetRealDataNodeWithIndex(data_node);
    size_t last_user_index = node_index;
    auto it = data_node_last_user_index_map.find({target_node, target_output_index});
    if (it != data_node_last_user_index_map.end() && it->second > node_index) {
      last_user_index = it->second;
    }
    const size_t release_distance =
      GetNodeReleaseDistance(data_node).value_or(HyperOffloadConfig::GetInstance().release_distance());
    size_t target_index = GetReleaseTargetIndex(exec_order_, node_index, last_user_index, release_distance);
    operations_->AddOperationAfter(target_index, recv_completion);
    operations_->AddOperationAfter(target_index, free_cnode);
    operations_->AddOperationAfter(target_index, assign_cnode);
    MS_LOG(DEBUG) << "Added event nodes for D2H, release at index " << target_index;
  }
}

void HyperOffloadEventInjector::AddEventNodesForD2HNodes(EventNodeFactory &event_factory) {
  MS_LOG(INFO) << "Adding event nodes for D2H nodes";
  const auto &data_node_last_user_index_map = BuildDataNodeLastUserIndexMap();
  const auto &all_ops = operations_->GetOperations();
  MS_EXCEPTION_IF_CHECK_FAIL(all_ops.size() == exec_order_.size(),
                             "The size of operations is not equal to the size of original order.");
  size_t d2h_count = 0;
  for (size_t i = 0; i < exec_order_.size(); ++i) {
    auto current_ops = all_ops[i].operations;
    for (const auto &node : current_ops) {
      if (IsD2HNode(node)) {
        AddEventNodesForSingleD2HNode(node, event_factory, i, data_node_last_user_index_map);
        ++d2h_count;
      }
    }
  }
  MS_LOG(INFO) << "Added event nodes for " << d2h_count << " D2H nodes";
}

void HyperOffloadEventInjector::AddEventNodesForNonLastH2DNode(const CNodePtr &h2d_node,
                                                               const AnfNodePtr &original_data, size_t first_user_idx,
                                                               size_t last_user_idx, const CNodePtr &recv_node) {
  MS_LOG(DEBUG) << "Adding event nodes for non-last H2D node, first_user: " << first_user_idx
                << ", last_user: " << last_user_idx;
  operations_->AddOperationBefore(first_user_idx, recv_node);
  // Before First User: AssignData(A, A2, Device)
  auto assign_dev = BuildAssignDataNode(graph_, original_data, h2d_node, kDeviceType);
  auto assign_dev_cnode = assign_dev->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(assign_dev_cnode);
  operations_->AddOperationBefore(first_user_idx, assign_dev_cnode);

  // After Last User: FreeData(A, Device) -> AssignData(A, A1, Host)
  MS_EXCEPTION_IF_CHECK_FAIL(h2d_node->inputs().size() > kPrimaryDataInputIndex, "H2D node input size invalid.");
  auto input_node = h2d_node->input(kPrimaryDataInputIndex);
  auto free_dev = BuildFreeDataNode(graph_, original_data, kDeviceType);
  auto assign_host = BuildAssignDataNode(graph_, original_data, input_node, kHostType);

  auto free_dev_cnode = free_dev->cast<CNodePtr>();
  auto assign_host_cnode = assign_host->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(free_dev_cnode);
  MS_EXCEPTION_IF_NULL(assign_host_cnode);
  operations_->AddOperationAfter(last_user_idx, free_dev_cnode);
  operations_->AddOperationAfter(last_user_idx, assign_host_cnode);
}

void HyperOffloadEventInjector::AddEventNodesForLastH2DNode(const CNodePtr &h2d_node, const AnfNodePtr &original_data,
                                                            size_t first_user_idx, const CNodePtr &recv_node) {
  MS_LOG(DEBUG) << "Adding event nodes for last H2D node, first_user: " << first_user_idx;
  operations_->AddOperationBefore(first_user_idx, recv_node);
  // Before First User: FreeData(A, Host) -> AssignData(A, A3, Device)
  auto free_host = BuildFreeDataNode(graph_, original_data, kHostType);
  auto assign_dev = BuildAssignDataNode(graph_, original_data, h2d_node, kDeviceType);

  auto free_host_cnode = free_host->cast<CNodePtr>();
  auto assign_dev_cnode = assign_dev->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(free_host_cnode);
  MS_EXCEPTION_IF_NULL(assign_dev_cnode);
  operations_->AddOperationBefore(first_user_idx, free_host_cnode);
  operations_->AddOperationBefore(first_user_idx, assign_dev_cnode);
}

HyperOffloadEventInjector::H2DGroupMap HyperOffloadEventInjector::GroupH2DNodes() const {
  H2DGroupMap h2d_groups;
  const auto &all_ops = operations_->GetOperations();
  for (size_t i = 0; i < all_ops.size(); ++i) {
    for (const auto &node : all_ops[i].operations) {
      if (IsH2DNode(node)) {
        MS_EXCEPTION_IF_CHECK_FAIL(node->inputs().size() > kPrimaryDataInputIndex, "H2D node input size invalid.");
        auto input_node = node->input(kPrimaryDataInputIndex);
        (void)h2d_groups[input_node].emplace_back(i, node);
      }
    }
  }
  MS_LOG(INFO) << "Grouped H2D nodes into " << h2d_groups.size() << " groups";
  return h2d_groups;
}

std::unordered_map<AnfNodePtr, std::pair<size_t, size_t>> HyperOffloadEventInjector::BuildH2DUserIndicesMap() const {
  std::unordered_map<AnfNodePtr, std::pair<size_t, size_t>> h2d_user_indices;
  for (size_t k = 0; k < exec_order_.size(); ++k) {
    for (const auto &input : exec_order_[k]->inputs()) {
      if (auto [it, inserted] = h2d_user_indices.try_emplace(input, k, k); !inserted) {
        it->second.second = k;
      }
    }
  }
  return h2d_user_indices;
}

void HyperOffloadEventInjector::ProcessH2DGroup(
  const AnfNodePtr &input_node, const std::vector<std::pair<size_t, CNodePtr>> &h2d_list,
  EventNodeFactory &event_factory, const std::unordered_map<AnfNodePtr, std::pair<size_t, size_t>> &h2d_user_indices) {
  MS_LOG(DEBUG) << "Processing H2D group with " << h2d_list.size() << " nodes";
  AnfNodePtr original_data = nullptr;
  if (IsD2HNode(input_node)) {
    auto d2h_node = input_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(d2h_node);
    MS_EXCEPTION_IF_CHECK_FAIL(d2h_node->inputs().size() > kPrimaryDataInputIndex, "D2H node input size invalid.");
    original_data = d2h_node->input(kPrimaryDataInputIndex);
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

    auto indices_it = h2d_user_indices.find(h2d_node);
    if (indices_it == h2d_user_indices.end()) {
      MS_LOG(EXCEPTION) << "H2D node " << h2d_node->fullname_with_scope() << " has no users in execution order.";
    }
    auto [first_user_idx, last_user_idx] = indices_it->second;

    if (!is_last) {
      AddEventNodesForNonLastH2DNode(h2d_node, original_data, first_user_idx, last_user_idx, recv_node);
    } else {
      AddEventNodesForLastH2DNode(h2d_node, original_data, first_user_idx, recv_node);
    }
  }
}

void HyperOffloadEventInjector::AddEventNodesForH2DNodes(EventNodeFactory &event_factory) {
  MS_LOG(INFO) << "Adding event nodes for H2D nodes";
  auto h2d_groups = GroupH2DNodes();
  auto h2d_user_indices = BuildH2DUserIndicesMap();
  for (const auto &group_entry : h2d_groups) {
    ProcessH2DGroup(group_entry.first, group_entry.second, event_factory, h2d_user_indices);
  }
  MS_LOG(INFO) << "Added event nodes for H2D nodes, total groups: " << h2d_groups.size();
}
}  // namespace hyper_offload
}  // namespace utils
}  // namespace mindspore
