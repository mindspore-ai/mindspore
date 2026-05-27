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

#include "utils/hyper_offload/hyper_offload_operation.h"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "utils/log_adapter.h"
#include "utils/hyper_offload/hyper_offload_utils.h"

namespace mindspore {
namespace utils {
namespace hyper_offload {
namespace {
void RemoveNodeFromOperations(CNodePtrList *nodes, const CNodePtr &target_node) {
  MS_EXCEPTION_IF_NULL(nodes);
  nodes->erase(std::remove(nodes->begin(), nodes->end(), target_node), nodes->end());
}
}  // namespace

void HyperOffloadOperations::Init(size_t size) {
  operations_.clear();
  operations_.resize(size);
  pre_operations_.operations.clear();
}

void HyperOffloadOperations::AddPreOperation(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  pre_operations_.operations.push_back(node);
}

void HyperOffloadOperations::AddOperationAfter(size_t index, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_CHECK_FAIL(index < operations_.size(), "Index out of range for AddOperationAfter.");
  operations_[index].operations.push_back(node);
}

void HyperOffloadOperations::AddOperationBefore(size_t index, const CNodePtr &node) {
  if (index == 0) {
    AddPreOperation(node);
    return;
  }
  MS_EXCEPTION_IF_CHECK_FAIL(index <= operations_.size(), "Index out of range for AddOperationBefore.");
  AddOperationAfter(index - 1, node);
}

void HyperOffloadOperations::MergeFrom(const HyperOffloadOperations &other) {
  MS_LOG(DEBUG) << "Merging HyperOffloadOperations, current size: " << operations_.size()
                << ", other size: " << other.operations_.size();
  if (operations_.empty()) {
    operations_.resize(other.operations_.size());
  }
  MS_EXCEPTION_IF_CHECK_FAIL(operations_.size() == other.operations_.size(),
                             "Size mismatch when merging HyperOffloadOperations.");
  for (size_t i = 0; i < operations_.size(); ++i) {
    auto &dst = operations_[i].operations;
    const auto &src = other.operations_[i].operations;
    dst.insert(dst.end(), src.begin(), src.end());
  }
  pre_operations_.operations.insert(pre_operations_.operations.end(), other.pre_operations_.operations.begin(),
                                    other.pre_operations_.operations.end());
}

void HyperOffloadOperations::RemoveNode(const CNodePtr &node) {
  RemoveNodeFromOperations(&pre_operations_.operations, node);
  for (auto &slot : operations_) {
    RemoveNodeFromOperations(&slot.operations, node);
  }
}

HyperOffloadOperations::AttachPosition HyperOffloadOperations::GetAttachPosition(const CNodePtr &node) const {
  AttachPosition pos;
  const auto &pre_vec = pre_operations_.operations;
  if (std::find(pre_vec.begin(), pre_vec.end(), node) != pre_vec.end()) {
    pos.is_pre = true;
    pos.index = 0;
    return pos;
  }
  for (size_t i = 0; i < operations_.size(); ++i) {
    const auto &vec = operations_[i].operations;
    if (std::find(vec.begin(), vec.end(), node) != vec.end()) {
      pos.is_pre = false;
      pos.index = i;
      return pos;
    }
  }
  pos.is_pre = true;
  pos.index = 0;
  return pos;
}

CNodePtrList HyperOffloadOperations::GetAllH2DNodes() const {
  CNodePtrList h2d_nodes;
  auto collect_h2d = [&](const auto &nodes) {
    std::copy_if(nodes.begin(), nodes.end(), std::back_inserter(h2d_nodes),
                 [](const CNodePtr &node) { return IsH2DNode(node); });
  };

  collect_h2d(GetPreOperations().operations);
  for (const auto &e : GetOperations()) {
    collect_h2d(e.operations);
  }
  return h2d_nodes;
}

size_t HyperOffloadOperations::GetNodeAttachIndex(const CNodePtr &node) const {
  auto pos = GetAttachPosition(node);
  return pos.is_pre ? 0 : pos.index;
}

void HyperOffloadOperations::MoveAhead(const CNodePtr &node, size_t ahead_distance, const CNodePtr &must_remain_after,
                                       std::optional<size_t> min_attach_after_exec_idx) {
  MS_EXCEPTION_IF_NULL(node);
  if (ahead_distance == 0 && !min_attach_after_exec_idx.has_value()) {
    return;
  }
  auto pos = GetAttachPosition(node);
  if (pos.is_pre) {
    return;
  }

  size_t capped_req_distance = ahead_distance;
  if (must_remain_after != nullptr) {
    const size_t after_index = GetNodeAttachIndex(must_remain_after);
    if (pos.index > after_index) {
      const size_t max_by_d2h = pos.index - after_index - 1;
      capped_req_distance = std::min(ahead_distance, max_by_d2h);
    }
  }

  const size_t old_slot = pos.index;
  bool use_pre_ops = (old_slot < capped_req_distance);
  size_t new_index = 0;
  if (!use_pre_ops) {
    new_index = old_slot - capped_req_distance;
  }

  if (min_attach_after_exec_idx.has_value()) {
    const size_t min_attach_index = *min_attach_after_exec_idx;
    if (use_pre_ops || new_index < min_attach_index) {
      new_index = min_attach_index;
      use_pre_ops = false;
    }
  }

  RemoveNode(node);
  if (use_pre_ops) {
    AddPreOperation(node);
  } else {
    AddOperationAfter(new_index, node);
  }
}

const NodeOperations &HyperOffloadOperations::GetPreOperations() const { return pre_operations_; }

const std::vector<NodeOperations> &HyperOffloadOperations::GetOperations() const { return operations_; }

void HyperOffloadOperations::Print(const CNodePtrList &origin_execution_order, bool is_error) const {
  auto log_func = [is_error](const std::string &msg) {
    if (is_error) {
      MS_LOG(ERROR) << msg;
    } else {
      MS_LOG(INFO) << msg;
    }
  };

  if (operations_.size() != origin_execution_order.size()) {
    std::ostringstream oss;
    oss << "HyperOffloadOperations size mismatch: operations_ size " << operations_.size()
        << " vs origin execution order size " << origin_execution_order.size();
    log_func(oss.str());
    return;
  }

  // Helper to print added nodes with specific attributes
  auto print_added_node = [&](const CNodePtr &node) {
    std::ostringstream oss;
    oss << "    [Added] " << node->DebugString();
    if (IsViewNode(node)) {
      oss << " [View]";
    }
    if (IsInplaceNode(node)) {
      oss << " [Inplace]";
    }
    auto event_id_ptr = node->user_data<uint32_t>(kHyperOffloadEventId);
    if (event_id_ptr != nullptr) {
      oss << " EventId: " << *event_id_ptr;
    }
    auto pair_id_ptr = node->user_data<size_t>(kHyperOffloadEventStreamId);
    if (pair_id_ptr != nullptr) {
      oss << " StreamId: " << *pair_id_ptr;
    }

    auto offload_id = GetHyperOffloadId(node);
    if (offload_id.has_value()) {
      oss << " OffloadId: " << offload_id.value();
    }

    log_func(oss.str());
  };

  log_func("========== HyperOffloadOperations Trace Start ==========");

  // Print Pre-Operations
  if (!pre_operations_.operations.empty()) {
    log_func("[Pre-Operations]:");
    for (const auto &node : pre_operations_.operations) {
      print_added_node(node);
    }
  }

  // Print Original Order and Attached Operations
  for (size_t i = 0; i < origin_execution_order.size(); ++i) {
    auto origin_node = origin_execution_order[i];
    std::ostringstream oss;
    oss << "[" << i << "] [Origin] " << origin_node->DebugString();
    if (IsViewNode(origin_node)) {
      oss << " [View]";
    }
    if (IsInplaceNode(origin_node)) {
      oss << " [Inplace]";
    }
    auto origin_offload_id = GetHyperOffloadId(origin_node);
    if (origin_offload_id.has_value()) {
      oss << " OffloadId: " << origin_offload_id.value();
    }
    log_func(oss.str());

    const auto &added_ops = operations_[i].operations;
    for (const auto &node : added_ops) {
      print_added_node(node);
    }
  }

  log_func("========== HyperOffloadOperations Trace End ==========");
}
}  // namespace hyper_offload
}  // namespace utils
}  // namespace mindspore
