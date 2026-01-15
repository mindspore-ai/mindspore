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
#include "utils/log_adapter.h"
#include "utils/hyper_offload/hyper_offload_utils.h"

namespace mindspore {
namespace utils {
namespace hyper_offload {
void HyperOffloadOperations::Init(size_t size) {
  operations_.clear();
  operations_.resize(size);
  pre_operations_.operations.clear();
}

void HyperOffloadOperations::AddPreOperation(const CNodePtr &node) { pre_operations_.operations.push_back(node); }

void HyperOffloadOperations::AddOperationAfter(size_t index, const CNodePtr &node) {
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
  auto &pre_vec = pre_operations_.operations;
  pre_vec.erase(std::remove(pre_vec.begin(), pre_vec.end(), node), pre_vec.end());
  for (auto &slot : operations_) {
    auto &vec = slot.operations;
    vec.erase(std::remove(vec.begin(), vec.end(), node), vec.end());
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

void HyperOffloadOperations::MoveAhead(const CNodePtr &node, size_t ahead_distance) {
  auto pos = GetAttachPosition(node);

  if (pos.is_pre) {
    return;
  }

  RemoveNode(node);
  if (pos.index >= ahead_distance) {
    size_t new_index = pos.index - ahead_distance;
    AddOperationAfter(new_index, node);
  } else {
    AddPreOperation(node);
  }
}

const NodeOperations &HyperOffloadOperations::GetPreOperations() const { return pre_operations_; }

const std::vector<NodeOperations> &HyperOffloadOperations::GetOperations() const { return operations_; }
}  // namespace hyper_offload
}  // namespace utils
}  // namespace mindspore
