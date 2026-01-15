/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "pynative/hyper_offload/offload_strategy.h"
#include <sstream>
#include "utils/log_adapter.h"
#include "utils/hyper_offload/hyper_offload_utils.h"
#include "include/utils/anfalgo.h"
#include "mindspore/ops/include/primitive/framework_ops.h"
#include "mindspore/ops/ops_utils/op_constants.h"

namespace mindspore {
namespace pynative {
namespace hyper_offload {

namespace {
// Stream ID constants (from graph side)
constexpr uint32_t kGraphComputeStreamId = 0;
constexpr uint32_t kGraphCopyStreamId = 1;

// Helper to check if node is Send primitive
bool IsSendNode(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  return prim != nullptr && prim->name() == prim::kPrimSend->name();
}

// Helper to check if node is Receive primitive
bool IsReceiveNode(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  return prim != nullptr && prim->name() == prim::kPrimReceive->name();
}

bool IsWaitD2H(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);

  auto event_type = utils::hyper_offload::GetEventNodeType(cnode);
  if (event_type.has_value() && event_type == utils::hyper_offload::EventNodeType::kSendRecvAfterD2H) {
    return true;
  }
  return false;
}

bool IsWaitH2D(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto event_type = utils::hyper_offload::GetEventNodeType(cnode);
  if (event_type.has_value() && event_type == utils::hyper_offload::EventNodeType::kSendRecvAfterH2D) {
    return true;
  }
  return false;
}

// Helper to check if node is FreeData primitive
bool IsFreeDataNode(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  return prim != nullptr && prim->name() == "FreeData";
}

bool IsFreeCache(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  auto type = GetValueNode<UInt64ImmPtr>(cnode->input(2))->value();
  return prim != nullptr && prim->name() == "FreeData" && type == utils::hyper_offload::kHostType;
}

// Helper to check if node is AssignData primitive
bool IsAssignDataNode(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  return prim != nullptr && prim->name() == "AssignData";
}

// Get stream_id from node attribute
uint32_t GetStreamId(const CNodePtr &cnode) {
  auto stream_id = utils::hyper_offload::GetSendOrRecvStreamId(cnode);
  if (stream_id.has_value()) {
    return stream_id.value();
  }
  return UINT32_MAX;  // Invalid stream id
}

size_t GetTensorId(const CNodePtr &cnode) {
  auto tensor_id = utils::hyper_offload::GetHyperOffloadId(cnode);
  if (tensor_id.has_value()) {
    return tensor_id.value();
  }
  MS_LOG(EXCEPTION) << "GetTensorId failed for node: " << cnode->DebugString();
  return 0;
}

bool IsTupleGetItemNode(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  return prim != nullptr && prim->name() == prim::kPrimTupleGetItem->name();
}

size_t GetOutputIndex(const CNodePtr &d2h_node) {
  MS_EXCEPTION_IF_NULL(d2h_node);
  if (d2h_node->size() < kIndex2) {
    return 0;
  }

  auto input_node = d2h_node->input(kIndex1);
  if (input_node == nullptr || !input_node->isa<CNode>()) {
    return 0;
  }

  auto input_cnode = input_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(input_cnode);
  if (!IsTupleGetItemNode(input_cnode)) {
    return 0;
  }

  if (input_cnode->size() < kIndex3) {
    return 0;
  }

  auto index_node = input_cnode->input(2);
  if (index_node == nullptr) {
    return 0;
  }

  auto index_value = GetValueNode<Int64ImmPtr>(index_node);
  if (index_value != nullptr) {
    return static_cast<size_t>(index_value->value());
  }

  return 0;
}

// ========== Pattern Matcher ==========
// Matches operation sequences to TensorAction

struct PatternMatchResult {
  TensorAction action{TensorAction::kNone};
  size_t consumed_count{0};  // Number of nodes consumed by this pattern
  size_t tensor_id{0};
  size_t output_index{0};  // Output index for multi-output operators (from TupleGetItem)
};

// Pattern: Send(0) + Recv(1) + D2H + Send(1) -> kD2HWithEvent
PatternMatchResult TryMatchD2HWithEvent(const CNodePtrList &nodes, size_t start_idx) {
  PatternMatchResult result;
  if (start_idx + 3 >= nodes.size()) {
    return result;
  }

  const auto &n0 = nodes[start_idx];
  const auto &n1 = nodes[start_idx + 1];
  const auto &n2 = nodes[start_idx + 2];
  const auto &n3 = nodes[start_idx + 3];

  // Check: Send(stream_id=0) + Recv(stream_id=1) + D2H + Send(stream_id=1)
  if (IsSendNode(n0) && GetStreamId(n0) == kGraphComputeStreamId && IsReceiveNode(n1) &&
      GetStreamId(n1) == kGraphCopyStreamId && utils::hyper_offload::IsD2HNode(n2) && IsSendNode(n3) &&
      GetStreamId(n3) == kGraphCopyStreamId) {
    result.action = TensorAction::kD2HWithEvent;
    result.consumed_count = 4;
    result.tensor_id = GetTensorId(n2);
    result.output_index = GetOutputIndex(n2);
    MS_LOG(DEBUG) << "Matched kD2HWithEvent pattern at index " << start_idx << " with tensor_id: " << result.tensor_id
                  << ", output_index: " << result.output_index;
  }
  return result;
}

// Pattern: Recv(stream_id=0) + FreeData + AssignData -> kWaitD2HAndSwitch
PatternMatchResult TryMatchWaitD2HAndSwitch(const CNodePtrList &nodes, size_t start_idx) {
  PatternMatchResult result;
  if (start_idx + 2 >= nodes.size()) {
    return result;
  }

  const auto &n0 = nodes[start_idx];
  const auto &n1 = nodes[start_idx + 1];
  const auto &n2 = nodes[start_idx + 2];

  // Check: Recv(stream_id=1) + FreeData + AssignData
  if (IsReceiveNode(n0) && GetStreamId(n0) == kGraphComputeStreamId && IsWaitD2H(n0) && IsFreeDataNode(n1) &&
      IsAssignDataNode(n2)) {
    result.action = TensorAction::kWaitD2HAndSwitch;
    result.consumed_count = 3;
    result.tensor_id = GetTensorId(n1);  // tensor_id from FreeData node
    MS_LOG(DEBUG) << "Matched kWaitD2HAndSwitch pattern at index " << start_idx
                  << "with tensor id: " << result.tensor_id;
  }
  return result;
}

// Pattern: H2D + Send(1) -> kH2DWithEvent
PatternMatchResult TryMatchH2DWithEvent(const CNodePtrList &nodes, size_t start_idx) {
  PatternMatchResult result;
  if (start_idx + 1 >= nodes.size()) {
    return result;
  }

  const auto &n0 = nodes[start_idx];
  const auto &n1 = nodes[start_idx + 1];

  // Check: H2D + Send(stream_id=1)
  if (utils::hyper_offload::IsH2DNode(n0) && IsSendNode(n1) && GetStreamId(n1) == kGraphCopyStreamId) {
    result.action = TensorAction::kH2DWithEvent;
    result.consumed_count = 2;
    result.tensor_id = GetTensorId(n0);  // tensor_id from H2D node
    MS_LOG(DEBUG) << "Matched kH2DWithEvent pattern at index " << start_idx << "with tensor id: " << result.tensor_id;
  }
  return result;
}

// Pattern: Recv(0) + AssignData -> kWaitH2DAndSwitch
PatternMatchResult TryMatchWaitH2DAndSwitch(const CNodePtrList &nodes, size_t start_idx) {
  PatternMatchResult result;
  if (start_idx + 1 >= nodes.size()) {
    return result;
  }

  const auto &n0 = nodes[start_idx];
  const auto &n1 = nodes[start_idx + 1];

  // Check: Recv(stream_id=0) + FreeData(host) + AssignData
  if (IsReceiveNode(n0) && GetStreamId(n0) == kGraphComputeStreamId && IsWaitH2D(n0) && IsAssignDataNode(n1)) {
    result.action = TensorAction::kWaitH2DAndSwitch;
    result.consumed_count = 2;
    result.tensor_id = GetTensorId(n1);  // tensor_id from FreeData node
    MS_LOG(DEBUG) << "Matched kWaitH2DAndSwitch pattern at index " << start_idx
                  << "with tensor id: " << result.tensor_id;
  }
  return result;
}

// Pattern: Recv(0) + FreeData(host) + AssignData -> kWaitH2DAndSwitch
PatternMatchResult TryMatchWaitH2DAndFreeCache(const CNodePtrList &nodes, size_t start_idx) {
  PatternMatchResult result;
  if (start_idx + 2 >= nodes.size()) {
    return result;
  }

  const auto &n0 = nodes[start_idx];
  const auto &n1 = nodes[start_idx + 1];
  const auto &n2 = nodes[start_idx + 2];

  // Check: Recv(stream_id=0) + FreeData(host) + AssignData
  if (IsReceiveNode(n0) && GetStreamId(n0) == kGraphComputeStreamId && IsWaitH2D(n0) && IsFreeDataNode(n1) &&
      IsFreeCache(n1) && IsAssignDataNode(n2)) {
    result.action = TensorAction::kWaitH2DAndFreeCache;
    result.consumed_count = 3;
    result.tensor_id = GetTensorId(n1);  // tensor_id from FreeData node
    MS_LOG(DEBUG) << "Matched kWaitH2DAndFreeCache pattern at index " << start_idx
                  << "with tensor id: " << result.tensor_id;
  }
  return result;
}

// Pattern: FreeData(device) + AssignData -> kFreeDevice
PatternMatchResult TryMatchFreeDevice(const CNodePtrList &nodes, size_t start_idx) {
  PatternMatchResult result;
  if (start_idx + 1 >= nodes.size()) {
    return result;
  }

  const auto &n0 = nodes[start_idx];
  const auto &n1 = nodes[start_idx + 1];

  if (IsFreeDataNode(n0) && !IsFreeCache(n0) && IsAssignDataNode(n1)) {
    result.action = TensorAction::kFreeDevice;
    result.consumed_count = 2;
    result.tensor_id = GetTensorId(n0);  // tensor_id from FreeData node
    MS_LOG(DEBUG) << "Matched kFreeDevice pattern at index " << start_idx << " with tensor_id: " << result.tensor_id;
  }
  return result;
}

// Try all patterns and return the first match
PatternMatchResult TryMatchPattern(const CNodePtrList &nodes, size_t start_idx) {
  // Try patterns in order of specificity (longer patterns first)
  PatternMatchResult result;

  // 1. Try kD2HWithEvent (4 nodes)
  result = TryMatchD2HWithEvent(nodes, start_idx);
  if (result.action != TensorAction::kNone) {
    return result;
  }

  // 2. Try kWaitD2HAndSwitch (3 nodes)
  result = TryMatchWaitD2HAndSwitch(nodes, start_idx);
  if (result.action != TensorAction::kNone) {
    return result;
  }

  // 3. Try kWaitH2DAndSwitch (3 nodes)
  result = TryMatchWaitH2DAndSwitch(nodes, start_idx);
  if (result.action != TensorAction::kNone) {
    return result;
  }

  // 4. Try kWaitH2DAndFreeCache (3 nodes)
  result = TryMatchWaitH2DAndFreeCache(nodes, start_idx);
  if (result.action != TensorAction::kNone) {
    return result;
  }

  // 5. Try kH2DWithEvent (2 nodes)
  result = TryMatchH2DWithEvent(nodes, start_idx);
  if (result.action != TensorAction::kNone) {
    return result;
  }

  // 6. Try kFreeDevice (2 nodes: FreeData + AssignData, no event wait)
  result = TryMatchFreeDevice(nodes, start_idx);
  if (result.action != TensorAction::kNone) {
    return result;
  }

  return result;  // No match, action = kNone
}

}  // namespace

void OffloadStrategy::ConvertFromPlan(const utils::hyper_offload::HyperOffloadPlan &plan) {
  position_ops_.clear();

  const auto &operations = plan.hyper_offload_operations.GetOperations();
  for (size_t exec_id = 0; exec_id < operations.size(); ++exec_id) {
    const auto &node_ops = operations[exec_id];
    if (node_ops.operations.empty()) {
      continue;
    }

    std::vector<TensorOpEntry> entries;
    const auto &cnodes = node_ops.operations;

    // Pattern matching: iterate through nodes and match patterns
    size_t idx = 0;
    while (idx < cnodes.size()) {
      PatternMatchResult match = TryMatchPattern(cnodes, idx);

      if (match.action != TensorAction::kNone) {
        TensorOpEntry entry;
        entry.action = match.action;
        entry.tensor_id = match.tensor_id;
        entry.source = TensorSource::kOutput;  // Default: output
        entry.index = match.output_index;      // Output index from pattern match (for multi-output ops)
        entries.push_back(entry);

        idx += match.consumed_count;  // Skip consumed nodes
      } else {
        // No pattern matched, skip this node
        idx++;
      }
    }

    if (!entries.empty()) {
      position_ops_[exec_id] = std::move(entries);
    }
  }

  is_ready_ = true;
  MS_LOG(INFO) << "OffloadStrategy converted. Total positions with ops: " << position_ops_.size();
}

const std::vector<TensorOpEntry> *OffloadStrategy::GetPreOpsAtPosition(size_t exec_id) const {
  auto it = pre_position_ops_.find(exec_id);
  if (it == pre_position_ops_.end()) {
    return nullptr;
  }
  return &it->second;
}

const std::vector<TensorOpEntry> *OffloadStrategy::GetOpsAtPosition(size_t exec_id) const {
  auto it = position_ops_.find(exec_id);
  if (it == position_ops_.end()) {
    return nullptr;
  }
  return &it->second;
}

bool OffloadStrategy::HasPreOpsAtPosition(size_t exec_id) const {
  return pre_position_ops_.find(exec_id) != pre_position_ops_.end();
}

bool OffloadStrategy::HasOpsAtPosition(size_t exec_id) const {
  return position_ops_.find(exec_id) != position_ops_.end();
}

std::string OffloadStrategy::DebugString() const {
  std::ostringstream oss;
  oss << "OffloadStrategy { is_ready=" << is_ready_ << ", post_positions=" << position_ops_.size()
      << ", pre_positions=" << pre_position_ops_.size() << " }\n";

  auto action_to_str = [](TensorAction action) -> std::string {
    switch (action) {
      case TensorAction::kD2HWithEvent:
        return "D2HWithEvent";
      case TensorAction::kH2DWithEvent:
        return "H2DWithEvent";
      case TensorAction::kWaitD2HAndSwitch:
        return "WaitD2HAndSwitch";
      case TensorAction::kWaitH2DAndSwitch:
        return "WaitH2DAndSwitch";
      case TensorAction::kWaitH2DAndFreeCache:
        return "WaitH2DAndFreeCache";
      case TensorAction::kFreeDevice:
        return "FreeDevice";
      default:
        return "None";
    }
  };

  auto source_to_str = [](TensorSource source) -> std::string {
    return source == TensorSource::kInput ? "input" : "output";
  };

  oss << "Pre-operations:\n";
  for (const auto &[exec_id, entries] : pre_position_ops_) {
    oss << "  exec_id=" << exec_id << ":\n";
    for (const auto &entry : entries) {
      oss << "    tensor_id=" << entry.tensor_id << ", " << source_to_str(entry.source) << "[" << entry.index << "] -> "
          << action_to_str(entry.action) << "\n";
    }
  }

  oss << "Post-operations:\n";
  for (const auto &[exec_id, entries] : position_ops_) {
    oss << "  exec_id=" << exec_id << ":\n";
    for (const auto &entry : entries) {
      oss << "    tensor_id=" << entry.tensor_id << ", " << source_to_str(entry.source) << "[" << entry.index << "] -> "
          << action_to_str(entry.action) << "\n";
    }
  }

  return oss.str();
}

}  // namespace hyper_offload
}  // namespace pynative
}  // namespace mindspore
