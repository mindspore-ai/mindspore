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
#include "frontend/jit/pi/graph_compiler/pi_ir/custom_nodes.h"

namespace mindspore {
namespace pijit {
namespace ir {
std::string RefNode::ToString() const {
  return "%" + std::to_string(GetNodeId()) + " = [" + GetType()->GetName() + "](" + GetNodeName() + ", " +
         std::to_string(real_node_->GetNodeId()) + ")\n";
}

std::string PlaceHolder::ToString() const {
  return "%" + std::to_string(GetNodeId()) + " = [" + GetType()->GetName() + "](" + GetNodeName() + ", " + tag_ + ")\n";
}

void SubscrNode::SetNodeId(size_t *id) {
  base_->SetNodeId(id);
  subscr_->SetNodeId(id);
}

void SubscrNode::SetOffset(size_t *offset) {
  base_->SetOffset(offset);
  subscr_->SetOffset(offset);
}

std::string SubscrNode::ToString() const {
  return base_->ToString() + "\n" + subscr_->ToString() + "\n%" + std::to_string(GetNodeId()) + " = %" +
         std::to_string(base_->GetNodeId()) + "[%" + std::to_string(subscr_->GetNodeId()) + "]\n";
}

void AttrNode::SetNodeId(size_t *id) {
  base_->SetNodeId(id);
  attr_->SetNodeId(id);
  Node::SetNodeId(id);
}

void AttrNode::SetOffset(size_t *offset) { base_->SetOffset(offset); }

std::string AttrNode::ToString() const {
  return base_->ToString() + "\n" + attr_->ToString() + "\n%" + std::to_string(GetNodeId()) + " = %" +
         std::to_string(base_->GetNodeId()) + ".%" + std::to_string(attr_->GetNodeId()) + "\n";
}

std::string PairNode::ToString() const {
  return first_->ToString() + "\n" + second_->ToString() + "\n%" + std::to_string(GetNodeId()) + " = (" +
         std::to_string(first_->GetNodeId()) + ", " + std::to_string(second_->GetNodeId()) + ")\n";
}

void JumpNode::SetNodeId(size_t *id) {
  auto left = GetLeftArg();
  if (left != nullptr) {
    left->SetNodeId(id);
  }
  Node::SetNodeId(id);
}

void JumpNode::SetOffset(size_t *offset) {
  auto left = GetLeftArg();
  if (left != nullptr) {
    left->SetOffset(offset);
  }
  Node::SetOffset(offset);
}

std::string JumpNode::ToString() const {
  std::string str;
  auto left = GetLeftArg();
  if (left != nullptr) {
    str += left->ToString() + "\n";
  }
  str += "%" + std::to_string(GetNodeId()) + " = " + GetNodeName() + "[" + GetType()->GetName() + "](" +
         GetOpName(GetOpCode());
  if (left != nullptr) {
    str += ", %" + std::to_string(left->GetNodeId());
  } else {
    str += ", nullptr";
  }
  auto right = GetRightArg();
  if (right != nullptr) {
    str += ", %" + std::to_string(right->GetNodeId());
  } else {
    str += ", nullptr";
  }
  return str + ")\n";
}

std::string CompareNode::ToString() const {
  auto left = GetLeftArg();
  auto right = GetRightArg();
  return left->ToString() + "\n" + right->ToString() + "\n%" + std::to_string(GetNodeId()) + " = " + GetNodeName() +
         "[" + GetType()->GetName() + "](" + GetOpName(GetOpCode()) + ", " + std::to_string(GetInstrArg()) + ", %" +
         std::to_string(left->GetNodeId()) + ", %" + std::to_string(right->GetNodeId()) + ")\n";
}
}  // namespace ir
}  // namespace pijit
}  // namespace mindspore
