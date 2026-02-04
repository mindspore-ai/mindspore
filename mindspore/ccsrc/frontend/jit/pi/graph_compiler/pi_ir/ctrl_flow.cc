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
#include "frontend/jit/pi/graph_compiler/pi_ir/ctrl_flow.h"
#include <sstream>

namespace mindspore {
namespace pijit {
namespace ir {
void Parameter::SetNodeId(size_t *id) {
  if (value_ != nullptr) {
    value_->SetNodeId(id);
  }
  if (default_value_ != nullptr) {
    default_value_->SetNodeId(id);
  }
  Node::SetNodeId(id);
}

std::string Parameter::ToString() const {
  std::stringstream ss;
  ss << (value_ == nullptr ? "" : value_->ToString()) << "\n";
  ss << (default_value_ == nullptr ? "" : default_value_->ToString()) << "\n";
  ss << "%" << std::to_string(GetNodeId()) << " = Parameter[" << GetType()->GetName() << "](Name : " << name_;
  ss << " Value : " << (value_ == nullptr ? "Null" : "%" + std::to_string(value_->GetNodeId()))
     << " Default Value : " << (default_value_ == nullptr ? "Null" : "%" + std::to_string(default_value_->GetNodeId()))
     << ")";
  return ss.str();
}

void FunctionNode::SetNodeId(size_t *id) {
  for (const auto &parameter : parameters_) {
    parameter->SetNodeId(id);
  }
  for (const auto &node : nodes_) {
    node->SetNodeId(id);
  }
  Node::SetNodeId(id);
}

void FunctionNode::SetOffset(size_t *offset) {
  /// Inputs must be valueNodes, no need to set offset
  /// Only the operation need to be set offset
  for (const auto &node : nodes_) {
    node->SetOffset(offset);
  }
}

bool FunctionNode::HasVarArg() const { return (flags_ & 0x0004) != 0x0; }

void FunctionNode::SetHasVarArg(bool has_var_arg) { flags_ = has_var_arg ? flags_ | 0x0004 : flags_ & 0xFFFB; }

bool FunctionNode::HasKwArg() const { return (flags_ & 0x0008) != 0x0; }

void FunctionNode::SetHasKwArg(bool has_kw_arg) { flags_ = has_kw_arg ? flags_ | 0x0008 : flags_ & 0xFFF7; }

void FunctionNode::AddNode(const NodePtr &node) {
  if (nodes_.empty() || !nodes_.back()->isa<ReturnNode>()) {
    nodes_.push_back(node);
  }
}

std::string FunctionNode::ToString() const {
  std::stringstream ss;
  for (const auto &parameter : parameters_) {
    ss << parameter->ToString() << "\n";
  }
  ss << "%" << std::to_string(GetNodeId()) << " = FunctionNode " << name_ << "(";
  for (const auto &parameter : parameters_) {
    ss << "%" << std::to_string(parameter->GetNodeId()) << ", ";
  }
  ss << ") {\n";
  for (const auto &node : nodes_) {
    ss << node->ToString() << "\n";
  }
  ss << "}\n";
  return ss.str();
}

void IfNode::SetNodeId(size_t *id) {
  condition_jump_->SetNodeId(id);
  for (const auto &node : then_) {
    node->SetNodeId(id);
  }
  for (const auto &node : else_) {
    node->SetNodeId(id);
  }
  Node::SetNodeId(id);
}

void IfNode::SetOffset(size_t *offset) {
  /// Only the operation need to be set offset
  condition_jump_->SetOffset(offset);
  for (const auto &node : then_) {
    node->SetOffset(offset);
  }
  for (const auto &node : else_) {
    node->SetOffset(offset);
  }
}

void IfNode::AddThen(const NodePtr &node) {
  if (then_.empty() || !then_.back()->isa<ReturnNode>()) {
    then_.push_back(node);
  }
}

void IfNode::AddElse(const NodePtr &node) {
  if (else_.empty() || !else_.back()->isa<ReturnNode>()) {
    else_.push_back(node);
  }
}

std::string IfNode::ToString() const {
  std::stringstream ss;
  ss << condition_jump_->ToString();
  ss << "%" << std::to_string(GetNodeId()) << " = If (%" << std::to_string(condition_jump_->GetNodeId()) << ") {\n";
  for (const auto &node : then_) {
    ss << node->ToString();
  }
  ss << "} else {\n";
  for (const auto &node : else_) {
    ss << node->ToString();
  }
  ss << "}\n";
  return ss.str();
}

void WhileNode::SetNodeId(size_t *id) {
  condition_jump_->SetNodeId(id);
  for (const auto &node : body_) {
    node->SetNodeId(id);
  }
  Node::SetNodeId(id);
}

void WhileNode::SetOffset(size_t *offset) {
  /// Only the operation need to be set offset
  condition_jump_->SetOffset(offset);
  for (const auto &node : body_) {
    node->SetOffset(offset);
  }
}

std::string WhileNode::ToString() const {
  std::stringstream ss;
  ss << condition_jump_->ToString();
  ss << "%" << std::to_string(GetNodeId()) << " = While (%" << std::to_string(condition_jump_->GetNodeId()) << ") {";
  for (const auto &node : body_) {
    ss << node->ToString();
  }
  ss << "}\n";
  return ss.str();
}
}  // namespace ir
}  // namespace pijit
}  // namespace mindspore
