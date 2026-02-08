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
#include "frontend/jit/pi/graph_compiler/pi_ir/operation.h"
#include <sstream>

namespace mindspore {
namespace pijit {
namespace ir {
void Operation::SetNodeId(size_t *id) {
  for (const auto &arg : args_) {
    arg->SetNodeId(id);
  }
  Node::SetNodeId(id);
}

void Operation::SetOffset(size_t *offset) {
  for (const auto &arg : args_) {
    arg->SetOffset(offset);
  }
  Node::SetOffset(offset);
}

std::string UnaryOperation::ToString() const {
  return GetArg()->ToString() + "\n%" + std::to_string(GetNodeId()) + " = " + GetNodeName() + "[" +
         GetType()->GetName() + "](" + GetOpName(GetOpCode()) + ", %" + std::to_string(GetArg()->GetNodeId()) + ")\n";
}

std::string BinaryOperation::ToString() const {
  return GetArg(0)->ToString() + "\n" + GetArg(1)->ToString() + "\n%" + std::to_string(GetNodeId()) + " = " +
         GetNodeName() + "[" + GetType()->GetName() + "](" + GetOpName(GetOpCode()) + ", %" +
         std::to_string(GetArg(0)->GetNodeId()) + ", %" + std::to_string(GetArg(1)->GetNodeId()) + ")\n";
}

std::string NaryOperation::ToString() const {
  std::stringstream ss;
  for (const auto &arg : GetArgs()) {
    ss << arg->ToString() << "\n";
  }
  ss << "%" << std::to_string(GetNodeId()) << " = " << GetNodeName() << "[" << GetType()->GetName() << "]("
     << GetOpName(GetOpCode());
  for (const auto &arg : GetArgs()) {
    ss << ", %" << std::to_string(arg->GetNodeId());
  }
  ss << ")\n";
  return ss.str();
}

std::string GetOpName(OpCode op) {
  static const std::vector<std::string> op_names =
    py::cast<std::vector<std::string>>(py::module::import("opcode").attr("opname"));
  return op_names[op];
}
}  // namespace ir
}  // namespace pijit
}  // namespace mindspore
