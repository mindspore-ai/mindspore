/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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
#include "frontend/jit/pi/graph_capture/node.h"
#include <iterator>
#include <string>
#include <vector>
#include "frontend/jit/pi/graph_capture/abstract_object.h"
#include "frontend/jit/pi/graph_capture/cfg.h"
#include "frontend/jit/pi/graph_capture/graph.h"
#include "frontend/jit/pi/graph_guard/infer.h"

namespace mindspore {
namespace pijit {

static AbstractObjectBase kNullObject(AObject::kTypeAnyValue);
ValueNode ValueNode::kUnboundLocal(ValueNode::kUnbound, &kNullObject, 0, 0);
ValueNode ValueNode::kStackNull(ValueNode::kUnbound, &kNullObject, 0, 0);

// these value node not in locals
bool IsNonLocalValue(ValueNode *i) {
  int op = i->GetOpcode();
  return op == LOAD_CONST || op == LOAD_GLOBAL || op == LOAD_DEREF || i == &ValueNode::kUnboundLocal ||
         i == &ValueNode::kStackNull;
}

void ValueNode::SetVobj(AObject *object_info) {
  if (object_info == this->vobj_) {
    MS_LOG(INFO) << "Try to overwrite vobj with itself.";
    return;
  }
  if (this->vobj_ != nullptr && this->vobj_->GetType() != AObject::kTypeAnyValue && object_info != nullptr) {
    MS_LOG(INFO) << "Try to overwrite " << this->vobj_->ToString() << " with " << object_info->ToString() << " for "
                 << ToString();
  }
  auto replaced = this->vobj_;
  this->vobj_ = object_info;
  if (this->GetGraph() == nullptr) {
    return;
  }
  const auto &data = this->GetGraph()->GetSideEffect()->data();
  if (replaced != nullptr) {
    data->UnTrack(replaced->GetPyObject().ptr(), this);
  }
  if (object_info != nullptr) {
    data->Track(object_info->GetPyObject().ptr(), this);
  }
}

AObject *ValueNode::get_attr(const std::string &nam) {
  if (vobj_ == nullptr) {
    return AObject::MakeAObject(AObject::kTypeAnyValue);
  }
  return GetVobj()->GetAttr(nam);
}

bool ValueNode::IsConstantValue() const {
  return constant_info_ != nullptr && constant_info_->value().ptr() != nullptr;
}

void ValueNode::SetConstantValue(bool constant) {
  if (constant && this->GetVobj() != nullptr) {
    MakeConstantInfo()->set_value(this->GetVobj()->GetPyObject());
    return;
  }
  if (constant_info_ != nullptr) {
    constant_info_->set_value(py::object());
  }
}

const std::unique_ptr<ConstantInfo> &ValueNode::MakeConstantInfo() {
  if (constant_info_ == nullptr) {
    constant_info_ = std::make_unique<ConstantInfo>();
  }
  return constant_info_;
}

std::string ParamNode::ToString() const {
  std::stringstream s;
  s << this->AbstractNode::ToString() << " Parameter " << GetOparg() << "("
    << (GetName().empty() ? "<unnamed>" : GetName()) << ") = " << GetOwnVobj()->ToString();
  return s.str();
}

std::string CallNode::ToString() const {
  std::stringstream s;
  s << this->ValueNode::ToString()
    << (kw_names().ptr() != nullptr ? ("kw:" + std::string(py::str(kw_names().ptr()))) : std::string())
    << " sub-graph=" << sub_graph_;
  if (!inputs().empty()) {
    ValueNode *func_node = input(0);
    MS_EXCEPTION_IF_NULL(func_node);
    if (func_node->GetVobj() != nullptr && func_node->GetVobj()->GetPyObject().ptr() != nullptr) {
      s << ", function=" << py::str(func_node->GetVobj()->GetPyObject());
    }
  }
  return s.str();
}

std::vector<py::object> CallNode::GetArgs() {
  std::vector<py::object> args;
  std::transform(inputs().begin() + 1, inputs().end(), std::back_inserter(args),
                 [](ValueNode *n) { return n->GetVobj() ? n->GetVobj()->GetPyObject() : py::object(); });
  return args;
}

ValueNode *CallNode::GetSelf() const {
  auto method = input(0);
  auto func = method->GetOwnVobj()->GetPyObject().ptr();
  if (func != nullptr && !PyMethod_Check(func) && !(PyCFunction_Check(func) && PyCFunction_GET_SELF(func) != nullptr)) {
    return nullptr;
  }
  Opcode opcode(method->GetOpcode());
  if (opcode.IsCall() && opcode != CALL_FUNCTION_EX) {
    // method from the CALL_FUNCTION
    py::object tp = method->input(0)->GetOwnVobj() ? method->input(0)->GetOwnVobj()->GetPyObject() : py::object();
    if (tp.ptr() == reinterpret_cast<PyObject *>(&PyMethod_Type)) {
      return method->input(2);
    }
  }
  if (method->GetOpcode() != LOAD_ATTR) {
    return nullptr;
  }
  ValueNode *self = method->input(0);
  PyTypeObject *tp = self->GetVobj() ? self->GetVobj()->GetTypeObject() : nullptr;
  PyTypeObject *real_tp = method->GetVobj()->GetAttr("__self__")->GetTypeObject();
  if (tp == real_tp) {
    return self;
  }
  MS_LOG(DEBUG) << "Types of 'self' are different, " << (tp ? tp->tp_name : "NULL") << " vs "
                << (real_tp ? real_tp->tp_name : "NULL");
  // In case of this situation:
  // a = TypeA(); b = TypeB(); a.method = b.method
  // then tp of a.method is TypeA, but real_tp is TypeB.
  return nullptr;
}

void CallNode::UpdateVobj() {
  MS_EXCEPTION_IF_CHECK_FAIL(!inputs().empty(), "inputs should not be empty!");
  MS_EXCEPTION_IF_CHECK_FAIL(input(0) != nullptr && input(0)->GetOwnVobj() != nullptr, "input(0) has nullptr!");
  const auto &func = input(0)->GetOwnVobj()->GetPyObject();
  std::vector<AObject *> args;
  auto self = GetSelf();
  if (self != nullptr) {
    args.push_back(self->GetOwnVobj());
  }
  std::transform(inputs().begin() + 1, inputs().end(), std::back_inserter(args),
                 [](auto &input) { return input->GetOwnVobj(); });
  auto vobj = AObject::FuncAObjectUpdater(func, args);
  MS_EXCEPTION_IF_NULL(vobj);
  if (vobj->GetType() != AObject::kTypeAnyValue) {
    SetVobj(vobj);
  }
}

std::string ValueNode::ToString() const {
  if (this == &ValueNode::kUnboundLocal) {
    return "(kUnboundLocal)";
  }
  if (this == &ValueNode::kStackNull) {
    return "(kStackNull)";
  }
  std::stringstream s;
  int w = 30;
  s << std::setw(w) << std::left << this->InstrNode::ToString() << " ";
  s << this->AbstractNode::ToString();
  s << ((IsVmNode() && IsGraphNode()) ? " [V|G]" : (IsVmNode() ? " [V]" : " [G]"));
  s << "[" << GetScopeDesc() << "]";
  s << " vobj=" << (vobj_ ? vobj_->ToString() : "(nil)");
  s << " inputs=(";
  for (auto i : inputs_) {
    s << i << ',';
  }
  s << ") ";
  if (constant_info_ != nullptr) {
    s << " constant: " << constant_info_->ToString();
  }
  if (GetGraph() != nullptr) {
    s << " at \"" << pijit::GetFileName(GetGraph()) << ":" << GetLineNo() << "\"";
  }
  return s.str();
}

std::string InstrNode::ToString() const {
  std::stringstream s;
  const int width = 4;
  s << std::setw(width) << std::left;
  if (bci() >= 0) {
    s << bci();
  }
  s << ' ';
  constexpr auto w = 12;
  s << std::setw(w) << std::left << Opcode(GetOpcode()).name();
  if (Opcode(GetOpcode()).HasArg()) {
    s << ' ' << GetOparg() << ' ' << GetName();
  }
  return s.str();
}

std::string AbstractNode::ToString() const {
  std::stringstream s;
  s << "Node(" << this << ")";
  return s.str();
}

void CallNode::SetSubGraph(Graph *n) {
  sub_graph_ = n;
  if (n) {
    n->SetParent(GetGraph());
  }
}

CallNode::CallNode(int opcode, int oparg, const std::vector<ValueNode *> &inputs)
    : ValueNode(Call, nullptr, opcode, oparg, inputs), sub_graph_(nullptr) {
#if !IS_PYTHON_3_11_PLUS
  if (opcode != CALL_FUNCTION_KW) {
    return;
  }

  MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() > 0, "error stack status");
  kw_names_ = inputs.back()->GetVobj()->GetPyObject();  // must be tuple of str
  MS_EXCEPTION_IF_CHECK_FAIL(kw_names().ptr() != nullptr && PyTuple_CheckExact(kw_names().ptr()),
                             "key words must be tuple of str");
#endif
}

bool CallNode::IsCallKW() { return kw_names().ptr() != nullptr; }
bool CallNode::IsCallEX() { return GetOpcode() == CALL_FUNCTION_EX; }

std::string ToString(const pijit::AbstractNode *node) { return node == nullptr ? "NULL" : node->ToString(); }

}  // namespace pijit
}  // namespace mindspore
