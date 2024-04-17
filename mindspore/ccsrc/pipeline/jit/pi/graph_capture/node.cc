/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include <string>
#include "pipeline/jit/pi/graph_capture/node.h"
#include "pipeline/jit/pi/graph_capture/cfg.h"
#include "pipeline/jit/pi/graph_capture/graph.h"

namespace mindspore {
namespace pijit {

static AbstractObjectBase kNullObject(AObject::kTypeAnyValue);
ValueNode ValueNode::kUnboundLocal(ValueNode::kUnbound, &kNullObject, 0, 0);

// these value node not in locals
bool IsNonLocalValue(ValueNode *i) {
  int op = i->GetOpcode();
  return op == LOAD_CONST || op == LOAD_GLOBAL || op == LOAD_DEREF || i->GetType() == ValueNode::CellVar ||
         i->GetType() == ValueNode::FreeVar;
}

void ValueNode::store_attr(const std::string &nam, ValueNode *v) { attr_ = true; }

void ValueNode::store_subscr(ValueNode *sub, ValueNode *v) { subscr_ = true; }

AObject *ValueNode::get_attr(const std::string &nam) {
  if (!attr_ && vobj_) {
    return vobj_->GetAttr(nam);
  }
  return AObject::MakeAObject(AObject::kTypeAnyValue);
}

AObject *ValueNode::binary_subscr(ValueNode *sub) {
  if (!subscr_ && vobj_) {
    return vobj_->GetItem(sub->GetVobj());
  }
  return AObject::MakeAObject(AObject::kTypeAnyValue);
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
  s << GetOparg() << ":" << GetVobj()->ToString() << '<' << this << '>';
  return s.str();
}

std::string CellVarNode::ToString() const {
  if (val_) {
    return std::string("Cell:").append(val_->ToString());
  }
  char buf[64];
  snprintf(buf, sizeof(buf), "Cell:%p->(nil)", this);
  return buf;
}

std::string CallNode::ToString() const {
  std::stringstream s;
  s << this->ValueNode::ToString() << "sub-graph " << sub_graph_;
  return s.str();
}

std::string ValueNode::ToString() const {
  if (this == &ValueNode::kUnboundLocal) {
    return "(kUnboundLocal)";
  }
  std::stringstream s;
  s << this->InstrNode::ToString();
  s << " vobj:{" << vobj_ << ":" << (vobj_ ? vobj_->ToString() : "(nil)") << "}";
  for (auto i : inputs_) {
    s << i << ',';
  }
  if (constant_info_ != nullptr) {
    s << " constant: " << constant_info_->ToString();
  }
  return s.str();
}

std::string InstrNode::ToString() const {
  std::stringstream s;
  s << this->AbstractNode::ToString() << " bci " << bci() << " lno " << GetLineNo() << ' '
    << Utils::GetOpName(GetOpcode()) << ' ' << GetOparg() << ' ' << GetName();
  return s.str();
}

std::string AbstractNode::ToString() const {
  std::stringstream s;
  s << this << " type " << type_ << " graph " << graph_;
  return s.str();
}

void ValueNode::SetParent(ValueNode *parent) { parent_ = std::make_optional<ValueNode *>(parent); }

void CallNode::SetSubGraph(Graph *n) {
  sub_graph_ = n;
  if (n) {
    n->SetParent(GetGraph());
  }
}
}  // namespace pijit
}  // namespace mindspore
