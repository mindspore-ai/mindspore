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
#include "pipeline/jit/pi/graph_capture/node.h"
#include <string>
#include "pipeline/jit/pi/graph_capture/cfg.h"

namespace mindspore {
namespace jit {
namespace graph {
ValueNode ValueNode::UnboundLocal(ValueNode::Unbound, nullptr, 0, 0);

// these value node not in locals
bool IsNonLocalValue(ValueNode *i) {
  int op = i->GetOpcode();
  return op == LOAD_CONST || op == LOAD_GLOBAL || op == LOAD_DEREF || i->GetType() == ValueNode::CellVar ||
         i->GetType() == ValueNode::FreeVar;
}

void ValueNode::store_attr(const std::string &nam, ValueNode *v) {
  vobj_->SetAttr(nam, v->vobj_);
  attrs_[nam] = v;
}

void ValueNode::store_subscr(ValueNode *sub, ValueNode *v) {
  if (vobj_) {
    vobj_->SetItem(sub->vobj_, v->vobj_);
  }
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
  std::stringstream s;
  s << this->InstrNode::ToString();
  s << " vobj:{" << vobj_ << ":" << (vobj_ ? vobj_->ToString() : "(nil)") << "}";
  if (inputs_.empty()) {
    return s.str();
  }
  for (auto i : inputs_) {
    s << i << ',';
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
  s << this << " type " << type_ << " graph " << owner_;
  if (jump_) {
    s << " jump " << jump_;
  }
  return s.str();
}
}  // namespace graph
}  // namespace jit
}  // namespace mindspore
