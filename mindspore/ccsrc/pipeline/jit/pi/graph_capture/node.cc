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
  return op == LOAD_CONST || op == LOAD_GLOBAL || i->GetType() == ValueNode::CellVar ||
         i->GetType() == ValueNode::FreeVar;
}

void AbstractNodeList::erase(AbstractNode *tar) {
  if (!tar) {
    return;
  }
  if (tar == head_ && tar == back_) {
    head_ = nullptr;
    back_ = nullptr;
  } else if (tar == head_) {
    head_ = tar->GetNext();
    head_->GetPres().clear();
  } else if (tar == back_) {
    back_ = tar->GetPres()[0];
    back_->SetNext(nullptr);
  } else {
    auto p = tar->GetPres()[0];
    auto n = tar->GetNext();
    p->SetNext(n);
    n->GetPres()[0] = p;
  }
  tar->GetPres().clear();
  tar->SetNext(nullptr);
}

void AbstractNodeList::push_back(AbstractNode *n) {
  if (back_) {
    back_->insertBack(n);
  } else {
    head_ = n;
  }
  back_ = n;
}

void AbstractNodeList::push_front(AbstractNode *n) {
  if (head_) {
    n->insertBack(head_);
  } else {
    back_ = n;
  }
  head_ = n;
}

bool AbstractNodeList::insert(AbstractNode *pos, AbstractNodeList *l) {
  if (!l->head() || !l->back()) {
    return false;
  }
  if (!head_ || !back_) {
    head_ = l->head();
    back_ = l->back();
    return true;
  }
  if (l->head()->GetPres().size() || l->back()->GetNext()) {
    MS_LOG(EXCEPTION) << "bad list insert";
    return false;
  }
  if (!pos) {
    back_->insertBack(l->head());
    back_ = l->back();
    return true;
  }
  auto pre = pos->GetPres().size() ? pos->GetPres()[0] : nullptr;
  if (pre) {
    pre->SetNext(l->head());
    l->head()->AddPre(pre);
    pos->GetPres()[0] = l->back();
  } else {
    head_ = l->head();
    pos->AddPre(l->back());
  }
  l->back()->SetNext(pos);
  return true;
}

void AbstractNodeList::cutList(AbstractNode *pos, AbstractNodeList *second_half) {
  if (!pos || pos == head_ || head_ == back_) {
    MS_LOG(EXCEPTION) << "bad list cut";
    return;
  }
  *second_half = {pos, back_};
  back_ = pos->GetPres()[0];
  second_half->head()->GetPres().clear();
  back_->SetNext(nullptr);
}

// return ture if object is mindspore support
bool ValueNode::IsMindsporeSupportedOperation() {
  if (vobj_ && vobj_->IsMindSporeSupportedType()) {
    return true;
  }
  return false;
}

void ValueNode::store_attr(const std::string &nam, ValueNode *v) {
  vobj_->SetAttr(nam, v->vobj_);
  attrs_.insert({nam, v});
}

void ValueNode::store_subscr(ValueNode *sub, ValueNode *v) {
  if (vobj_) {
    vobj_->SetItem(sub->vobj_, v->vobj_);
  }
}

void MergeNode::AddInput(ValueNode *node) {
  MS_ASSERT(predecessor_index_.size() == getInputs().size());
  int i = node->GetOutputs().size();
  node->AddOutput(this);
  this->ValueNode::AddInput(node);
  predecessor_index_.push_back(i);
}

void MergeNode::Merge(MergeNode *o) {
  for (size_t i = 0; i < o->getInputs().size(); ++i) {
    ValueNode *node = o->input(i);
    MS_ASSERT(node->GetOutput(o->predecessor_index_[i]) == o);
    predecessor_index_.push_back(o->predecessor_index_[i]);
    node->SetOutput(o->predecessor_index_[i], this);
  }
  o->getInputs().clear();
  o->predecessor_index_.clear();
}

std::string MergeNode::to_str() const {
  std::stringstream s;
  s << " merge:{";
  for (size_t i = 0; i < getInputs().size(); ++i) {
    auto b = input(i)->GetBlock();
    s << input(i) << ":[b:" << (b ? b->id() : -1) << ']';
  }
  s << "} vobj " << (GetVobj() ? GetVobj()->ToString() : "(nil)");
  return s.str();
}

std::string ParamNode::to_str() const {
  std::stringstream s;
  s << GetOparg() << ":" << GetVobj()->ToString() << '<' << this << '>';
  return s.str();
}

std::string CellVarNode::to_str() const {
  if (val_) {
    return std::string("Cell:").append(val_->to_str());
  }
  char buf[64];
  snprintf(buf, sizeof(buf), "Cell:%p->(nil)", this);
  return buf;
}

std::string CallNode::to_str() const {
  std::stringstream s;
  s << this->ValueNode::to_str() << "sub-graph " << sub_graph_;
  return s.str();
}

std::string ValueNode::to_str() const {
  std::stringstream s;
  s << this->InstrNode::to_str();
  s << " vobj:{" << vobj_ << ":" << (vobj_ ? vobj_->ToString() : "(nil)") << "}";
  if (inputs_.empty()) {
    return s.str();
  }
  for (auto i : inputs_) {
    s << i << ',';
  }
  return s.str();
}

std::string InstrNode::to_str() const {
  std::stringstream s;
  s << this->AbstractNode::to_str() << " bci " << bci() << " lno " << GetLineNo() << ' '
    << Utils::GetOpName(GetOpcode()) << ' ' << GetOparg() << ' ' << GetName();
  return s.str();
}

std::string AbstractNode::to_str() const {
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
