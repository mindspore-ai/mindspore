/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "ir/anf.h"

#include <algorithm>
#include <sstream>
#include <vector>
#include <unordered_map>

#include "ir/visitor.h"
#include "pipeline/static_analysis/static_analysis.h"
#include "operator/ops.h"
#include "parallel/ops_info/ops_utils.h"

namespace mindspore {
// namespace to support intermediate representation definition
// Methods of AnfNode
TypePtr AnfNode::Type() const { return (abstract_ == nullptr) ? nullptr : abstract_->BuildType(); }
BaseShapePtr AnfNode::Shape() const { return (abstract_ == nullptr) ? nullptr : abstract_->BuildShape(); }

std::string AnfNode::ToString() const {
  return mindspore::label_manage::Label(const_cast<AnfNode*>(this)->shared_from_base<AnfNode>()->debug_info());
}

CNode::CNode(const std::vector<AnfNodePtr>& inputs, const FuncGraphPtr& func_graph)
    : AnfNode(func_graph), inputs_(inputs), stop_gradient_(false) {}

// Check if CNode is an apply with the specific Primitive.
bool CNode::IsApply(const PrimitivePtr& value) const {
  if (value == nullptr) {
    return false;
  }

  if (inputs_.size() != 0 && IsValueNode<Primitive>(inputs_[0])) {
    PrimitivePtr fn_value = GetValueNode<PrimitivePtr>(inputs_[0]);
    if (fn_value->Hash() == value->Hash() && fn_value->name() == value->name()) {
      return true;
    }
  }

  return false;
}

void CNode::set_input(size_t i, const AnfNodePtr& new_input) { inputs_[i] = new_input; }

std::string CNode::DebugString(int recursive_level) const {
  std::ostringstream buffer;
  if (recursive_level > 0) {
    if (func_graph() != nullptr) {
      buffer << func_graph()->ToString() << ":";
    }
    buffer << ToString() << "{";
    bool is_first_node = true;
    int idx = 0;
    for (auto& node : inputs_) {
      MS_EXCEPTION_IF_NULL(node);
      if (is_first_node) {
        is_first_node = false;
      } else {
        buffer << ", ";
      }
      buffer << "[" << idx << "]: " << node->DebugString(recursive_level - 1);
      idx++;
    }
    buffer << "}";
  } else {
    buffer << ToString();
  }
  return buffer.str();
}

OperatorInfoPtr CNode::set_operator_info(const OperatorInfoPtr& operator_info) {
  if (operator_info_ != nullptr) {
    MS_LOG(WARNING) << "The CNode: " << ToString() << " has already been set OperatorInfo: " << operator_info_->name()
                    << ", using the new one: " << operator_info->name();
    auto old_ptr = operator_info_;
    operator_info_ = operator_info;
    return old_ptr;
  }
  operator_info_ = operator_info;
  return nullptr;
}

std::string CNode::fullname_with_scope() {
  // if full name is set, return its name immediately
  if (!fullname_with_scope_.empty()) {
    return fullname_with_scope_;
  }

  if (IsApply(prim::kPrimScalarSummary) || IsApply(prim::kPrimTensorSummary) || IsApply(prim::kPrimImageSummary) ||
      IsApply(prim::kPrimHistogramSummary)) {
    std::string tag = GetValue<std::string>(GetValueNode(input(1)));
    if (tag == "") {
      MS_LOG(EXCEPTION) << "The tag name is null, should be valid string";
    }
    std::string name;
    if (IsApply(prim::kPrimScalarSummary)) {
      name = tag + "[:Scalar]";
    } else if (IsApply(prim::kPrimImageSummary)) {
      name = tag + "[:Image]";
    } else if (IsApply(prim::kPrimHistogramSummary)) {
      name = tag + "[:Histogram]";
    } else {
      name = tag + "[:Tensor]";
    }
    fullname_with_scope_ = name;
  } else {
    // cnode input 0 should be primitive ptr
    auto value_ptr = input(0)->cast<ValueNodePtr>();
    if (value_ptr == nullptr) {
      MS_LOG(WARNING) << "Input 0 of cnode is not a value node, its type is " << input(0)->type_name() << ".";
      fullname_with_scope_ = id_generator::get_id(shared_from_base<CNode>());
      return fullname_with_scope_;
    }
    auto input_value = value_ptr->value();
    if (input_value == nullptr) {
      MS_LOG(WARNING) << "Value of input 0 of cnode is nullptr.";
      fullname_with_scope_ = id_generator::get_id(shared_from_base<CNode>());
      return fullname_with_scope_;
    }

    PrimitivePtr prim = GetValue<PrimitivePtr>(input_value);
    MS_EXCEPTION_IF_NULL(scope());
    MS_EXCEPTION_IF_NULL(prim);
    fullname_with_scope_ =
      scope()->name() + "/" + prim->name() + "-op" + id_generator::get_id(shared_from_base<CNode>());
  }

  return fullname_with_scope_;
}

std::string ValueNode::ToString() const {
  MS_EXCEPTION_IF_NULL(value_);
  if (value_->isa<FuncGraph>()) {
    return value_->cast<FuncGraphPtr>()->ToString();
  }
  std::ostringstream buffer;
  buffer << AnfNode::ToString();
  buffer << "(" << value_->ToString() << ")";
  return buffer.str();
}

std::string ValueNode::DebugString(int) const {
  MS_EXCEPTION_IF_NULL(value_);
  std::ostringstream buffer;
  buffer << "ValueNode<" << value_->type_name() << "> " << value_->ToString();
  return buffer.str();
}

std::string ValueNode::fullname_with_scope() {
  if (!fullname_with_scope_.empty()) {
    return fullname_with_scope_;
  }

  MS_EXCEPTION_IF_NULL(scope());
  fullname_with_scope_ = scope()->name() + "/" + "data-" + id_generator::get_id(shared_from_base<ValueNode>());
  return fullname_with_scope_;
}

void CNode::accept(AnfVisitor* v) { v->Visit(shared_from_base<CNode>()); }
void ValueNode::accept(AnfVisitor* v) { v->Visit(shared_from_base<ValueNode>()); }
void Parameter::accept(AnfVisitor* v) { v->Visit(shared_from_base<Parameter>()); }

bool IsPrimitiveCNode(const AnfNodePtr& node, const PrimitivePtr& value) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  if (cnode != nullptr) {
    return cnode->IsApply(value);
  }
  return false;
}

PrimitivePtr GetCNodePrimitive(const AnfNodePtr& node) {
  if (node == nullptr) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  if (cnode != nullptr) {
    if (cnode->size() > 0) {
      auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
      return prim;
    }
  }
  return nullptr;
}

std::string GetCNodeFuncName(const CNodePtr cnode) {
  if (cnode->inputs().empty()) {
    return "";
  }

  AnfNodePtr valuenode = cnode->input(0);
  if (valuenode->isa<ValueNode>()) {
    auto value = GetValueNode(valuenode);
    // check whether the valuenode is primitive
    if (value->isa<Primitive>()) {
      return value->cast<PrimitivePtr>()->name();
    }
    return value->ToString();
  }
  return "";
}

bool IsPrimitive(const AnfNodePtr& node, const PrimitivePtr& value) {
  if (IsValueNode<Primitive>(node)) {
    PrimitivePtr fn_value = GetValueNode<PrimitivePtr>(node);
    MS_EXCEPTION_IF_NULL(value);
    if (fn_value->Hash() == value->Hash() && fn_value->name() == value->name()) {
      return true;
    }
  }
  return false;
}
namespace id_generator {
static std::unordered_map<std::string, int> node_ids;
std::string get_id(const AnfNodePtr& node) {
  auto type_name = node->type_name();
  if (node_ids.find(type_name) == node_ids.end()) {
    node_ids[type_name] = 0;
  } else {
    node_ids[type_name]++;
  }
  return std::to_string(node_ids[type_name]);
}

void reset_id() { node_ids.clear(); }
}  // namespace id_generator
}  // namespace mindspore
