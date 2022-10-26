/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include <sstream>

#include "utils/hash_map.h"
#include "ir/visitor.h"
#include "ir/func_graph.h"
#include "utils/anf_utils.h"
#include "mindspore/core/ops/core_ops.h"

namespace mindspore {
// namespace to support intermediate representation definition
// Methods of AnfNode
TypePtr AnfNode::Type() const {
  // cppcheck-suppress unreadVariable
  auto lock = AnfUtils::GetAbstractLock(this);
  return (abstract_ == nullptr) ? nullptr : abstract_->BuildType();
}
BaseShapePtr AnfNode::Shape() const {
  // cppcheck-suppress unreadVariable
  auto lock = AnfUtils::GetAbstractLock(this);
  return (abstract_ == nullptr) ? nullptr : abstract_->BuildShape();
}

std::string AnfNode::ToString() const {
  return mindspore::label_manage::Label(const_cast<AnfNode *>(this)->shared_from_base<AnfNode>()->debug_info());
}

std::string CNode::fullname_with_scope() {
  // if full name is set, return its name immediately
  if (!fullname_with_scope_.empty()) {
    return fullname_with_scope_;
  }

#ifndef ENABLE_SECURITY
  if (IsApply(prim::kPrimScalarSummary) || IsApply(prim::kPrimTensorSummary) || IsApply(prim::kPrimImageSummary) ||
      IsApply(prim::kPrimHistogramSummary)) {
    std::string tag = GetValue<std::string>(GetValueNode(input(1)));
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
    return fullname_with_scope_;
  }
#endif
  // cnode input 0 should be primitive ptr or funcgraph ptr
  auto value_ptr = input(0)->cast<ValueNodePtr>();
  if (value_ptr == nullptr) {
    MS_LOG(DEBUG) << "Input 0 of cnode is not a value node, its type is " << input(0)->type_name() << ".";
    fullname_with_scope_ = id_generator::get_id(shared_from_base<CNode>());
    return fullname_with_scope_;
  }
  auto input_value = value_ptr->value();
  if (input_value == nullptr) {
    MS_LOG(WARNING) << "Value of input 0 of cnode is nullptr.";
    fullname_with_scope_ = id_generator::get_id(shared_from_base<CNode>());
    return fullname_with_scope_;
  }

  MS_EXCEPTION_IF_NULL(scope());
  fullname_with_scope_ = scope()->name() + "/";
  if (input_value->isa<Primitive>()) {
    auto prim = input_value->cast_ptr<Primitive>();
    fullname_with_scope_ += prim->name();
  } else if (input_value->isa<FuncGraph>()) {
    auto func_graph = input_value->cast_ptr<FuncGraph>();
    auto fg_flag = func_graph->get_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL);
    if (fg_flag != nullptr) {
      auto fg_name = GetValue<std::string>(fg_flag);
      fullname_with_scope_ += "GraphKernel_" + fg_name;
    } else {
      fullname_with_scope_ += func_graph->ToString();
    }
  } else {
    // For the node after parse, the value maybe ClassType or others.
    fullname_with_scope_ += input_value->ToString();
  }
  fullname_with_scope_ += "-op" + id_generator::get_id(shared_from_base<CNode>());
  return fullname_with_scope_;
}

void CNode::accept(AnfIrVisitor *v) { v->Visit(shared_from_base<CNode>()); }
void ValueNode::accept(AnfIrVisitor *v) { v->Visit(shared_from_base<ValueNode>()); }
void Parameter::accept(AnfIrVisitor *v) { v->Visit(shared_from_base<Parameter>()); }
}  // namespace mindspore
