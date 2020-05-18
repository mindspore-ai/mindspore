/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
  return mindspore::label_manage::Label(const_cast<AnfNode *>(this)->shared_from_base<AnfNode>()->debug_info());
}

OperatorInfoPtr CNode::set_operator_info(const OperatorInfoPtr &operator_info) {
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

void CNode::accept(AnfVisitor *v) { v->Visit(shared_from_base<CNode>()); }
void ValueNode::accept(AnfVisitor *v) { v->Visit(shared_from_base<ValueNode>()); }
void Parameter::accept(AnfVisitor *v) { v->Visit(shared_from_base<Parameter>()); }

}  // namespace mindspore
