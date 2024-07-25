/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "include/common/expander/core/node.h"
#include "include/common/expander/core/emitter.h"
#include "include/common/expander/core/infer.h"

namespace mindspore {
namespace expander {
Node::Node(Emitter *emitter) : emitter_(emitter) { MS_EXCEPTION_IF_NULL(emitter); }

InputType Node::input_type() { MS_EXCEPTION(NotImplementedError) << "Base node not implement input_type() method"; }

AbstractBasePtr Node::abstract() { MS_EXCEPTION(NotImplementedError) << "Base node not implement abstract() method"; }

bool Node::HasAbstractValue() { MS_EXCEPTION(NotImplementedError) << "Base node not implement abstract() method"; }

BaseShapePtr Node::GetShape() { MS_EXCEPTION(NotImplementedError) << "Base node not implement GetShape() method"; }

TypePtr Node::GetType() { MS_EXCEPTION(NotImplementedError) << "Base node not implement GetType() method"; }

ValuePtr Node::BuildValue() { MS_EXCEPTION(NotImplementedError) << "Base node not implement BuildValue() method"; }

std::vector<int64_t> Node::shape() {
  if (shape_ == nullptr) {
    shape_ = GetShape();
    MS_EXCEPTION_IF_NULL(shape_);
  }
  if (shape_->isa<abstract::NoShape>()) {
    return {};
  }
  auto shape = shape_->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape);
  return shape->shape();
}

std::vector<std::vector<int64_t>> Node::shapes() {
  if (shape_ == nullptr) {
    shape_ = GetShape();
    MS_EXCEPTION_IF_NULL(shape_);
  }
  auto tuple_shape = shape_->cast<abstract::SequenceShapePtr>();
  MS_EXCEPTION_IF_NULL(tuple_shape);
  auto &shape_list = tuple_shape->shape();
  std::vector<ShapeVector> shapes(shape_list.size());
  (void)std::transform(shape_list.cbegin(), shape_list.cend(), shapes.begin(), [](const BaseShapePtr &bs) {
    if (bs->isa<abstract::NoShape>()) {
      return ShapeVector();
    }
    auto shape = bs->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(shape);
    return shape->shape();
  });
  return shapes;
}

TypePtr Node::dtype() {
  if (type_ == nullptr) {
    type_ = GetType();
    MS_EXCEPTION_IF_NULL(type_);
    if (type_->isa<TensorType>()) {
      type_ = type_->cast<TensorTypePtr>()->element();
      MS_EXCEPTION_IF_NULL(type_);
    }
  }
  return type_;
}

std::vector<TypePtr> Node::dtypes() {
  if (type_ == nullptr) {
    type_ = GetType();
    MS_EXCEPTION_IF_NULL(type_);
  }
  auto tuple = type_->cast<TuplePtr>();
  MS_EXCEPTION_IF_NULL(tuple);
  std::vector<TypePtr> result(tuple->size());
  auto elements = tuple->elements();
  (void)std::transform(elements.cbegin(), elements.cend(), result.begin(),
                       [](const TypePtr &t) { return t->isa<TensorType>() ? t->cast<TensorTypePtr>()->element() : t; });
  return result;
}

std::string Node::ToString() const { return value_ != nullptr ? value_->ToString() : ""; }

InputType IrNode::input_type() {
  if (anf_node_->isa<ValueNode>()) {
    return InputType::kConstant;
  } else if (anf_node_->isa<Parameter>()) {
    return InputType::kParameter;
  } else {
    return InputType::kOpOutput;
  }
}

AbstractBasePtr IrNode::abstract() { return emitter()->infer()->GetAbstract(shared_from_this()); }

ValuePtr IrNode::BuildValue() {
  is_used_value_ = true;
  if (value_ == nullptr) {
    if (anf_node_->isa<ValueNode>()) {
      return value_ = anf_node_->cast<ValueNodePtr>()->value();
    } else {
      return value_ = abstract()->BuildValue();
    }
  }
  return value_;
}

bool IrNode::HasAbstractValue() {
  ValuePtr value = abstract()->BuildValue();
  return (value != nullptr && !value->ContainsValueAny());
}

BaseShapePtr IrNode::GetShape() {
  auto shape = emitter()->infer()->GetShape(shared_from_this());
  MS_EXCEPTION_IF_NULL(shape);
  return shape;
}

TypePtr IrNode::GetType() {
  auto type = emitter()->infer()->GetDtype(shared_from_this());
  MS_EXCEPTION_IF_NULL(type);
  return type;
}

std::string IrNode::ToString() const {
  MS_EXCEPTION_IF_NULL(anf_node_);
  return anf_node_->ToString();
}

void IrNode::set_debug_info(const std::string &debug_info) {
  auto primitive = GetCNodePrimitive(anf_node_);
  primitive->set_instance_name(debug_info);
}

std::string IrNode::debug_info() const {
  auto primitive = GetCNodePrimitive(anf_node_);
  return primitive->instance_name();
}

ValuePtr FuncNode::BuildValue() { return value_; }
InputType FuncNode::input_type() { return input_type_; }

std::vector<int64_t> FuncNode::shape() {
  if (value_->isa<tensor::BaseTensor>()) {
    const auto &tensor = value_->cast<tensor::BaseTensorPtr>();
    return tensor->shape();
  }
  return Node::shape();
}

TypePtr FuncNode::dtype() {
  if (value_->isa<tensor::BaseTensor>()) {
    const auto &tensor = value_->cast<tensor::BaseTensorPtr>();
    return type_ = tensor->Dtype();
  }
  return Node::dtype();
}

AbstractBasePtr FuncNode::abstract() {
  if (abstract_ != nullptr) {
    return abstract_;
  }
  return abstract_ = value_->ToAbstract();
}

BaseShapePtr FuncNode::GetShape() {
  auto shape = abstract()->BuildShape();
  MS_EXCEPTION_IF_NULL(shape);
  return shape;
}

TypePtr FuncNode::GetType() {
  auto type = abstract()->BuildType();
  MS_EXCEPTION_IF_NULL(type);
  return type;
}
}  // namespace expander
}  // namespace mindspore
