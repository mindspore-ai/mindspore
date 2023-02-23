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

#include "common/expander/core/node.h"
#include <algorithm>
#include "common/expander/core/emitter.h"
#include "common/expander/core/infer.h"

namespace mindspore {
namespace expander {
Node::Node(const AnfNodePtr &node, const Emitter *emitter) : anf_node_(node), emitter_(emitter) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(emitter);
}

AbstractBasePtr Node::abstract() { return emitter()->infer()->GetAbstract(shared_from_this()); }

std::vector<int64_t> Node::shape() {
  if (shape_ == nullptr) {
    shape_ = emitter()->infer()->GetShape(shared_from_this());
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
    shape_ = emitter()->infer()->GetShape(shared_from_this());
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
    type_ = emitter()->infer()->GetDtype(shared_from_this());
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
    type_ = emitter()->infer()->GetDtype(shared_from_this());
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
}  // namespace expander
}  // namespace mindspore
