/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "ir/anf_utils.h"

namespace mindspore {
bool AnfUtils::IsShapeDynamic(const abstract::ShapePtr &shape) {
  MS_EXCEPTION_IF_NULL(shape);
  return std::any_of(shape->shape().begin(), shape->shape().end(), [](int64_t s) { return s < 0; });
}

bool AnfUtils::IsShapeDynamic(const std::vector<size_t> &shape) {
  return std::any_of(shape.begin(), shape.end(), [](int64_t s) { return s < 0; });
}

bool AnfUtils::IsNodeOutputDynamicShape(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto base_shape = node->Shape();
  if (base_shape == nullptr) {
    MS_LOG(INFO) << "Invalid base shape, node: " << node->fullname_with_scope();
    return false;
  }
  if (base_shape->isa<abstract::Shape>() && IsShapeDynamic(base_shape->cast<abstract::ShapePtr>())) {
    return true;
  } else if (base_shape->isa<abstract::TupleShape>()) {
    auto tuple_shape = base_shape->cast<abstract::TupleShapePtr>();
    MS_EXCEPTION_IF_NULL(tuple_shape);
    for (size_t i = 0; i < tuple_shape->size(); i++) {
      auto b_shape = (*tuple_shape)[i];
      if (b_shape->isa<abstract::Shape>() && IsShapeDynamic(b_shape->cast<abstract::ShapePtr>())) {
        return true;
      }
    }
  }
  return false;
}
}  // namespace mindspore
