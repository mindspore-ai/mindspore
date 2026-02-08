/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#include "mindspore/core/abstract/abstract_join.h"

#include "ir/dtype/tensor_type.h"
#include "utils/ms_context.h"
#include "utils/symbolic.h"
#include "abstract/abstract_function.h"

namespace mindspore {
namespace abstract {
namespace {
bool IsShapesDynamicRank(const std::vector<ShapeVector> &shapes) {
  return std::any_of(shapes.begin(), shapes.end(), [](const ShapeVector &shape) {
    return std::any_of(shape.begin(), shape.end(), [](int64_t dim) { return dim == Shape::kShapeRankAny; });
  });
}

ShapePtr SingleElementShapeJoin(const ShapePtr &shape1, const ShapePtr &shape2) {
  // special case: shape(1), shape() -> shape(1)
  if (shape1->shape().size() == 1 && shape1->shape()[0] == 1 && shape2->shape().empty()) {
    return shape1;
  }
  if (shape2->shape().size() == 1 && shape2->shape()[0] == 1 && shape1->shape().empty()) {
    return shape2;
  }
  return nullptr;
}

ShapeValueDType SingleShapeValueJoin(const ShapeValueDType &shape_value1, const ShapeValueDType &shape_value2) {
  if (shape_value1 == shape_value2) {
    return shape_value1;
  }
  return Shape::kShapeDimAny;
}
}  // namespace

ShapePtr ShapeJoin(const ShapePtr &shape1, const ShapePtr &shape2) {
  MS_EXCEPTION_IF_NULL(shape1);
  MS_EXCEPTION_IF_NULL(shape2);
  if (*shape1 == *shape2) {
    return shape1;
  }

  bool has_dynamic_rank = IsShapesDynamicRank({shape1->shape(), shape2->shape()});
  if (has_dynamic_rank) {
    return std::make_shared<Shape>(ShapeVector{Shape::kShapeRankAny});
  }
  // lengths of two shapes are not same, join failed
  if (shape1->shape().size() != shape2->shape().size()) {
    auto joined_shape = SingleElementShapeJoin(shape1, shape2);
    if (joined_shape != nullptr) {
      return joined_shape;
    }
    return std::make_shared<Shape>(ShapeVector({Shape::kShapeRankAny}));
  }
  ShapeVector dims(shape1->shape().size());
  for (std::size_t i = 0; i < shape1->shape().size(); i++) {
    auto joined_shape_value = SingleShapeValueJoin(shape1->shape()[i], shape2->shape()[i]);
    if (joined_shape_value == Shape::kShapeError) {
      return nullptr;
    }
    dims[i] = joined_shape_value;
  }
  return std::make_shared<Shape>(dims);
}

ValuePtr ValueJoin(const ValuePtr &value1, const ValuePtr &value2) {
  MS_EXCEPTION_IF_NULL(value1);
  MS_EXCEPTION_IF_NULL(value2);
  if (*value1 == *value2) {
    return value1;
  }
  return kValueAny;
}
}  // namespace abstract
}  // namespace mindspore
