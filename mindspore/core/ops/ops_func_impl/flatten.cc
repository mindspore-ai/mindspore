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

#include "ops/ops_func_impl/flatten.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "mindapi/ir/value.h"
#include "mindapi/ir/primitive.h"

namespace mindspore {
namespace ops {
BaseShapePtr FlattenFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  const auto &input_x_shape = input_args[kIndex0]->GetShape();
  if (input_x_shape->IsDimZero()) {
    MS_LOG(EXCEPTION) << "Unsupported input shape dimension. The shape should not be empty.";
  }

  auto x_shape = input_x_shape->GetShapeVector();
  if (IsDynamicRank(x_shape)) {
    ShapeVector out_shape = {abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny};
    return std::make_shared<abstract::Shape>(out_shape);
  }

  int64_t prod = 1;
  size_t size = x_shape.size();
  for (size_t i = 1; i < size; i++) {
    if (x_shape[i] == -1) {
      prod = -1;
      break;
    }
    prod = prod * x_shape[i];
  }
  ShapeVector out_shape = {x_shape[0], prod};
  return std::make_shared<abstract::TensorShape>(out_shape);
}

TypePtr FlattenFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetType()->Clone();
}
}  // namespace ops
}  // namespace mindspore
