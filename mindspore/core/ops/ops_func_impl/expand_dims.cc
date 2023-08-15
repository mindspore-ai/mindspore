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

#include "ops/ops_func_impl/expand_dims.h"
#include <utility>
#include <memory>
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr ExpandDimsFuncImpl::InferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  auto input_x_shape_vec = input_args[kIndex0]->GetShape()->GetShapeVector();
  if (IsDynamicRank(input_x_shape_vec)) {
    ShapeVector dyn_output{abstract::Shape::kShapeRankAny};
    return std::make_shared<abstract::Shape>(std::move(dyn_output));
  }

  auto input_x_rank = input_x_shape_vec.size();
  auto output_rank = input_x_rank + 1;
  auto axis_value = input_args[kIndex1]->GetValue();
  auto axis_value_scalar = GetScalarValue<int64_t>(axis_value);

  ShapeVector expand_dims_shape;
  if (!axis_value_scalar.has_value()) {
    expand_dims_shape.resize(output_rank, abstract::Shape::kShapeDimAny);
  } else {
    auto axis = axis_value_scalar.value();
    axis = axis < 0 ? axis + SizeToLong(output_rank) : axis;
    MS_CHECK_VALUE(axis >= 0, primitive->name() + "error: axis value invalid.");
    expand_dims_shape.assign(input_x_shape_vec.begin(), input_x_shape_vec.end());
    (void)expand_dims_shape.insert(expand_dims_shape.begin() + axis, 1);
  }

  return std::make_shared<abstract::TensorShape>(expand_dims_shape);
}

TypePtr ExpandDimsFuncImpl::InferType(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetType()->Clone();
}
}  // namespace ops
}  // namespace mindspore
