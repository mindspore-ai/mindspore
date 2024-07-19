/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "ops/ops_func_impl/index_select.h"
#include <memory>
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr IndexSelectFuncImpl::InferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
  if (MS_UNLIKELY(IsDynamicRank(input_shape))) {
    return std::make_shared<abstract::TensorShape>(ShapeVector({abstract::TensorShape::kShapeRankAny}));
  }

  int64_t input_rank = SizeToLong(input_shape.size());
  auto axis_opt = GetScalarValue<int64_t>(input_args[kIndex1]->GetValue());
  if (MS_UNLIKELY(!axis_opt.has_value())) {
    return std::make_shared<abstract::TensorShape>(ShapeVector(input_rank, abstract::TensorShape::kShapeDimAny));
  }
  if (MS_UNLIKELY(axis_opt.value() >= input_rank || axis_opt.value() < -input_rank)) {
    MS_EXCEPTION(ValueError) << "For 'IndexSelect', the axis must be in '[" << -input_rank << ", " << input_rank
                             << ")', but got " << axis_opt.value() << ".";
  }
  auto axis = axis_opt.value() < 0 ? axis_opt.value() + input_rank : axis_opt.value();

  auto output_shape = input_shape;
  auto index_shape = input_args[kIndex2]->GetShape()->GetShapeVector();
  if (MS_LIKELY(!IsDynamicRank(index_shape))) {
    MS_CHECK_VALUE(index_shape.size() == 1, "For 'IndexSelect', the dimension of 'index' must be 1, but got " +
                                              std::to_string(index_shape.size()) + ".");
  }

  if (MS_UNLIKELY(IsDynamic(index_shape))) {
    output_shape[axis] = abstract::TensorShape::kShapeDimAny;
  } else {
    output_shape[axis] = index_shape[kIndex0];
  }

  return std::make_shared<abstract::TensorShape>(output_shape);
}

TypePtr IndexSelectFuncImpl::InferType(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetType();
}
}  // namespace ops
}  // namespace mindspore
