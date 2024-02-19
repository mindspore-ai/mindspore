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

#include "ops/ops_func_impl/argmax_ext.h"
#include <utility>
#include <memory>
#include "ops/op_utils.h"
#include "ir/dtype.h"
#include "ops/op_name.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr ArgMaxExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape_vec = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  if (IsDynamicRank(x_shape_vec)) {
    ShapeVector out_shape{abstract::TensorShape::kShapeRankAny};
    return std::make_shared<abstract::TensorShape>(std::move(out_shape));
  }

  ShapeVector output_shape;
  if (!input_args[kInputIndex1]->GetType()->isa<TypeNone>()) {
    auto axis_value_scalar = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue());
    if (axis_value_scalar.has_value()) {
      int64_t axis = 0;
      axis = axis_value_scalar.value();
      auto x_rank = SizeToLong(x_shape_vec.size());
      MS_CHECK_VALUE(axis >= -x_rank && axis < x_rank, CheckAndConvertUtils::FormatCheckInRangeMsg(
                                                         "dim", axis, kIncludeLeft, {-x_rank, x_rank}, primitive));
      axis = axis < 0 ? axis + x_rank : axis;
      output_shape = x_shape_vec;
      auto keepdim = GetScalarValue<bool>(input_args[kInputIndex2]->GetValue());
      auto keepdim_value = keepdim.value();
      if (keepdim_value) {
        output_shape[axis] = 1;
      } else {
        output_shape.erase(output_shape.cbegin() + axis);
      }
    } else {
      auto output_rank = x_shape_vec.size() - 1;
      output_shape.assign(output_rank, abstract::Shape::kShapeDimAny);
    }
  } else {
    auto output_rank = x_shape_vec.size() - 1;
    output_shape.assign(output_rank, abstract::Shape::kShapeDimAny);
  }
  return std::make_shared<abstract::TensorShape>(output_shape);
}

TypePtr ArgMaxExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  auto type = kInt64;
  return std::make_shared<TensorType>(type);
}
}  // namespace ops
}  // namespace mindspore
