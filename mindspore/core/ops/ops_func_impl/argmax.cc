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

#include "ops/ops_func_impl/argmax.h"
#include <utility>
#include <memory>
#include "ops/op_utils.h"
#include "ir/dtype.h"
#include "ops/op_name.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr ArgmaxFuncImpl::InferShape(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape_vec = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  if (IsDynamicRank(x_shape_vec)) {
    ShapeVector out_shape{abstract::TensorShape::kShapeRankAny};
    return std::make_shared<abstract::TensorShape>(std::move(out_shape));
  }

  auto axis_value = input_args[kInputIndex1]->GetValue();
  auto axis_value_scalar = GetScalarValue<int64_t>(axis_value);

  ShapeVector output_shape;
  if (!axis_value_scalar.has_value()) {
    auto output_rank = x_shape_vec.size() - 1;
    output_shape.assign(output_rank, abstract::Shape::kShapeDimAny);
  } else {
    auto x_rank = SizeToLong(x_shape_vec.size());
    auto axis = axis_value_scalar.value();
    MS_CHECK_VALUE(axis >= -x_rank && axis < x_rank, CheckAndConvertUtils::FormatCheckInRangeMsg(
                                                       "axis", axis, kIncludeLeft, {-x_rank, x_rank}, primitive));
    axis = axis < 0 ? axis + x_rank : axis;
    output_shape = x_shape_vec;
    output_shape.erase(output_shape.cbegin() + axis);
  }
  return std::make_shared<abstract::TensorShape>(output_shape);
}

TypePtr ArgmaxFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto dtype_ptr = GetScalarValue<int64_t>(input_args[kInputIndex2]->GetValue());
  MS_CHECK_VALUE(dtype_ptr.has_value(), primitive->name() + " error: dtype input should has valid value.");
  auto type = TypeIdToType(static_cast<TypeId>(dtype_ptr.value()));
  MS_CHECK_VALUE(type == kInt32 || type == kInt64, primitive->name() + " error: dtype should be " + kInt32->ToString() +
                                                     " or " + kInt64->ToString() + " but got " + type->ToString());
  return std::make_shared<TensorType>(type);
}
}  // namespace ops
}  // namespace mindspore
