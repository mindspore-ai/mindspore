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

#include "ops/ops_func_impl/unstack_ext.h"

#include <vector>
#include <memory>
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/shape_utils.h"
#include "utils/log_adapter.h"
#include "ir/primitive.h"
#include "abstract/dshape.h"
#include "base/base.h"

namespace mindspore::ops {
namespace {
size_t CheckShapeAndGetNormalizedAxisValue(const PrimitivePtr &primitive, const ShapeVector &input_shape,
                                           const AbstractBasePtr &axis_abstract) {
  if (MS_UNLIKELY(IsDynamic(input_shape))) {
    MS_EXCEPTION(ValueError) << "For Primitive[" << primitive->name()
                             << "], input shape should not be dynamic in this phase, but got " << input_shape;
  }

  size_t input_rank = input_shape.size();
  MS_CHECK_VALUE(input_rank > 0,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("input rank", input_rank, kGreaterThan, 0, primitive));
  auto axis_value = axis_abstract->GetValue();
  auto axis_res = GetScalarValue<int64_t>(axis_value);
  if (MS_UNLIKELY(!axis_res.has_value())) {
    MS_EXCEPTION(ValueError) << "For Primitive[" << primitive->name()
                             << "], should has valid axis value for this phase!";
  }
  auto axis_temp = axis_res.value();
  MS_CHECK_VALUE(-SizeToLong(input_rank) <= axis_temp && axis_temp < SizeToLong(input_rank),
                 CheckAndConvertUtils::FormatCheckInRangeMsg(
                   "axis", axis_temp, kIncludeLeft, {-SizeToLong(input_rank), SizeToLong(input_rank)}, primitive));
  return LongToSize(axis_temp < 0 ? SizeToLong(input_rank) + axis_temp : axis_temp);
}
}  // namespace
BaseShapePtr UnstackExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  auto axis = CheckShapeAndGetNormalizedAxisValue(primitive, input_shape, input_args[kInputIndex1]);

  auto input_rank = input_shape.size();
  ShapeVector element_shape;
  element_shape.reserve(input_rank - 1);
  for (size_t i = 0; i < input_rank; ++i) {
    if (MS_UNLIKELY(i == axis)) {
      continue;
    }
    element_shape.push_back(input_shape[i]);
  }

  auto element_size = LongToSize(input_shape[axis]);
  abstract::BaseShapePtrList out_shapes;
  out_shapes.reserve(element_size);
  for (size_t i = 0; i < element_size; ++i) {
    out_shapes.push_back(std::make_shared<abstract::TensorShape>(element_shape));
  }

  return std::make_shared<abstract::TupleShape>(out_shapes);
}

TypePtr UnstackExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  auto axis = CheckShapeAndGetNormalizedAxisValue(primitive, input_shape, input_args[kInputIndex1]);
  auto element_size = LongToSize(input_shape[axis]);

  auto input_type = input_args[kInputIndex0]->GetType();
  TypePtrList types;
  types.reserve(element_size);
  for (size_t i = 0; i < element_size; ++i) {
    types.push_back(input_type->Clone());
  }
  return std::make_shared<Tuple>(types);
}
}  // namespace mindspore::ops
