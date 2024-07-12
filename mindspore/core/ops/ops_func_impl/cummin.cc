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

#include "ops/ops_func_impl/cummin.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
BaseShapePtr CumminFuncImpl::InferShape(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape = input_args[kIndex0]->GetShape();
  auto x_shape_vec = x_shape->GetShapeVector();
  if (IsDynamicRank(x_shape_vec)) {
    auto dyn_output = std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{dyn_output, dyn_output});
  }
  auto rank = SizeToLong(x_shape_vec.size());
  (void)CheckAndConvertUtils::CheckInteger("rank of 'x'", rank, kGreaterThan, 0, primitive->name());
  auto axis = input_args[kIndex1]->GetValue();
  auto axis_opt = GetScalarValue<int64_t>(axis);
  if (axis_opt.has_value()) {
    auto axis_value = axis_opt.value();
    MS_CHECK_VALUE(
      axis_value >= -rank && axis_value < rank,
      CheckAndConvertUtils::FormatCheckInRangeMsg("axis", axis_value, kIncludeLeft, {-rank, rank}, primitive));
  }
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{
    std::make_shared<abstract::TensorShape>(x_shape_vec), std::make_shared<abstract::TensorShape>(x_shape_vec)});
}

TypePtr CumminFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto x_type = input_args[kInputIndex0]->GetType();
  return std::make_shared<Tuple>(std::vector{x_type, kInt32});
}

TypePtrList CumminFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &input_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  return {input_tensor->Dtype(), kInt32};
}

ShapeArray CumminFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  auto &x_shape = x_tensor->shape();
  bool is_dynamic = IsDynamic(x_shape);
  if (!is_dynamic) {
    if (input_values[kInputIndex1] != mindspore::kNone) {
      const auto &axis_scalar = GetScalarValue<int64_t>(input_values[kInputIndex1]);
      int64_t axis_value = axis_scalar.value();
      auto rank = SizeToLong(x_shape.size());
      MS_CHECK_VALUE(
        axis_value >= -rank && axis_value < rank,
        CheckAndConvertUtils::FormatCheckInRangeMsg("axis", axis_value, kIncludeLeft, {-rank, rank}, primitive));
    }
  }
  return {x_shape, x_shape};
}

REGISTER_SIMPLE_INFER(kNameCummin, CumminFuncImpl)
}  // namespace ops
}  // namespace mindspore
