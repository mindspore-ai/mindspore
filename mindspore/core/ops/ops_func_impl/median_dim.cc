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
#include "ops/ops_func_impl/median_dim.h"
#include <set>
#include <memory>
#include "ir/dtype.h"
#include "ops/op_name.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "ops/ops_func_impl/reduce_arithmetic.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
inline BaseShapePtr GetOutputShape(const ShapeVector &output_shape) {
  return std::make_shared<abstract::TupleShape>(abstract::BaseShapePtrList{
    std::make_shared<abstract::TensorShape>(output_shape), std::make_shared<abstract::TensorShape>(output_shape)});
}

inline ShapeArray GetOutputShapeArray(const ShapeVector &output_shape) {
  ShapeArray shape_array{output_shape, output_shape};
  return shape_array;
}
}  // namespace

BaseShapePtr MedianDimFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto x_shape = x_shape_ptr->GetShapeVector();
  // 1. judging whether the input is dynamic rank
  if (MS_UNLIKELY(IsDynamicRank(x_shape))) {
    return GetOutputShape(x_shape);
  }
  auto x_rank = SizeToLong(x_shape.size());

  // 2. judging whether the keepdim is variable
  auto keep_dims_value = input_args[kInputIndex2]->GetValue();
  auto keep_dims_opt = GetScalarValue<bool>(keep_dims_value);
  if (MS_UNLIKELY(!keep_dims_opt.has_value())) {
    ShapeVector dynamic_rank_shape{abstract::TensorShape::kShapeRankAny};
    return GetOutputShape(dynamic_rank_shape);
  }
  // testing empty tensor
  for (size_t i = 0; i < x_shape.size(); i++) {
    if (x_shape[i] == 0) {
      MS_EXCEPTION(ValueError) << primitive->name() << " cannot deal with empty input. Please try other inputs";
    }
  }
  auto keep_dims = keep_dims_opt.value();
  auto out_dim = keep_dims ? x_rank : x_rank - 1;

  // 3. judging whether the axis is variable
  auto axis_value = input_args[kInputIndex1]->GetValue();
  auto axis_value_opt = GetScalarValue<int64_t>(axis_value);
  if (MS_UNLIKELY(!axis_value_opt.has_value())) {
    if (x_rank == 0) {
      out_dim = 0;
    }
    return GetOutputShape(ShapeVector(out_dim, abstract::TensorShape::kShapeDimAny));
  }
  auto axis = axis_value_opt.value();
  if (MS_UNLIKELY(x_rank == 0)) {
    if (axis != -1 && axis != 0) {
      MS_EXCEPTION(ValueError) << "For " << primitive->name()
                               << " with 0d input tensor, dim must be one of 0 or -1, but got " << axis << ".";
    }
    return std::make_shared<abstract::TupleShape>(
      abstract::BaseShapePtrList{x_shape_ptr->Clone(), x_shape_ptr->Clone()});
  }
  MS_CHECK_VALUE(axis >= -x_rank && axis < x_rank,
                 CheckAndConvertUtils::FormatCheckInRangeMsg("dim", axis, kIncludeLeft, {-x_rank, x_rank}, primitive));
  axis = axis < 0 ? axis + x_rank : axis;

  ShapeVector output_shape(x_shape);
  if (keep_dims) {
    output_shape[axis] = 1;
  } else {
    (void)output_shape.erase(output_shape.begin() + axis);
  }
  return GetOutputShape(output_shape);
}

TypePtr MedianDimFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kIndex0]);
  TypePtr input_x_type = input_args[kIndex0]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_x_type, common_valid_types_with_bool, primitive->name());
  return std::make_shared<Tuple>(TypePtrList{input_args[0]->GetType(), std::make_shared<TensorType>(kInt64)});
}

ShapeArray MedianDimFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto x_shape = x_tensor->shape();

  const auto &keep_dims_value = input_values[kInputIndex2];
  auto keep_dims_opt = GetScalarValue<bool>(keep_dims_value);
  if (std::any_of(x_shape.begin(), x_shape.end(), [](const auto &item) { return item == 0; })) {
    MS_EXCEPTION(ValueError) << primitive->name() << " cannot deal with empty input. Please try other inputs";
  }

  auto x_rank = SizeToLong(x_shape.size());
  auto keep_dims = keep_dims_opt.value();

  const auto &axis_value = input_values[kInputIndex1];
  auto axis_value_opt = GetScalarValue<int64_t>(axis_value);
  auto axis = axis_value_opt.value();
  if (MS_UNLIKELY(x_rank == 0)) {
    if (axis != -1 && axis != 0) {
      MS_EXCEPTION(ValueError) << "For " << primitive->name()
                               << " with 0d input tensor, dim must be one of 0 or -1, but got " << axis << ".";
    }
    return GetOutputShapeArray(x_shape);
  }
  MS_CHECK_VALUE(axis >= -x_rank && axis < x_rank,
                 CheckAndConvertUtils::FormatCheckInRangeMsg("dim", axis, kIncludeLeft, {-x_rank, x_rank}, primitive));
  axis = axis < 0 ? axis + x_rank : axis;
  if (MS_UNLIKELY(x_shape[axis] == 0)) {
    MS_EXCEPTION(ValueError) << "For " << primitive->name() << ", the pos:" << axis
                             << " of input_x's shape can not be 0, but got " << x_shape;
  }

  ShapeVector output_shape(x_shape);
  if (keep_dims) {
    output_shape[axis] = 1;
  } else {
    (void)output_shape.erase(output_shape.begin() + axis);
  }
  return GetOutputShapeArray(output_shape);
}

TypePtrList MedianDimFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  return {x_tensor->Dtype(), kInt64};
}

REGISTER_SIMPLE_INFER(kNameMedianDim, MedianDimFuncImpl)
}  // namespace ops
}  // namespace mindspore
