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

#include "ops/ops_func_impl/reduce_arithmetic.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
int64_t CalRealAixs(const int64_t &axis, const size_t &x_shape_size, const PrimitivePtr &primitive) {
  auto size = SizeToLong(x_shape_size);
  size = size == 0 ? 1 : size;  // if x_shape_size is 0, the data is scaler.
  MS_CHECK_VALUE(axis >= -1 * size && axis < size, CheckAndConvertUtils::FormatCheckInRangeMsg(
                                                     "axis value", axis, kIncludeLeft, {-1 * size, size}, primitive));
  auto real_axis = axis < 0 ? axis + size : axis;
  return real_axis;
}

BaseShapePtr ReduceInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto keep_dims_value = input_args[kInputIndex2]->GetValue();
  auto keep_dims_opt = GetScalarValue<bool>(keep_dims_value);
  if (MS_UNLIKELY(!keep_dims_opt.has_value())) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }
  auto keep_dims = keep_dims_opt.value();

  auto axis_array_opt = GetArrayValue<int64_t>(input_args[kInputIndex1]);
  if (axis_array_opt.has_value()) {
    // If axis is empty tuple and keep_dims is False, return a zero-dimensional Tensor
    if (axis_array_opt->size() == 0 && !keep_dims) {
      return std::make_shared<abstract::Shape>(ShapeVector({}));
    }
  }

  auto x_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(x_shape);
  }
  if (!axis_array_opt.has_value()) {
    // axis is dynamic.
    return keep_dims ? std::make_shared<abstract::Shape>(ShapeVector(x_shape.size(), -1))
                     : std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  auto x_shape_size = x_shape.size();
  auto axis_array = axis_array_opt.value();
  // All values of the axis are known.
  if (!axis_array.HasUnknownValue()) {
    std::vector<int64_t> axis_vec = axis_array.ToVector();
    std::vector<int64_t> real_axis_vec;
    (void)std::transform(
      axis_vec.begin(), axis_vec.end(), std::back_inserter(real_axis_vec),
      [&x_shape_size, &primitive](const int64_t &axis) { return CalRealAixs(axis, x_shape_size, primitive); });
    auto out_shape = ReduceFuncCalShapeInferImpl(primitive, x_shape, real_axis_vec, keep_dims);
    return std::make_shared<abstract::Shape>(out_shape);
  }

  // If the axis has unknown value, the reduction position will be any of the input dimensions.
  if (!keep_dims) {
    MS_CHECK_VALUE(x_shape.size() >= axis_array_opt->size(),
                   CheckAndConvertUtils::FormatCheckInRangeMsg("axis size", axis_array_opt->size(), kIncludeLeft,
                                                               {0, x_shape.size()}, primitive));
    return std::make_shared<abstract::Shape>(ShapeVector(x_shape.size() - axis_array_opt->size(), -1));
  }
  auto out_shape = ShapeVector(x_shape.size(), -1);
  for (size_t i = 0; i < axis_array.size(); ++i) {
    if (!axis_array.IsValueUnknown(i)) {
      auto axis = CalRealAixs(axis_array[i], x_shape_size, primitive);
      out_shape[axis] = 1;
    }
  }
  return std::make_shared<abstract::Shape>(out_shape);
}

BaseShapePtr NormInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto keep_dims_value = input_args[kInputIndex3]->GetValue();
  auto keep_dims_opt = GetScalarValue<bool>(keep_dims_value);
  if (MS_UNLIKELY(!keep_dims_opt.has_value())) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }
  auto keep_dims = keep_dims_opt.value();
  auto x_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();

  // If dim is None
  if (input_args[kInputIndex2]->GetType()->isa<TypeNone>()) {
    return keep_dims
             ? std::make_shared<abstract::Shape>(IsDynamicRank(x_shape) ? x_shape : ShapeVector(x_shape.size(), 1))
             : std::make_shared<abstract::Shape>(ShapeVector({}));
  }

  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(x_shape);
  }
  auto dim_opt = GetArrayValue<int64_t>(input_args[kInputIndex2]);
  if (dim_opt.has_value()) {
    // If axis is empty tuple and keep_dims is False, return a zero-dimensional Tensor
    if (dim_opt->size() == 0 && !keep_dims) {
      return std::make_shared<abstract::Shape>(ShapeVector({}));
    }
  }
  if (!dim_opt.has_value()) {
    // dim is dynamic.
    return keep_dims ? std::make_shared<abstract::Shape>(ShapeVector(x_shape.size(), -1))
                     : std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }
  auto x_shape_size = x_shape.size();
  auto dim = dim_opt.value();
  // All values of the dim are known.
  if (!dim.HasUnknownValue()) {
    std::vector<int64_t> dim_vec = dim.ToVector();
    std::vector<int64_t> real_dim_vec;
    (void)std::transform(
      dim_vec.begin(), dim_vec.end(), std::back_inserter(real_dim_vec),
      [&x_shape_size, &primitive](const int64_t &axis) { return CalRealAixs(axis, x_shape_size, primitive); });
    auto out_shape = ReduceFuncCalShapeInferImpl(primitive, x_shape, real_dim_vec, keep_dims);
    return std::make_shared<abstract::Shape>(out_shape);
  }

  // If the dim has unknown value, the reduction position will be any of the input dimensions.
  if (!keep_dims) {
    MS_CHECK_VALUE(x_shape.size() >= dim_opt->size(),
                   CheckAndConvertUtils::FormatCheckInRangeMsg("axis size", dim_opt->size(), kIncludeLeft,
                                                               {0, x_shape.size()}, primitive));
    return std::make_shared<abstract::Shape>(ShapeVector(x_shape.size() - dim_opt->size(), -1));
  }
  auto out_shape = ShapeVector(x_shape.size(), -1);
  for (size_t i = 0; i < dim.size(); ++i) {
    if (!dim.IsValueUnknown(i)) {
      auto axis = CalRealAixs(dim[i], x_shape_size, primitive);
      out_shape[axis] = 1;
    }
  }
  return std::make_shared<abstract::Shape>(out_shape);
}

BaseShapePtr ReduceExtandInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto keep_dims_value = input_args[kInputIndex2]->GetValue();
  auto keep_dims_opt = GetScalarValue<bool>(keep_dims_value);
  if (MS_UNLIKELY(!keep_dims_opt.has_value())) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }
  auto keep_dims = keep_dims_opt.value();
  auto x_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();

  // If axis is None
  if (input_args[kInputIndex1]->GetType()->isa<TypeNone>()) {
    return keep_dims
             ? std::make_shared<abstract::Shape>(IsDynamicRank(x_shape) ? x_shape : ShapeVector(x_shape.size(), 1))
             : std::make_shared<abstract::Shape>(ShapeVector({}));
  }

  auto axis_array_opt = GetArrayValue<int64_t>(input_args[kInputIndex1]);
  if (axis_array_opt.has_value()) {
    // If axis is empty tuple and keep_dims is False, return a zero-dimensional Tensor
    if (axis_array_opt->size() == 0 && !keep_dims) {
      return std::make_shared<abstract::Shape>(ShapeVector({}));
    }
  }

  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(x_shape);
  }
  if (!axis_array_opt.has_value()) {
    // axis is dynamic.
    return keep_dims ? std::make_shared<abstract::Shape>(ShapeVector(x_shape.size(), -1))
                     : std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  auto x_shape_size = x_shape.size();
  auto axis_array = axis_array_opt.value();
  // All values of the axis are known.
  if (!axis_array.HasUnknownValue()) {
    std::vector<int64_t> axis_vec = axis_array.ToVector();
    std::vector<int64_t> real_axis_vec;
    (void)std::transform(
      axis_vec.begin(), axis_vec.end(), std::back_inserter(real_axis_vec),
      [&x_shape_size, &primitive](const int64_t &axis) { return CalRealAixs(axis, x_shape_size, primitive); });
    auto out_shape = ReduceFuncCalShapeInferImpl(primitive, x_shape, real_axis_vec, keep_dims);
    return std::make_shared<abstract::Shape>(out_shape);
  }

  // If the axis has unknown value, the reduction position will be any of the input dimensions.
  if (!keep_dims) {
    MS_CHECK_VALUE(x_shape.size() >= axis_array_opt->size(),
                   CheckAndConvertUtils::FormatCheckInRangeMsg("axis size", axis_array_opt->size(), kIncludeLeft,
                                                               {0, x_shape.size()}, primitive));
    return std::make_shared<abstract::Shape>(ShapeVector(x_shape.size() - axis_array_opt->size(), -1));
  }
  auto out_shape = ShapeVector(x_shape.size(), -1);
  for (size_t i = 0; i < axis_array.size(); ++i) {
    if (!axis_array.IsValueUnknown(i)) {
      auto axis = CalRealAixs(axis_array[i], x_shape_size, primitive);
      out_shape[axis] = 1;
    }
  }
  return std::make_shared<abstract::Shape>(out_shape);
}

ShapeArray ReduceInferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) {
  const auto &keep_dims_opt = input_values[kIndex2]->cast<BoolImmPtr>();
  MS_EXCEPTION_IF_NULL(keep_dims_opt);
  const bool &keep_dims = keep_dims_opt->value();

  const auto &x_tensor = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto &x_shape = x_tensor->shape();

  const auto &axis_value = input_values[kIndex1];

  if (axis_value == mindspore::kNone) {
    return keep_dims ? ShapeArray{ShapeVector(x_shape.size(), 1)} : ShapeArray{ShapeVector({})};
  }

  const auto &axis_opt = axis_value->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(axis_opt);

  std::vector<int64_t> axis_vec;
  const auto &axis_items = axis_opt->value();
  for (const auto &axis_item : axis_items) {
    (void)axis_vec.emplace_back(GetValue<int64_t>(axis_item));
  }

  if (axis_vec.size() == 0) {
    return keep_dims ? ShapeArray{ShapeVector(x_shape.size(), 1)} : ShapeArray{ShapeVector({})};
  }

  std::vector<int64_t> real_axis_vec;
  const auto &x_shape_size = x_shape.size();
  (void)std::transform(
    axis_vec.begin(), axis_vec.end(), std::back_inserter(real_axis_vec),
    [&x_shape_size, &primitive](const int64_t &axis) { return CalRealAixs(axis, x_shape_size, primitive); });

  return {ReduceFuncCalShapeInferImpl(primitive, x_shape, real_axis_vec, keep_dims)};
}

}  // namespace ops
}  // namespace mindspore
