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
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
int64_t CalRealAixs(const int64_t &axis, const size_t &x_shape_size, const PrimitivePtr &primitive) {
  auto size = SizeToLong(x_shape_size);
  MS_CHECK_VALUE(axis >= -1 * size && axis < size, CheckAndConvertUtils::FormatCheckInRangeMsg(
                                                     "axis value", axis, kIncludeLeft, {-1 * size, size}, primitive));
  auto real_axis = axis < 0 ? axis + size : axis;
  return real_axis;
}

BaseShapePtr ReduceInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(input_args[0]);
  MS_EXCEPTION_IF_NULL(input_args[0]->GetShape());
  auto x_shape = input_args[0]->GetShape()->GetShapeVector();
  MS_EXCEPTION_IF_NULL(input_args[1]);
  auto axis_value = input_args[1]->GetValue();
  MS_EXCEPTION_IF_NULL(axis_value);
  auto axis_array_opt = GetArrayValue<int64_t>(axis_value);
  MS_EXCEPTION_IF_NULL(input_args[2]);
  auto keep_dims_value = input_args[2]->GetValue();
  MS_EXCEPTION_IF_NULL(keep_dims_value);
  if (keep_dims_value->isa<ValueAny>()) {
    // The keep_dims is unknown.
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }
  MS_CHECK_VALUE(keep_dims_value->isa<BoolImm>(), "The keep_dims input for " + primitive->name() + " must be bool.");
  bool keep_dims = GetValue<bool>(keep_dims_value);

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
}  // namespace ops
}  // namespace mindspore
