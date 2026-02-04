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

#include "infer/ops_func_impl/reduce_arithmetic.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "primitive/op_name.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "ops_utils/op_constants.h"

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
  if (input_args.size() < kReduceInputAtLeastLen) {
    MS_LOG(EXCEPTION) << "For " << primitive->name() << ", the input length should be at least "
                      << kReduceInputAtLeastLen << " but got " << input_args.size();
  }
  for (size_t i = 0; i < kReduceInputAtLeastLen; ++i) {
    MS_EXCEPTION_IF_NULL(input_args[i]);
  }

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

ShapeArray NormInferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) {
  auto keepdim_opt = input_infos[kInputIndex3]->GetScalarValue<bool>();
  if (MS_UNLIKELY(!keepdim_opt.has_value())) {
    return {ShapeVector({abstract::Shape::kShapeRankAny})};
  }
  auto keepdim = keepdim_opt.value();
  auto x_shape = input_infos[kInputIndex0]->GetShape();

  // If dim is None
  if (input_infos[kInputIndex2]->IsNone()) {
    return {keepdim ? IsDynamicRank(x_shape) ? x_shape : ShapeVector(x_shape.size(), 1) : ShapeVector({})};
  }

  if (IsDynamicRank(x_shape)) {
    return {x_shape};
  }

  auto dim_opt = input_infos[kInputIndex2]->GetArrayValue<int64_t>();
  if (dim_opt.has_value()) {
    // If dim is empty tuple and keepdim is False, return a zero-dimensional Tensor
    if (dim_opt->size() == 0 && !keepdim) {
      return {ShapeVector({})};
    }
  }
  if (!dim_opt.has_value()) {
    // dim is dynamic.
    return {keepdim ? ShapeVector(x_shape.size(), -1) : ShapeVector({abstract::Shape::kShapeRankAny})};
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
    auto out_shape = ReduceFuncCalShapeInferImpl(primitive, x_shape, real_dim_vec, keepdim);
    return {out_shape};
  }

  // If the dim has unknown value, the reduction position will be any of the input dimensions.
  if (!keepdim) {
    MS_CHECK_VALUE(x_shape.size() >= dim_opt->size(),
                   CheckAndConvertUtils::FormatCheckInRangeMsg("dim size", dim_opt->size(), kIncludeLeft,
                                                               {0, x_shape.size()}, primitive));
    return {ShapeVector(x_shape.size() - dim_opt->size(), -1)};
  }
  auto out_shape = ShapeVector(x_shape.size(), -1);
  for (size_t i = 0; i < dim.size(); ++i) {
    if (!dim.IsValueUnknown(i)) {
      auto axis = CalRealAixs(dim[i], x_shape_size, primitive);
      out_shape[axis] = 1;
    }
  }
  return {out_shape};
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

  const auto &x_tensor = input_values[kIndex0]->cast<tensor::TensorPtr>();
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

ShapeArray ReduceInferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) {
  auto keep_dims_opt = input_infos[kIndex2]->GetScalarValue<bool>();
  if (MS_UNLIKELY(!keep_dims_opt.has_value())) {
    return {ShapeVector({abstract::Shape::kShapeRankAny})};
  }
  bool keep_dims = keep_dims_opt.value();

  auto axis_array_opt = input_infos[kIndex1]->GetArrayValue<int64_t>();
  if (axis_array_opt.has_value() && axis_array_opt->size() == kIndex0 && !keep_dims) {
    // If axis is empty tuple and keep_dims is False, return a zero-dimensional Tensor
    return {ShapeVector({})};
  }

  const auto &x_shape = input_infos[kIndex0]->GetShape();
  if (MS_UNLIKELY(input_infos[kIndex0]->IsDynamicRank())) {
    return {x_shape};
  }
  if (MS_UNLIKELY(!axis_array_opt.has_value())) {
    auto out_shape = keep_dims ? ShapeVector(x_shape.size(), abstract::Shape::kShapeDimAny)
                               : ShapeVector({abstract::Shape::kShapeRankAny});
    return {out_shape};
  }

  auto x_shape_size = x_shape.size();
  const auto &axis_array = axis_array_opt.value();
  // All values of the axis are known.
  if (MS_LIKELY(!axis_array.HasUnknownValue())) {
    const auto &axis_vec = axis_array.ToVector();
    std::vector<int64_t> real_axis_vec;
    (void)std::transform(
      axis_vec.begin(), axis_vec.end(), std::back_inserter(real_axis_vec),
      [&x_shape_size, &primitive](const int64_t &axis) { return CalRealAixs(axis, x_shape_size, primitive); });
    auto out_shape = ReduceFuncCalShapeInferImpl(primitive, x_shape, real_axis_vec, keep_dims);
    return {out_shape};
  }

  // If the axis has unknown value, the reduction position will be any of the input dimensions.
  if (!keep_dims) {
    MS_CHECK_VALUE(x_shape_size >= axis_array.size(),
                   CheckAndConvertUtils::FormatCheckInRangeMsg("axis size", axis_array.size(), kIncludeLeft,
                                                               {0, x_shape_size}, primitive));
    return {ShapeVector(x_shape_size - axis_array.size(), abstract::Shape::kShapeDimAny)};
  }

  auto out_shape = ShapeVector(x_shape.size(), abstract::Shape::kShapeDimAny);
  for (size_t i = 0; i < axis_array.size(); ++i) {
    if (!axis_array.IsValueUnknown(i)) {
      auto axis = CalRealAixs(axis_array[i], x_shape_size, primitive);
      out_shape[axis] = 1;
    }
  }
  return {out_shape};
}

ShapeArray ReduceExtandSimpleInferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) {
  const auto &input = input_values[kIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input);
  const auto &input_shape = input->shape();
  const auto input_shape_size = input_shape.size();
  const auto &keep_dims = input_values[kIndex2]->cast<BoolImmPtr>();
  MS_EXCEPTION_IF_NULL(keep_dims);

  if (input_values[kIndex1] == mindspore::kNone) {
    return keep_dims->value() ? ShapeArray{ShapeVector(input_shape_size, 1)} : ShapeArray{ShapeVector({})};
  }

  const auto &axis = input_values[kIndex1]->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(axis);

  std::vector<int64_t> axis_vector;
  for (const auto &value : axis->value()) {
    const auto &axis_element = value->cast<Int64ImmPtr>();
    MS_EXCEPTION_IF_NULL(axis_element);
    axis_vector.emplace_back(axis_element->value());
  }

  if (axis_vector.empty()) {
    return keep_dims->value() ? ShapeArray{ShapeVector(input_shape_size, 1)} : ShapeArray{ShapeVector({})};
  }

  std::vector<int64_t> real_axis_vector;
  (void)std::transform(
    axis_vector.begin(), axis_vector.end(), std::back_inserter(real_axis_vector),
    [&input_shape_size, &primitive](const int64_t &axis) { return CalRealAixs(axis, input_shape_size, primitive); });
  auto out_shape = ReduceFuncCalShapeInferImpl(primitive, input_shape, real_axis_vector, keep_dims->value());
  return {out_shape};
}

ShapeArray ReduceGeneralInferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) {
  MS_LOG(DEBUG) << "Run ReduceGeneralInferShape" << primitive->name() << " start";
  const auto &input = input_infos[kInputIndex0];
  const auto input_shape = input->GetShape();
  const auto input_shape_size = input_shape.size();

  const auto keep_dims_opt = input_infos[kInputIndex2]->GetScalarValue<bool>();
  if (MS_UNLIKELY(!keep_dims_opt.has_value())) {
    return ShapeArray{ShapeVector({abstract::Shape::kShapeRankAny})};
  }
  const auto keep_dims = keep_dims_opt.value();

  const auto &axis = input_infos[kInputIndex1];
  // If axis is None
  if (axis->IsNone()) {
    return keep_dims ? ShapeArray{ShapeVector(input_shape_size, 1)} : ShapeArray{ShapeVector({})};
  }

  const auto &axis_opt = axis->GetArrayValue<int64_t>();
  const auto axis_size = axis_opt->size();
  if (axis_opt.has_value()) {
    // If axis is empty tuple and keep_dims is False, return a zero-dimensional Tensor
    if (axis_size == 0 && !keep_dims) {
      return ShapeArray{ShapeVector({})};
    }
  }

  if (input->IsDynamicRank()) {
    return {input_shape};
  }
  if (!axis_opt.has_value()) {
    // If axis is dynamic.
    return keep_dims ? ShapeArray{ShapeVector(input_shape_size, -1)}
                     : ShapeArray{ShapeVector({abstract::Shape::kShapeRankAny})};
  }

  const auto axis_array = axis_opt.value();
  // All values of the axis are known.
  if (!axis_array.HasUnknownValue()) {
    std::vector<int64_t> axis_vector = axis_array.ToVector();
    std::vector<int64_t> real_axis_vector;
    (void)std::transform(
      axis_vector.begin(), axis_vector.end(), std::back_inserter(real_axis_vector),
      [&input_shape_size, &primitive](const int64_t &axis) { return CalRealAixs(axis, input_shape_size, primitive); });
    auto out_shape = ReduceFuncCalShapeInferImpl(primitive, input_shape, real_axis_vector, keep_dims);
    return {out_shape};
  }

  // If the axis has unknown value, the reduction position will be any of the input dimensions.
  if (!keep_dims) {
    MS_CHECK_VALUE(input_shape_size >= axis_size,
                   CheckAndConvertUtils::FormatCheckInRangeMsg("axis size", axis_size, kIncludeLeft,
                                                               {0, input_shape_size}, primitive));
    return ShapeArray{ShapeVector(input_shape_size - axis_size, -1)};
  }
  auto out_shape = ShapeVector(input_shape_size, -1);
  for (size_t i = 0; i < axis_array.size(); ++i) {
    if (!axis_array.IsValueUnknown(i)) {
      auto axis_i = CalRealAixs(axis_array[i], input_shape_size, primitive);
      out_shape[axis_i] = 1;
    }
  }
  MS_LOG(DEBUG) << "Run ReduceGeneralInferShape" << primitive->name() << " end";
  return {out_shape};
}

ShapeArray ReduceGeneralInferShapeV2(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) {
  MS_LOG(DEBUG) << "Run ReduceExtandGeneralInferShapeV2_" << primitive->name() << " start";
  const auto &input = input_infos[kInputIndex0];
  const auto input_shape = input->GetShape();
  const auto input_shape_size = input_shape.size();

  const auto keepdim_opt = input_infos[kInputIndex3]->GetScalarValue<bool>();
  if (MS_UNLIKELY(!keepdim_opt.has_value())) {
    return ShapeArray{ShapeVector({abstract::Shape::kShapeRankAny}), ShapeVector({abstract::Shape::kShapeRankAny})};
  }
  const auto keepdim = keepdim_opt.value();

  const auto &dim = input_infos[kInputIndex1];
  // If dim is None
  if (dim->IsNone()) {
    return keepdim ? ShapeArray{ShapeVector(input_shape_size, 1), ShapeVector(input_shape_size, 1)}
                   : ShapeArray{ShapeVector({}), ShapeVector({})};
  }

  const auto &dim_opt = dim->GetArrayValue<int64_t>();
  const auto dim_size = dim_opt->size();
  if (dim_opt.has_value()) {
    // If dim is empty tuple and keepdim is False, return a zero-dimensional Tensor
    if (dim_size == 0 && !keepdim) {
      return ShapeArray{ShapeVector({}), ShapeVector({})};
    }
  }

  if (input->IsDynamicRank()) {
    return {input_shape, input_shape};
  }
  if (!dim_opt.has_value()) {
    // If dim is dynamic.
    return keepdim
             ? ShapeArray{ShapeVector(input_shape_size, -1), ShapeVector(input_shape_size, -1)}
             : ShapeArray{ShapeVector({abstract::Shape::kShapeRankAny}), ShapeVector({abstract::Shape::kShapeRankAny})};
  }

  const auto dim_array = dim_opt.value();
  // All values of the dim are known.
  if (!dim_array.HasUnknownValue()) {
    std::vector<int64_t> dim_vector = dim_array.ToVector();
    std::vector<int64_t> real_dim_vector;
    (void)std::transform(
      dim_vector.begin(), dim_vector.end(), std::back_inserter(real_dim_vector),
      [&input_shape_size, &primitive](const int64_t &dim) { return CalRealAixs(dim, input_shape_size, primitive); });
    auto out_shape = ReduceFuncCalShapeInferImpl(primitive, input_shape, real_dim_vector, keepdim);
    return {out_shape, out_shape};
  }

  // If the dim has unknown value, the reduction position will be any of the input dimensions.
  if (!keepdim) {
    MS_CHECK_VALUE(input_shape_size >= dim_size,
                   CheckAndConvertUtils::FormatCheckInRangeMsg("dim size", dim_size, kIncludeLeft,
                                                               {0, input_shape_size}, primitive));
    return ShapeArray{ShapeVector(input_shape_size - dim_size, -1), ShapeVector(input_shape_size - dim_size, -1)};
  }
  auto out_shape = ShapeVector(input_shape_size, -1);
  for (size_t i = 0; i < dim_array.size(); ++i) {
    if (!dim_array.IsValueUnknown(i)) {
      auto dim_i = CalRealAixs(dim_array[i], input_shape_size, primitive);
      out_shape[dim_i] = 1;
    }
  }
  MS_LOG(DEBUG) << "Run ReduceExtandGeneralInferShapeV2_" << primitive->name() << " end";
  return {out_shape, out_shape};
}
}  // namespace ops
}  // namespace mindspore
