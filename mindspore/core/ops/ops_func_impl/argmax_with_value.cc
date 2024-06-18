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

#include "ops/ops_func_impl/argmax_with_value.h"
#include <memory>
#include "ops/op_utils.h"
#include "ir/dtype.h"
#include "ops/op_name.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kInputElementSize = 3;

inline BaseShapePtr GetOutputShape(const ShapeVector &output_shape) {
  return std::make_shared<abstract::TupleShape>(abstract::BaseShapePtrList{
    std::make_shared<abstract::TensorShape>(output_shape), std::make_shared<abstract::TensorShape>(output_shape)});
}

inline ShapeArray GetOutputShapeArray(const ShapeVector &output_shape) {
  ShapeArray shape_array{output_shape, output_shape};
  return shape_array;
}
}  // namespace

BaseShapePtr ArgMaxWithValueFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) const {
  if (input_args.size() < kInputElementSize) {
    MS_EXCEPTION(ValueError) << primitive->name() << " should have" << kInputElementSize
                             << "inputs. Please try other inputs";
  }
  auto x_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto x_shape = x_shape_ptr->GetShapeVector();
  if (MS_UNLIKELY(IsDynamicRank(x_shape))) {
    return GetOutputShape(x_shape);
  }
  auto x_rank = SizeToLong(x_shape.size());

  auto keep_dims_value = input_args[kInputIndex2]->GetValue();
  auto keep_dims_opt = GetScalarValue<bool>(keep_dims_value);
  if (MS_UNLIKELY(!keep_dims_opt.has_value())) {
    ShapeVector dynamic_rank_shape{abstract::TensorShape::kShapeRankAny};
    return GetOutputShape(dynamic_rank_shape);
  }
  for (size_t i = 0; i < x_shape.size(); i++) {
    if (x_shape[i] == 0) {
      MS_EXCEPTION(ValueError) << primitive->name() << " cannot deal with empty input. Please try other inputs";
    }
  }
  auto keep_dims = keep_dims_opt.value();
  auto out_dim = keep_dims ? x_rank : x_rank - 1;

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
                               << " with 0d input tensor, axis must be one of 0 or -1, but got " << axis << ".";
    }
    return std::make_shared<abstract::TupleShape>(
      abstract::BaseShapePtrList{x_shape_ptr->Clone(), x_shape_ptr->Clone()});
  }
  MS_CHECK_VALUE(axis >= -x_rank && axis < x_rank,
                 CheckAndConvertUtils::FormatCheckInRangeMsg("axis", axis, kIncludeLeft, {-x_rank, x_rank}, primitive));
  axis = axis < 0 ? axis + x_rank : axis;
  if (MS_UNLIKELY(x_shape[axis] == 0)) {
    MS_EXCEPTION(ValueError) << "For " << primitive->name() << ", the pos:" << axis
                             << " of input_x's shape can not be 0, but got " << x_shape_ptr->ToString();
  }

  ShapeVector output_shape(x_shape);
  if (keep_dims) {
    output_shape[axis] = 1;
  } else {
    (void)output_shape.erase(output_shape.begin() + axis);
  }
  return GetOutputShape(output_shape);
}

ShapeArray ArgMaxWithValueFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  if (input_values.size() < kInputElementSize) {
    MS_EXCEPTION(ValueError) << primitive->name() << " should have" << kInputElementSize
                             << "inputs. Please try other inputs";
  }
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto x_shape = x_tensor->shape();
  if (MS_UNLIKELY(IsDynamicRank(x_shape))) {
    return GetOutputShapeArray(x_shape);
  }

  const auto &keep_dims_value = input_values[kInputIndex2];
  auto keep_dims_opt = GetScalarValue<bool>(keep_dims_value);
  if (MS_UNLIKELY(!keep_dims_opt.has_value())) {
    ShapeVector dynamic_rank_shape{abstract::TensorShape::kShapeRankAny};
    return GetOutputShapeArray(x_shape);
  }
  if (std::any_of(x_shape.begin(), x_shape.end(), [](const auto &item) { return item == 0; })) {
    MS_EXCEPTION(ValueError) << primitive->name() << " cannot deal with empty input. Please try other inputs";
  }

  auto x_rank = SizeToLong(x_shape.size());
  auto keep_dims = keep_dims_opt.value();
  auto out_dim = keep_dims ? x_rank : x_rank - 1;

  const auto &axis_value = input_values[kInputIndex1];
  auto axis_value_opt = GetScalarValue<int64_t>(axis_value);
  if (MS_UNLIKELY(!axis_value_opt.has_value())) {
    if (x_rank == 0) {
      out_dim = 0;
    }
    return GetOutputShapeArray(ShapeVector(out_dim, abstract::TensorShape::kShapeDimAny));
  }
  auto axis = axis_value_opt.value();
  if (MS_UNLIKELY(x_rank == 0)) {
    if (axis != -1 && axis != 0) {
      MS_EXCEPTION(ValueError) << "For " << primitive->name()
                               << " with 0d input tensor, axis must be one of 0 or -1, but got " << axis << ".";
    }
    return GetOutputShapeArray(x_shape);
  }
  MS_CHECK_VALUE(axis >= -x_rank && axis < x_rank,
                 CheckAndConvertUtils::FormatCheckInRangeMsg("axis", axis, kIncludeLeft, {-x_rank, x_rank}, primitive));
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

TypePtr ArgMaxWithValueFuncImpl::InferType(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  TypePtr input_x_type = input_args[0]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_x_type, common_valid_types_with_bool, primitive->name());
  return std::make_shared<Tuple>(TypePtrList{std::make_shared<TensorType>(kInt64), input_args[0]->GetType()});
}

TypePtrList ArgMaxWithValueFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto &input_x_type = x_tensor->Dtype();
  TypePtrList type_ptr_list{kInt64, input_x_type};
  return type_ptr_list;
}

REGISTER_SIMPLE_INFER(kNameArgMaxWithValue, ArgMaxWithValueFuncImpl)
}  // namespace ops
}  // namespace mindspore
