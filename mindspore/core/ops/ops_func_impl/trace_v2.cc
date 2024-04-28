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

#include "ops/ops_func_impl/trace_v2.h"

#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/shape_utils.h"
#include "utils/log_adapter.h"
#include "ir/primitive.h"
#include "abstract/dshape.h"
#include "base/base.h"
#include "utils/check_convert_utils.h"

namespace mindspore::ops {
BaseShapePtr TraceV2FuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto base_shape = input_args[kInputIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(base_shape);
  const auto &shape = base_shape->GetShapeVector();
  int64_t x_rank = shape.size();
  if (IsDynamicRank(shape)) {
    ShapeVector out_shape = {abstract::Shape::kShapeRankAny};
    auto out_shape_ptr = std::make_shared<abstract::Shape>(out_shape);
    return out_shape_ptr;
  }
  if (x_rank < 2) {
    MS_LOG(EXCEPTION) << "For Primitive[Tracev2], the dim of input 'x' should greater or equal to 2, but got 'x' at "
                      << x_rank << "-dimention";
  }

  MS_EXCEPTION_IF_NULL(input_args[kInputIndex2]);
  auto axis1_opt = GetScalarValue<int64_t>(input_args[kInputIndex2]->GetValue());
  int64_t axis1_value = 0;
  if (axis1_opt.has_value()) {
    axis1_value = axis1_opt.value();
  }
  auto axis2_opt = GetScalarValue<int64_t>(input_args[kInputIndex3]->GetValue());
  int64_t axis2_value = 1;
  if (axis2_opt.has_value()) {
    axis2_value = axis2_opt.value();
  }
  MS_CHECK_VALUE(
    axis1_value >= -x_rank && axis1_value < x_rank,
    CheckAndConvertUtils::FormatCheckInRangeMsg("axis1", axis1_value, kIncludeLeft, {-x_rank, x_rank}, primitive));
  MS_CHECK_VALUE(
    axis2_value >= -x_rank && axis2_value < x_rank,
    CheckAndConvertUtils::FormatCheckInRangeMsg("axis2", axis2_value, kIncludeLeft, {-x_rank, x_rank}, primitive));

  axis1_value = axis1_value < 0 ? axis1_value + x_rank : axis1_value;
  axis2_value = axis2_value < 0 ? axis2_value + x_rank : axis2_value;

  if (axis1_value == axis2_value) {
    MS_LOG(EXCEPTION) << "For Primitive[Tracev2], the value of 'axis1' and 'axis2' must be different, but got 'axis1': "
                      << axis1_value << " and 'axis2': " << axis2_value;
  }
  std::vector<int64_t> out_dims;
  for (int64_t i = 0; i < x_rank; i++) {
    if (i != axis1_value && i != axis2_value) {
      out_dims.emplace_back(shape[i]);
    }
  }
  auto out_shape = std::make_shared<abstract::Shape>(out_dims);
  return out_shape;
}

TypePtr TraceV2FuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex4]);
  auto op_name = primitive->name();
  TypePtr dst_type{nullptr};
  if (input_args[kInputIndex4]->GetType()->isa<TypeNone>()) {
    MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
    auto input_type = input_args[kInputIndex0]->GetType();
    auto input_type_ptr = input_type->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(input_type_ptr);
    auto input_type_id = input_type_ptr->element()->type_id();
    static const std::vector<TypeId> type_to_int64 = {
      kNumberTypeInt8, kNumberTypeInt16, kNumberTypeInt32, kNumberTypeUInt8, kNumberTypeUInt16, kNumberTypeUInt32,
    };
    bool is_type_to_int64 = std::any_of(type_to_int64.begin(), type_to_int64.end(),
                                        [&input_type_id](const TypeId &type_id) { return input_type_id == type_id; });
    if (is_type_to_int64) {
      dst_type = std::make_shared<TensorType>(kInt64);
    } else {
      dst_type = input_type->Clone();
    }
  } else {
    auto dtype_value = GetScalarValue<int64_t>(input_args[kInputIndex4]->GetValue());
    MS_CHECK_VALUE(dtype_value.has_value(),
                   CheckAndConvertUtils::FormatCommMsg("For primitive[", op_name,
                                                       "], the `dtype` should has valid value for static type."));
    dst_type = std::make_shared<TensorType>(TypeIdToType(static_cast<TypeId>(dtype_value.value())));
  }
  return dst_type;
}
}  // namespace mindspore::ops
