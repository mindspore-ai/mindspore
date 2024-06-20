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

#include <vector>
#include <memory>
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "ops/ops_func_impl/slice_ext.h"

namespace mindspore::ops {
BaseShapePtr SliceExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto input_x_shape = input_args[0]->GetShape()->GetShapeVector();
  (void)CheckAndConvertUtils::CheckInteger("rank of input_x", SizeToLong(input_x_shape.size()), kGreaterThan, 0,
                                           prim_name);

  if (IsDynamicRank(input_x_shape)) {
    return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::TensorShape::kShapeRankAny});
  }

  auto axis_value_opt = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue());
  auto input_begin_value_opt = GetScalarValue<int64_t>(input_args[kInputIndex2]->GetValue());
  auto input_end_value_opt = GetScalarValue<int64_t>(input_args[kInputIndex3]->GetValue());
  auto input_step_value_opt = GetScalarValue<int64_t>(input_args[kInputIndex4]->GetValue());
  if (!axis_value_opt.has_value() || !input_begin_value_opt.has_value() || !input_end_value_opt.has_value() ||
      !input_step_value_opt.has_value()) {
    return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::TensorShape::kShapeRankAny});
  }

  auto axis_value = axis_value_opt.value();
  auto input_begin_value = input_begin_value_opt.value();
  auto input_end_value = input_end_value_opt.value();
  auto x_rank = SizeToLong(input_x_shape.size());

  if (input_begin_value > input_end_value) {
    MS_EXCEPTION(ValueError) << "For Slice, the end must be no greater than start.";
  }

  MS_CHECK_VALUE(axis_value >= -x_rank && axis_value < x_rank, "For primitive [SliceExt]: dim exceed range");
  axis_value = axis_value < 0 ? axis_value + x_rank : axis_value;

  auto x_axis_size = input_x_shape[axis_value];

  if (input_x_shape[axis_value] == abstract::Shape::kShapeDimAny) {
    return std::make_shared<abstract::TensorShape>(input_x_shape);
  }

  MS_CHECK_VALUE(input_begin_value >= -x_axis_size && input_begin_value <= x_axis_size,
                 "For primitive [SliceExt]: start exceed range");
  auto input_length = input_end_value - input_begin_value;
  input_begin_value = input_begin_value < 0 ? input_begin_value + x_axis_size : input_begin_value;
  input_end_value = input_begin_value + input_length;
  MS_CHECK_VALUE(input_end_value >= -x_axis_size && input_end_value <= x_axis_size,
                 "For primitive [SliceExt]: end exceed range");
  auto out_shape = input_x_shape;
  out_shape[axis_value] = input_end_value - input_begin_value;

  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr SliceExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const {
  auto input_type = input_args[kIndex0]->GetType();
  const std::set<TypePtr> valid_type = {kInt8, kInt32, kInt64, kUInt8, kFloat16, kFloat32, kBool, kBFloat16};
  (void)CheckAndConvertUtils::CheckTypeValid("input", input_type, valid_type, primitive->name());

  return input_type->Clone();
}
}  // namespace mindspore::ops
