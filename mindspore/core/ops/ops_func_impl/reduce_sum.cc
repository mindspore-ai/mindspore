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

#include "ops/ops_func_impl/reduce_sum.h"
#include "ops/ops_func_impl/reduce_arithmetic.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr ReduceSumFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_CHECK_VALUE(input_args.size() == 4, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                           "input_args number", SizeToLong(input_args.size()), kEqual, 4, primitive));
  MS_EXCEPTION_IF_NULL(input_args[0]);
  MS_EXCEPTION_IF_NULL(input_args[1]);
  MS_EXCEPTION_IF_NULL(input_args[3]);
  auto x_shape = input_args[0]->GetShape()->GetShapeVector();
  auto axis_value = input_args[1]->GetValue();
  MS_EXCEPTION_IF_NULL(axis_value);
  auto axis_array_opt = GetArrayValue<int64_t>(axis_value);
  auto skip_mode_value = input_args[3]->GetValue();
  MS_EXCEPTION_IF_NULL(skip_mode_value);
  bool skip_mode_unknown = skip_mode_value->isa<ValueAny>();  // the skip_mode is unknown
  if (skip_mode_unknown) {
    ShapeVector dynamic_rank_shape = {abstract::Shape::kShapeRankAny};
    return std::make_shared<abstract::Shape>(dynamic_rank_shape);
  }

  MS_CHECK_VALUE(skip_mode_value->isa<BoolImm>(), "The skip_mode input for " + primitive->name() + " must be bool.");
  bool skip_mode = GetValue<bool>(skip_mode_value);

  bool is_empty_axis = axis_array_opt.has_value() && axis_array_opt->size() == 0;
  if (skip_mode && is_empty_axis) {
    return std::make_shared<abstract::Shape>(x_shape);
  }
  return ReduceInferShape(primitive, input_args);
}

TypePtr ReduceSumFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[0]);
  MS_EXCEPTION_IF_NULL(input_args[0]->GetType());
  return input_args[0]->GetType()->Clone();
}
}  // namespace ops
}  // namespace mindspore
