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
#include "ops/op_name.h"
#include "ops/ops_func_impl/reduce_arithmetic.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr ReduceSumFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  MS_CHECK_VALUE(input_args.size() == 4, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                           "input_args number", SizeToLong(input_args.size()), kEqual, 4, primitive));
  auto axis_value = input_args[kInputIndex1]->GetValue();
  auto axis_array_opt = GetArrayValue<int64_t>(axis_value);
  bool is_empty_axis = axis_array_opt.has_value() && axis_array_opt->size() == 0;
  auto skip_mode_opt = GetScalarValue<bool>(input_args[kInputIndex3]->GetValue());
  if (MS_UNLIKELY(!skip_mode_opt.has_value())) {
    ShapeVector dynamic_rank_shape = {abstract::Shape::kShapeRankAny};
    return std::make_shared<abstract::Shape>(dynamic_rank_shape);
  }
  auto skip_mode = skip_mode_opt.value();
  if (skip_mode && is_empty_axis) {
    auto x_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
    return std::make_shared<abstract::Shape>(x_shape);
  }
  return ReduceInferShape(primitive, input_args);
}

TypePtr ReduceSumFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[0]->GetType()->Clone();
}
}  // namespace ops
}  // namespace mindspore
