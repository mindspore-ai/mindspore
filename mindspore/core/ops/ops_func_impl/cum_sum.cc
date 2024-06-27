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

#include "ops/ops_func_impl/cum_sum.h"
#include <set>
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
BaseShapePtr CumSumFuncImpl::InferShape(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape = input_args[kIndex0]->GetShape();
  if (x_shape->IsDynamic()) {
    return x_shape->cast<abstract::ShapePtr>();
  }
  auto x_shape_vec = x_shape->GetShapeVector();
  if (IsDynamicRank(x_shape_vec)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  }
  auto rank = SizeToLong(x_shape_vec.size());
  (void)CheckAndConvertUtils::CheckInteger("rank of 'x'", rank, kGreaterThan, 0, primitive->name());
  auto axis_ptr = input_args[kIndex1];
  MS_EXCEPTION_IF_NULL(axis_ptr);
  auto axis = axis_ptr->GetValue();
  auto axis_opt = GetScalarValue<int64_t>(axis);
  if (axis_opt.has_value()) {
    auto axis_value = axis_opt.value();
    MS_CHECK_VALUE(
      axis_value >= -rank && axis_value < rank,
      CheckAndConvertUtils::FormatCheckInRangeMsg("axis", axis_value, kIncludeLeft, {-rank, rank}, primitive));
  }
  return std::make_shared<abstract::TensorShape>(x_shape_vec);
}

TypePtr CumSumFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool is_ascend = (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice);
  std::set<TypePtr> valid_x_types;
  if (is_ascend) {
    valid_x_types = {kInt8, kUInt8, kInt32, kFloat16, kFloat32, kFloat64};
  } else {
    valid_x_types = common_valid_types_with_complex;
  }
  auto x_type = input_args[kInputIndex0]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_x_types, prim_name);
  auto axis_type = input_args[kInputIndex1]->GetType();
  const std::set<TypePtr> valid_axis_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTypeValid("axis", axis_type, valid_axis_types, prim_name);
  return x_type->Clone();
}
}  // namespace ops
}  // namespace mindspore
