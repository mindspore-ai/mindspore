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

#include "ops/ops_func_impl/cum_prod.h"
#include <set>
#include "utils/check_convert_utils.h"
#include "ops/ops_frontend_func_impl.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr CumProdFuncImpl::InferShape(const PrimitivePtr &primitive,
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

TypePtr CumProdFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  auto x_type = input_args[kInputIndex0]->GetType();
  return x_type->Clone();
}

class CumProdFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  // Do not override this interface if the op has no InferValue
  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    if (input_args.empty()) {
      return nullptr;
    }
    auto axis = input_args[kInputIndex1]->GetValue();
    if (axis == nullptr) {
      MS_EXCEPTION(ValueError) << "For " << primitive->name() << ", the 'axis' cannot be None, but got " << axis;
    }
    return nullptr;
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("CumProd", CumProdFrontendFuncImpl);
}  // namespace ops
}  // namespace mindspore
