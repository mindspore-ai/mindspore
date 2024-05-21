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

#include "ops/ops_func_impl/rms_norm.h"

#include <string>
#include <map>
#include "abstract/dshape.h"
#include "ops/op_def.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr RmsNormFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape_ptr = input_args[0]->GetShape();
  MS_EXCEPTION_IF_NULL(x_shape_ptr);
  auto x_shape = x_shape_ptr->GetShapeVector();
  auto x_rank = x_shape.size();
  auto gamma_shape_ptr = input_args[1]->GetShape();
  MS_EXCEPTION_IF_NULL(gamma_shape_ptr);
  auto gamma_shape = gamma_shape_ptr->GetShapeVector();
  auto gamma_rank = gamma_shape.size();
  auto rstd_shape = x_shape;
  if (IsDynamicRank(gamma_shape)) {
    if (!IsDynamicRank(x_shape)) {
      rstd_shape = ShapeVector(x_rank, abstract::TensorShape::kShapeDimAny);
    } else {
      rstd_shape = ShapeVector{abstract::TensorShape::kShapeRankAny};
    }
  } else if (!IsDynamicRank(x_shape)) {
    // gamma shape dim less than corresponding x shape dim and does not equal to 1 is allowed for now. But this will
    // be intercepted in Ascend backend in the future.
    MS_EXCEPTION_IF_CHECK_FAIL(gamma_rank <= x_rank, "The [gamma] rank can not be bigger than the [input] rank.");
    for (auto dim = x_rank - gamma_rank; dim < x_rank; dim++) {
      rstd_shape[dim] = 1;
    }
  }
  auto y_output_ptr = std::make_shared<abstract::Shape>(x_shape);
  auto rstd_output_ptr = std::make_shared<abstract::Shape>(rstd_shape);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{y_output_ptr, rstd_output_ptr});
}

TypePtr RmsNormFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x_dtype = input_args[kInputIndex0]->GetType();
  auto gamma_dtype = input_args[kInputIndex1]->GetType();
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", x_dtype);
  (void)types.emplace("gamma", gamma_dtype);
  (void)CheckAndConvertUtils::CheckTypeSame(types, prim_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{x_dtype, kFloat32});
}

int32_t RmsNormFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto epsilon_value = GetScalarValue<pyfloat>(input_args[kInputIndex2]->GetValue());
  if (MS_UNLIKELY(!epsilon_value.has_value())) {
    return OP_CHECK_RETRY;
  }
  MS_CHECK_VALUE(epsilon_value.value() > 0 && epsilon_value.value() <= 1,
                 CheckAndConvertUtils::FormatCheckInRangeMsg<pyfloat>("epsilon", epsilon_value.value(), kIncludeRight,
                                                                      {0., 1.}, primitive));
  return OP_CHECK_SUCCESS;
}
}  // namespace ops
}  // namespace mindspore
