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
#include "ops/ops_func_impl/matrix_exp.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr MatrixExpFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  const auto x_shape = input_args[kIndex0]->GetShape();
  const auto x_shape_vec = x_shape->GetShapeVector();
  if (MS_UNLIKELY(IsDynamicRank(x_shape_vec))) {
    return x_shape->Clone();
  }
  constexpr const int64_t kMinRank = 2;
  auto x_rank = x_shape_vec.size();
  MS_CHECK_VALUE(x_rank >= kMinRank,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("x rank", x_rank, kGreaterEqual, kMinRank, primitive));
  auto x_row = x_shape_vec[x_rank - kIndex1];
  auto x_col = x_shape_vec[x_rank - kIndex2];
  if (MS_UNLIKELY(x_row == abstract::TensorShape::kShapeDimAny || x_col == abstract::TensorShape::kShapeDimAny)) {
    return x_shape->Clone();
  }
  if (MS_UNLIKELY(x_row != x_col)) {
    MS_EXCEPTION(ValueError) << "For " << primitive->name() << ", the input expects a tensor of squared matrices"
                             << ", but got shape " << x_shape_vec << ".";
  }
  if (MS_UNLIKELY(x_row < 1)) {
    MS_EXCEPTION(ValueError) << "For MatrixExp, the input x's last dimension must be at least 1.";
  }
  return x_shape->Clone();
}

TypePtr MatrixExpFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetType()->Clone();
}
}  // namespace ops
}  // namespace mindspore
