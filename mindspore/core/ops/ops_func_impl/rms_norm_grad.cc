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

#include "ops/ops_func_impl/rms_norm_grad.h"

#include <string>
#include <map>
#include "abstract/dshape.h"
#include "ops/op_def.h"
#include "ops/op_name.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr RmsNormGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape_ptr = input_args[kInputIndex1]->GetShape();
  MS_EXCEPTION_IF_NULL(x_shape_ptr);
  auto x_shape = x_shape_ptr->GetShapeVector();
  auto rstd_shape_ptr = input_args[kInputIndex2]->GetShape();
  MS_EXCEPTION_IF_NULL(rstd_shape_ptr);
  auto rstd_shape = rstd_shape_ptr->GetShapeVector();
  auto gamma_shape_ptr = input_args[kInputIndex3]->GetShape();
  MS_EXCEPTION_IF_NULL(gamma_shape_ptr);
  auto gamma_shape = gamma_shape_ptr->GetShapeVector();
  if (!IsDynamicRank(gamma_shape)) {
    auto gamma_rank = gamma_shape.size();
    if (!IsDynamicRank(x_shape)) {
      auto x_rank = x_shape.size();
      MS_EXCEPTION_IF_CHECK_FAIL(gamma_rank <= x_rank, "The [gamma] rank must not be bigger than the [input] rank.");
    }
    if (!IsDynamicRank(rstd_shape)) {
      auto rstd_rank = rstd_shape.size();
      for (auto dim = rstd_rank - gamma_rank; dim < rstd_rank; dim++) {
        MS_EXCEPTION_IF_CHECK_FAIL(
          rstd_shape[dim] == 1 || rstd_shape[dim] == abstract::Shape::kShapeDimAny,
          "The [rstd] got wrong shape, expect to be 1, but got: " + std::to_string(rstd_shape[dim]));
      }
    }
  }
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{x_shape_ptr->Clone(), gamma_shape_ptr->Clone()});
}

TypePtr RmsNormGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  auto x_dtype = input_args[kInputIndex1]->GetType();
  return std::make_shared<Tuple>(std::vector<TypePtr>{x_dtype, std::make_shared<TensorType>(kFloat32)});
}
}  // namespace ops
}  // namespace mindspore
