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
#include "ops/ops_func_impl/logit_grad.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr LogitGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto grad_shape = input_args[kIndex0]->GetShape();
  auto x_shape = input_args[kIndex1]->GetShape();
  const auto grad_shape_vec = grad_shape->GetShapeVector();
  const auto x_shape_vec = x_shape->GetShapeVector();
  if (MS_UNLIKELY((IsDynamic(grad_shape_vec) && !IsDynamic(x_shape_vec)) ||
                  (IsDynamicRank(grad_shape_vec) && !IsDynamicRank(x_shape_vec)))) {
    return x_shape->Clone();
  }
  return grad_shape->Clone();
}

TypePtr LogitGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetType()->Clone();
}
}  // namespace ops
}  // namespace mindspore
