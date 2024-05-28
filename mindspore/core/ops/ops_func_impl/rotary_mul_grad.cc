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

#include "ops/ops_func_impl/rotary_mul_grad.h"
#include <memory>

namespace mindspore {
namespace ops {
BaseShapePtr RotaryMulGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) const {
  const auto &dx_shape = input_args[kIndex0]->GetShape()->Clone();
  const auto &dr1_shape = input_args[kIndex1]->GetShape()->Clone();
  const auto &dr2_shape = input_args[kIndex2]->GetShape()->Clone();
  return std::make_shared<abstract::TupleShape>(abstract::BaseShapePtrList{dx_shape, dr1_shape, dr2_shape});
}

TypePtr RotaryMulGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  const auto &dx_type = input_args[kIndex0]->GetType();
  const auto &dr1_type = input_args[kIndex1]->GetType();
  const auto &dr2_type = input_args[kIndex2]->GetType();
  return std::make_shared<Tuple>(TypePtrList{dx_type, dr1_type, dr2_type});
}
}  // namespace ops
}  // namespace mindspore
