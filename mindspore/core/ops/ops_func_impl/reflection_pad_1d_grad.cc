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

#include "ops/ops_func_impl/reflection_pad_1d_grad.h"

namespace mindspore {
namespace ops {
BaseShapePtr ReflectionPad1DGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args) const {
  auto input = input_args[kInputIndex1]->GetShape();
  auto input_shape = input->GetShapeVector();
  return std::make_shared<abstract::Shape>(input_shape);
}

TypePtr ReflectionPad1DGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) const {
  auto grad_output_type = input_args[kInputIndex0]->GetType();
  return grad_output_type->Clone();
}
}  // namespace ops
}  // namespace mindspore
