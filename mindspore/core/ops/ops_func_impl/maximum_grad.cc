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
#include "ops/ops_func_impl/maximum_grad.h"

#include <memory>
#include <utility>

namespace mindspore {
namespace ops {
BaseShapePtr MaximumGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  const auto x_shape = input_args[kIndex0]->GetShape();
  const auto y_shape = input_args[kIndex1]->GetShape();
  std::vector<abstract::BaseShapePtr> shape_list{x_shape->Clone(), y_shape->Clone()};
  return std::make_shared<abstract::TupleShape>(std::move(shape_list));
}

TypePtr MaximumGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  const auto x_type = input_args[kIndex0]->GetType();
  const auto y_type = input_args[kIndex1]->GetType();
  std::vector<TypePtr> type_list{x_type->Clone(), y_type->Clone()};
  return std::make_shared<Tuple>(std::move(type_list));
}
}  // namespace ops
}  // namespace mindspore
