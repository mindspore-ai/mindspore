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

#include "ops/ops_func_impl/minimum_grad.h"

#include <memory>
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr MinimumGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape = input_args[kInputIndex0]->GetShape()->Clone();
  auto y_shape = input_args[kInputIndex1]->GetShape()->Clone();
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{x_shape, y_shape});
}

TypePtr MinimumGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  auto x_type = input_args[kInputIndex0]->GetType();
  MS_EXCEPTION_IF_NULL(x_type);
  std::vector<TypePtr> type_tuple;
  type_tuple.push_back(x_type->Clone());
  type_tuple.push_back(x_type->Clone());
  return std::make_shared<Tuple>(type_tuple);
}
}  // namespace ops
}  // namespace mindspore
