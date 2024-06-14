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

#include "ops/ops_func_impl/l1_loss_backward_ext.h"
#include <algorithm>
#include <memory>
#include <set>
#include <utility>
#include "ops/op_utils.h"
#include "utils/ms_context.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
TypePtr L1LossBackwardExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                             const std::vector<abstract::AbstractBasePtr> &input_args) const {
  auto input_type = input_args[kInputIndex1]->GetType();
  return input_type;
}

BaseShapePtr L1LossBackwardExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                   const std::vector<abstract::AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &prim_name = primitive->name();

  return BroadCastInferShape(prim_name, input_args);
}

TypePtrList L1LossBackwardExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                                 const ValuePtrList &input_values) const {
  auto input_type = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>()->Dtype();
  return {input_type};
}

ShapeArray L1LossBackwardExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                 const ValuePtrList &input_values) const {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &prim_name = primitive->name();

  return {BroadCastInferShape(prim_name, input_values)};
}

REGISTER_SIMPLE_INFER(kNameL1LossBackwardExt, L1LossBackwardExtFuncImpl)
}  // namespace ops
}  // namespace mindspore
