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
#include <set>
#include "ops/ops_func_impl/l1_loss_ext.h"
#include "utils/check_convert_utils.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
BaseShapePtr L1LossExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &prim_name = primitive->name();

  MS_EXCEPTION_IF_NULL(input_args[kInputIndex2]);
  auto reduction = GetScalarValue<int64_t>(input_args[kInputIndex2]->BuildValue());
  if (reduction.has_value() && static_cast<Reduction>(reduction.value()) == Reduction::NONE) {
    return BroadCastInferShape(prim_name, input_args);
  }

  return std::make_shared<abstract::Shape>(ShapeVector{});
}

TypePtr L1LossExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  auto input_type = input_args[kInputIndex0]->GetType();
  auto target_type = input_args[kInputIndex1]->GetType();
  MS_EXCEPTION_IF_NULL(input_type);
  MS_EXCEPTION_IF_NULL(target_type);
  auto input_real_type = input_type->cast<TensorTypePtr>()->element()->type_id();
  return input_real_type == kNumberTypeInt64 ? target_type : input_type;
}

TypePtrList L1LossExtFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  auto input_type = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>()->Dtype();
  auto target_type = input_values[kInputIndex1]->cast<tensor::BaseTensorPtr>()->Dtype();
  MS_EXCEPTION_IF_NULL(input_type);
  MS_EXCEPTION_IF_NULL(target_type);
  auto input_real_type = input_type->type_id();
  if (input_real_type == kNumberTypeInt64) {
    return {target_type};
  } else {
    return {input_type};
  }
}

ShapeArray L1LossExtFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &prim_name = primitive->name();

  MS_EXCEPTION_IF_NULL(input_values[kInputIndex2]);
  auto reduction = GetScalarValue<int64_t>(input_values[kInputIndex2]);
  if (reduction.has_value() && static_cast<Reduction>(reduction.value()) == Reduction::NONE) {
    return {BroadCastInferShape(prim_name, input_values)};
  }

  return {ShapeVector{}};
}

REGISTER_SIMPLE_INFER(kNameL1LossExt, L1LossExtFuncImpl)
}  // namespace ops
}  // namespace mindspore
