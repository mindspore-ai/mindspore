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

#include <memory>
#include "ops/ops_func_impl/remainder_tensor_scalar.h"
#include "ops/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore::ops {
BaseShapePtr RemainderTensorScalarFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                       const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kInputIndex0]->GetShape();
}

TypePtr RemainderTensorScalarFuncImpl::InferType(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) const {
  auto input_type = input_args[kInputIndex0]->GetType();
  auto other_type = input_args[kInputIndex1]->GetType();
  MS_EXCEPTION_IF_NULL(input_type);
  MS_EXCEPTION_IF_NULL(other_type);
  auto out_type = PromoteType(input_type, other_type, primitive->name());
  return std::make_shared<TensorType>(out_type);
}

// simple infer
TypePtrList RemainderTensorScalarFuncImpl::InferType(const PrimitivePtr &primitive,
                                                     const ValuePtrList &input_values) const {
  const auto &input_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  const auto &other_value = input_values[kInputIndex1];
  MS_EXCEPTION_IF_NULL(input_tensor);
  MS_EXCEPTION_IF_NULL(other_value);
  const auto &input_type = input_tensor->Dtype();
  const auto &other_type = other_value->type();
  return {PromoteType(input_type, other_type, primitive->name())};
}

ShapeArray RemainderTensorScalarFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                     const ValuePtrList &input_values) const {
  const auto &input_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  return {input_tensor->shape()};
}

REGISTER_SIMPLE_INFER(kNameRemainderTensorScalar, RemainderTensorScalarFuncImpl)
}  // namespace mindspore::ops
