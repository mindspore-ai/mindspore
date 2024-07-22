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
#include "ops/ops_func_impl/remainder_scalar_tensor.h"
#include "ops/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore::ops {
BaseShapePtr RemainderScalarTensorFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                       const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kInputIndex1]->GetShape();
}

TypePtr RemainderScalarTensorFuncImpl::InferType(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) const {
  auto input_type = input_args[kInputIndex0]->GetType();
  auto other_type = input_args[kInputIndex1]->GetType();
  MS_EXCEPTION_IF_NULL(input_type);
  MS_EXCEPTION_IF_NULL(other_type);
  auto out_type = PromoteType(input_type, other_type, primitive->name());
  return std::make_shared<TensorType>(out_type);
}

// simple infer
TypePtrList RemainderScalarTensorFuncImpl::InferType(const PrimitivePtr &primitive,
                                                     const ValuePtrList &input_values) const {
  const auto &input_value = input_values[kInputIndex0];
  const auto &other_tensor = input_values[kInputIndex1]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_value);
  MS_EXCEPTION_IF_NULL(other_tensor);
  const auto &input_type = input_value->type();
  const auto &other_type = other_tensor->Dtype();
  return {PromoteType(input_type, other_type, primitive->name())};
}

ShapeArray RemainderScalarTensorFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                     const ValuePtrList &input_values) const {
  const auto &other_tensor = input_values[kInputIndex1]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(other_tensor);
  return {other_tensor->shape()};
}

REGISTER_SIMPLE_INFER(kNameRemainderScalarTensor, RemainderScalarTensorFuncImpl)
}  // namespace mindspore::ops
