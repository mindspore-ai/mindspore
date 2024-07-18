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

#include "ops/ops_func_impl/xlogy_scalar_other.h"
#include <memory>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
BaseShapePtr XLogYScalarOtherFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) const {
  auto input0_shape = input_args[kInputIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(input0_shape);
  return input0_shape->Clone();
}

TypePtr XLogYScalarOtherFuncImpl::InferType(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  auto x1_type = input_args[kInputIndex0]->GetType();
  MS_EXCEPTION_IF_NULL(x1_type);
  if (IsFloatType(x1_type)) {
    if (x1_type->isa<TensorType>()) {
      return {x1_type};
    }
    return {std::make_shared<TensorType>(x1_type)};
  }
  return {std::make_shared<TensorType>(kFloat32)};
}

ShapeArray XLogYScalarOtherFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  auto x_shape_vector = x_tensor->shape();
  return {x_shape_vector};
}

TypePtrList XLogYScalarOtherFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  auto x1_type = x_tensor->Dtype();
  if (IsFloatType(x1_type)) {
    return {x1_type};
  }
  return {kFloat32};
}

REGISTER_SIMPLE_INFER(kNameXLogYScalarOther, XLogYScalarOtherFuncImpl)
}  // namespace ops
}  // namespace mindspore
