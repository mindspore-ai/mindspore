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

#include "ops/ops_func_impl/xlogy_scalar_self.h"
#include <memory>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
BaseShapePtr XLogYScalarSelfFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) const {
  auto input1_shape = input_args[kInputIndex1]->GetShape();
  MS_EXCEPTION_IF_NULL(input1_shape);
  return input1_shape->Clone();
}

TypePtr XLogYScalarSelfFuncImpl::InferType(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto x2_type = input_args[kInputIndex1]->GetType();
  MS_EXCEPTION_IF_NULL(x2_type);
  if (IsFloatType(x2_type)) {
    if (x2_type->isa<TensorType>()) {
      return {x2_type};
    }
    return {std::make_shared<TensorType>(x2_type)};
  }
  return {std::make_shared<TensorType>(kFloat32)};
}

ShapeArray XLogYScalarSelfFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &y_tensor = input_values[kInputIndex1]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(y_tensor);
  auto y_shape_vector = y_tensor->shape();
  return {y_shape_vector};
}

TypePtrList XLogYScalarSelfFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &y_tensor = input_values[kInputIndex1]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(y_tensor);
  auto x2_type = y_tensor->Dtype();

  if (IsFloatType(x2_type)) {
    return {x2_type};
  }
  return {kFloat32};
}

REGISTER_SIMPLE_INFER(kNameXLogYScalarSelf, XLogYScalarSelfFuncImpl)
}  // namespace ops
}  // namespace mindspore
