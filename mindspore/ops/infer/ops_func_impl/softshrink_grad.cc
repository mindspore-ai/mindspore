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
#include "infer/ops_func_impl/softshrink_grad.h"

#include "abstract/dshape.h"
#include "primitive/op_name.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "ops_utils/op_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace ops {

BaseShapePtr SoftShrinkGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) const {
  // Get input tensor shape.
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex1]);
  auto in_shape = input_args[kInputIndex1]->GetShape();
  MS_EXCEPTION_IF_NULL(in_shape);

  return in_shape->Clone();
}

TypePtr SoftShrinkGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex1]);
  auto x_type = input_args[kInputIndex1]->GetType();
  MS_EXCEPTION_IF_NULL(x_type);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kBFloat16};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("input_grad", input_args[kInputIndex0]->GetType());
  (void)types.emplace("input_x", input_args[kInputIndex1]->GetType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());
  return x_type;
}

TypePtrList SoftShrinkGradFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &input_grad_tensor = input_values[kIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input_grad_tensor);
  const auto &input_x_tensor = input_values[kIndex1]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input_x_tensor);

  const auto &input_grad_type = input_grad_tensor->Dtype();
  const auto &input_x_type = input_x_tensor->Dtype();

  if (input_grad_type->type_id() != input_x_type->type_id()) {
    MS_LOG_EXCEPTION << "For " << primitive->name()
                     << ", the grad type must be same as input type, but got input_grad_type: "
                     << input_grad_type->ToString() << " and input_x_type: " << input_x_type->ToString();
  }
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kBFloat16};
  (void)CheckAndConvertUtils::CheckSubClass("input_x", input_x_type, valid_types, primitive->name());
  return {input_x_type};
}

ShapeArray SoftShrinkGradFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex1]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  return {x_tensor->shape()};
}
REGISTER_SIMPLE_INFER(kNameSoftShrinkGrad, SoftShrinkGradFuncImpl)

}  // namespace ops
}  // namespace mindspore
