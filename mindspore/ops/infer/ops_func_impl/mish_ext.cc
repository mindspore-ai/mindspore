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
#include "infer/ops_func_impl/mish_ext.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace ops {
BaseShapePtr MishExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kInputIndex0]->GetShape()->Clone();
}

TypePtr MishExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  auto x_type = input_args[kInputIndex0]->GetType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  auto tensor_type = x_type->cast<TensorTypePtr>();
  auto real_type = tensor_type->element();
  (void)CheckAndConvertUtils::CheckSubClass("input", real_type, valid_types, primitive->name());
  return x_type;
}

TypePtrList MishExtFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto &input_type = x_tensor->Dtype();
  const auto input_type_id = input_type->type_id();
  if (input_type_id != kNumberTypeFloat16 && input_type_id != kNumberTypeFloat32) {
    MS_EXCEPTION(TypeError) << "For primitive[" << primitive->name()
                            << "], the input argument[input] must be a type of {Float16, Float32}"
                            << " but got " << input_type->ToString() << ".";
  }
  return {input_type};
}

ShapeArray MishExtFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  return {x_tensor->shape()};
}
REGISTER_SIMPLE_INFER(kNameMishExt, MishExtFuncImpl)
}  // namespace ops
}  // namespace mindspore
