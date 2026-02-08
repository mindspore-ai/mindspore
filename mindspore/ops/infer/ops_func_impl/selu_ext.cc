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
#include <set>
#include "infer/ops_func_impl/selu_ext.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace ops {
BaseShapePtr SeLUExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetShape()->Clone();
}

TypePtr SeLUExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  auto x_type = input_args[kIndex0]->GetType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kBFloat16};
  auto tensor_type = x_type->cast<TensorTypePtr>();
  auto real_type = tensor_type->element();
  (void)CheckAndConvertUtils::CheckSubClass("input", real_type, valid_types, primitive->name());
  return x_type;
}
TypePtrList SeLUExtFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::TensorPtr>();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kBFloat16};
  const auto &input_type = x_tensor->Dtype();
  (void)CheckAndConvertUtils::CheckSubClass("input", input_type, valid_types, primitive->name());
  return {input_type};
}
ShapeArray SeLUExtFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::TensorPtr>();
  return {x_tensor->shape()};
}
REGISTER_SIMPLE_INFER(kNameSeLUExt, SeLUExtFuncImpl)
}  // namespace ops
}  // namespace mindspore
