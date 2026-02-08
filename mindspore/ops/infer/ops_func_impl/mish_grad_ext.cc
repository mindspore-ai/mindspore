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

#include "infer/ops_func_impl/mish_grad_ext.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace ops {

BaseShapePtr MishGradExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kInputIndex1]->GetShape()->Clone();
}

TypePtr MishGradExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  auto x_type = input_args[kInputIndex1]->GetType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("dout", input_args[kInputIndex0]->GetType());
  (void)types.emplace("x", input_args[kInputIndex1]->GetType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());
  return x_type;
}

TypePtrList MishGradExtFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &dout_tensor = input_values[kIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(dout_tensor);
  const auto &x_tensor = input_values[kIndex1]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);

  const auto &dout_type = dout_tensor->Dtype();
  const auto &x_type = x_tensor->Dtype();
  const auto x_type_id = x_type->type_id();
  if (dout_type->type_id() != x_type_id) {
    MS_EXCEPTION(TypeError) << "For " << primitive->name()
                            << ", the grad type must be same as input type, but got dout_type: "
                            << dout_type->ToString() << " and x_type: " << x_type->ToString();
  }
  if (x_type_id != kNumberTypeFloat16 && x_type_id != kNumberTypeFloat32) {
    MS_EXCEPTION(TypeError) << "For primitive[" << primitive->name()
                            << "], the input argument[x] must be a type of {Float16, Float32}"
                            << " but got " << x_type->ToString() << ".";
  }
  return {x_type};
}

ShapeArray MishGradExtFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex1]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  return {x_tensor->shape()};
}
REGISTER_SIMPLE_INFER(kNameMishGradExt, MishGradExtFuncImpl)

}  // namespace ops
}  // namespace mindspore
