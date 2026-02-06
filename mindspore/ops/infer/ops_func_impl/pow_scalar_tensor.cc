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

#include "infer/ops_func_impl/pow_scalar_tensor.h"
#include <set>
#include "ops/ops_func_impl/simple_infer.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_p.h"

namespace mindspore::ops {
BaseShapePtr PowScalarTensorFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape = input_args[kInputIndex1]->GetShape();
  return input_shape->Clone();
}

TypePtr PowCheckAndInferType(const TypeId input_type_id, const TypeId exp_type_id, const TypePtr exp_type) {
  static const std::set<TypeId> valid_types = {
    kNumberTypeUInt8, kNumberTypeInt8,    kNumberTypeInt16,   kNumberTypeInt32,   kNumberTypeInt64,
    kNumberTypeBool,  kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64, kNumberTypeBFloat16};
  if (valid_types.find(exp_type_id) == valid_types.end()) {
    MS_EXCEPTION(TypeError) << "For primitive [PowTensorScalar], 'exponent' type is not supported, 'exponent' type: "
                            << exp_type->ToString();
  }
  static const std::vector<TypeId> int_or_bool = {kNumberTypeUInt8, kNumberTypeInt8,  kNumberTypeInt16,
                                                  kNumberTypeInt32, kNumberTypeInt64, kNumberTypeBool};
  bool is_int_or_bool = std::any_of(int_or_bool.begin(), int_or_bool.end(),
                                    [&exp_type_id](const TypeId &type_id) { return exp_type_id == type_id; });
  if (input_type_id == kNumberTypeFloat32 && is_int_or_bool) {
    return kFloat32;
  }

  if (input_type_id == kNumberTypeInt64 && exp_type_id == kNumberTypeBool) {
    return kInt64;
  }

  if (exp_type_id == kNumberTypeBool && input_type_id == kNumberTypeBool) {
    MS_EXCEPTION(TypeError)
      << "For primitive [PowTensorScalar], 'input' and 'exponent' cannot be bool at the same time.";
  }

  return exp_type;
}

TypePtr PowScalarTensorFuncImpl::InferType(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  const auto &input_type = input_args[kInputIndex0]->GetType();
  auto input_type_id = input_type->type_id();
  const auto &exp_type = input_args[kInputIndex1]->GetType();
  auto exp_type_id = exp_type->cast<TensorTypePtr>()->element()->type_id();

  return PowCheckAndInferType(input_type_id, exp_type_id, exp_type);
}

TypePtrList PowScalarTensorFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &input_type = input_values[kInputIndex0]->type();
  auto input_type_id = input_type->type_id();
  const auto &exp_tensor = input_values[kInputIndex1]->cast<tensor::TensorPtr>();
  auto exp_type_id = exp_tensor->Dtype()->type_id();

  return {PowCheckAndInferType(input_type_id, exp_type_id, exp_tensor->Dtype())};
}

ShapeArray PowScalarTensorFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex1]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  return {x_tensor->shape()};
}

REGISTER_SIMPLE_INFER(kNamePowScalarTensor, PowScalarTensorFuncImpl)
}  // namespace mindspore::ops
