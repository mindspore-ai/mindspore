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

#include "infer/ops_func_impl/pow_tensor_scalar.h"
#include <set>
#include "ops/ops_func_impl/simple_infer.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_p.h"

namespace mindspore::ops {
BaseShapePtr PowTensorScalarFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape = input_args[kInputIndex0]->GetShape();
  return input_shape->Clone();
}

TypePtr PowCheckAndInferType(const TypeId input_type_id, const TypeId exp_type_id, const TypePtr input_type,
                             const ValuePtr exp_value) {
  static const std::vector<TypeId> int_or_bool = {kNumberTypeUInt8, kNumberTypeInt8,  kNumberTypeInt16,
                                                  kNumberTypeInt32, kNumberTypeInt64, kNumberTypeBool};
  static const std::set<TypeId> valid_types = {
    kNumberTypeUInt8, kNumberTypeInt8,    kNumberTypeInt16,   kNumberTypeInt32,   kNumberTypeInt64,
    kNumberTypeBool,  kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64, kNumberTypeBFloat16};
  if (valid_types.find(input_type_id) == valid_types.end()) {
    MS_EXCEPTION(TypeError) << "For primitive [PowTensorScalar], 'input' type is not supported, 'input' type: "
                            << input_type->ToString();
  }
  bool is_int_or_bool = std::any_of(int_or_bool.begin(), int_or_bool.end(),
                                    [&input_type_id](const TypeId &type_id) { return input_type_id == type_id; });
  if (exp_type_id == kNumberTypeFloat32 && is_int_or_bool) {
    return kFloat32;
  }

  if (exp_type_id == kNumberTypeInt64 && input_type_id == kNumberTypeBool) {
    return kInt64;
  }

  if (exp_type_id == kNumberTypeBool && input_type_id == kNumberTypeBool) {
    MS_EXCEPTION(TypeError)
      << "For primitive [PowTensorScalar], 'input' and 'exponent' cannot be bool at the same time.";
  }

  if (exp_type_id == kNumberTypeInt64 && input_type_id != kNumberTypeBool && is_int_or_bool) {
    auto exp_opt = GetScalarValue<int64_t>(exp_value);
    if (MS_UNLIKELY(exp_opt.has_value() && exp_opt.value() < 0)) {
      MS_EXCEPTION(RuntimeError)
        << "For primitive [PowTensorScalar], Integers to negative integer powers are not allowed.";
    }
  }

  return input_type;
}

TypePtr PowTensorScalarFuncImpl::InferType(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  const auto &input_type = input_args[kInputIndex0]->GetType();
  auto input_type_id = input_type->cast<TensorTypePtr>()->element()->type_id();
  const auto &exp_type = input_args[kInputIndex1]->GetType();
  auto exp_type_id = exp_type->type_id();

  return PowCheckAndInferType(input_type_id, exp_type_id, input_type, input_args[kInputIndex1]->GetValue());
}

TypePtrList PowTensorScalarFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::TensorPtr>();
  auto input_type_id = x_tensor->Dtype()->type_id();
  const auto &exp_type = input_values[kInputIndex1]->type();
  auto exp_type_id = exp_type->type_id();

  return {PowCheckAndInferType(input_type_id, exp_type_id, x_tensor->Dtype(), input_values[kInputIndex1])};
}
ShapeArray PowTensorScalarFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  return {x_tensor->shape()};
}
REGISTER_SIMPLE_INFER(kNamePowTensorScalar, PowTensorScalarFuncImpl)
}  // namespace mindspore::ops
