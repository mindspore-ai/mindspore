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

#include "infer/ops_func_impl/sign.h"
#include <set>
#include <string>
#include "ops/ops_func_impl/simple_infer.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace ops {
namespace {
void CheckSignValidTypes(const PrimitivePtr &primitive, const TypeId &type_id) {
  const std::set<TypeId> valid_types_set = {kNumberTypeBool,      kNumberTypeInt32,      kNumberTypeInt64,
                                            kNumberTypeFloat16,   kNumberTypeFloat32,    kNumberTypeFloat64,
                                            kNumberTypeComplex64, kNumberTypeComplex128, kNumberTypeBFloat16};
  if (MS_UNLIKELY(valid_types_set.find(type_id) == valid_types_set.end())) {
    std::string valid_types_str;
    std::string spot = ", ";
    for (const auto &type : valid_types_set) valid_types_str += (TypeIdToString(type) + spot);
    valid_types_str.erase(valid_types_str.size() - spot.size());
    MS_EXCEPTION(TypeError) << "For Primitive " << primitive->name() << ", the type of input must be in {"
                            << valid_types_str << "}, but got " << TypeIdToString(type_id) << ".";
  }
}
}  // namespace

BaseShapePtr SignFuncImpl::InferShape(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[0]->GetShape()->Clone();
}

TypePtr SignFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto x_type = input_args[0]->GetType();
  auto x_tensor_type = x_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(x_tensor_type);
  CheckSignValidTypes(primitive, x_tensor_type->element()->type_id());
  return x_type;
}

ShapeArray SignFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  return {x_tensor->shape()};
}

TypePtrList SignFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto &x_tensor_type = x_tensor->Dtype();
  CheckSignValidTypes(primitive, x_tensor_type->type_id());
  return {x_tensor_type};
}
REGISTER_SIMPLE_INFER(kNameSign, SignFuncImpl)
}  // namespace ops
}  // namespace mindspore
