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

#include "ops/ops_func_impl/complex.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr ComplexFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  return BroadCastInferShape(primitive->name(), input_args);
}

TypePtr ComplexFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  // Valid types: kFloat32, kFloat64.
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex1]);
  std::map<std::string, TypePtr> types;
  auto real_input_type = input_args[kInputIndex0]->GetType();
  auto imag_input_type = input_args[kInputIndex1]->GetType();
  (void)types.emplace("real", real_input_type);
  (void)types.emplace("imag", imag_input_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, std::set<TypePtr>{kFloat32, kFloat64}, primitive->name());
  auto real_input_tensor = real_input_type->cast<TensorTypePtr>();
  TypeId real_input_tensor_id = real_input_tensor->element()->type_id();
  return real_input_tensor_id == kNumberTypeFloat32 ? std::make_shared<TensorType>(kComplex64)
                                                    : std::make_shared<TensorType>(kComplex128);
}
}  // namespace ops
}  // namespace mindspore
