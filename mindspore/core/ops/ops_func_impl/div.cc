/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "ops/ops_func_impl/div.h"
#include <set>
#include <map>
#include <limits>
#include <string>
#include "utils/check_convert_utils.h"
#include "ops/ops_frontend_func_impl.h"
#include "ops/op_utils.h"
#include "utils/ms_context.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
BaseShapePtr DivFuncImpl::InferShape(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  return BroadCastInferShape(primitive->name(), input_args);
}

TypePtr DivFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto x_dtype = input_args[kIndex0]->GetType();
  auto y_dtype = input_args[kIndex1]->GetType();
  static std::set<int> x_set = {kNumberTypeBool,   kNumberTypeUInt8,  kNumberTypeInt8,
                                kNumberTypeInt16,  kNumberTypeUInt16, kNumberTypeInt32,
                                kNumberTypeUInt32, kNumberTypeInt64,  kNumberTypeUInt64};
  auto input_type_id = x_dtype->cast<TensorTypePtr>()->element()->type_id();
  if (x_set.find(input_type_id) != x_set.end()) {
    return std::make_shared<TensorType>(kFloat32);
  }

  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", x_dtype);
  (void)types.emplace("y", y_dtype);
  return CheckAndConvertUtils::CheckMathBinaryOpTensorType(types, common_valid_types_with_complex_and_bool, prim_name);
}
TypePtrList DivFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto &x_dtype = x_tensor->Dtype();
  const auto &x_type_id = x_tensor->Dtype()->type_id();
  static const std::vector<TypeId> int_or_bool = {kNumberTypeUInt8,  kNumberTypeInt8,   kNumberTypeInt16,
                                                  kNumberTypeUInt16, kNumberTypeInt32,  kNumberTypeUInt32,
                                                  kNumberTypeInt64,  kNumberTypeUInt64, kNumberTypeBool};
  bool is_int_or_bool = std::any_of(int_or_bool.begin(), int_or_bool.end(),
                                    [&x_type_id](const TypeId &type_id) { return x_type_id == type_id; });
  if (is_int_or_bool) {
    return {kFloat32};
  } else {
    return {x_dtype};
  }
}
ShapeArray DivFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  return {BroadCastInferShape(primitive->name(), input_values)};
}
REGISTER_SIMPLE_INFER(kNameDiv, DivFuncImpl)
}  // namespace ops
}  // namespace mindspore
