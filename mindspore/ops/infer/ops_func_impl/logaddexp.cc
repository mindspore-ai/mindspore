/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/logaddexp.h"

#include <vector>
#include <memory>
#include <set>
#include <map>
#include <string>
#include "primitive/op_name.h"
#include "utils/check_convert_utils.h"
#include "utils/shape_utils.h"
#include "utils/log_adapter.h"
#include "ir/primitive.h"
#include "abstract/dshape.h"
#include "base/base.h"
#include "ops_utils/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"

namespace mindspore::ops {
static inline void IsValidLogAddExpType(const std::string &type_name, const TypeId &t, const TypePtr &type) {
  static const std::set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeBFloat16};
  if (valid_types.find(t) == valid_types.end()) {
    MS_EXCEPTION(TypeError) << "For primitive[LogAddExp], the input argument [" << type_name
                            << "] must be a type of {Float16, Float32, BFloat16}, but got " << type << ".";
  }
}

BaseShapePtr LogAddExpFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  return BroadCastInferShape(primitive->name(), input_args);
}

TypePtr LogAddExpFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  std::map<std::string, TypePtr> types;
  (void)types.emplace("input", input_args[kInputIndex0]->GetType());
  (void)types.emplace("other", input_args[kInputIndex1]->GetType());
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kBFloat16};
  return CheckAndConvertUtils::CheckMathBinaryOpTensorType(types, valid_types, primitive->name());
}

ShapeArray LogAddExpFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  return {BroadCastInferShape(primitive->name(), input_values)};
}

TypePtrList LogAddExpFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &input_tensor = input_values[kInputIndex0]->cast<tensor::TensorPtr>();
  const auto &other_tensor = input_values[kInputIndex1]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  MS_EXCEPTION_IF_NULL(other_tensor);
  const auto &input_type = input_tensor->Dtype();
  const auto &other_type = other_tensor->Dtype();
  const auto &input_type_id = input_tensor->Dtype()->type_id();
  const auto &other_type_id = other_tensor->Dtype()->type_id();
  IsValidLogAddExpType("input", input_type_id, input_type);
  IsValidLogAddExpType("other", other_type_id, other_type);
  return {input_type};
}

REGISTER_SIMPLE_INFER(kNameLogAddExp, LogAddExpFuncImpl)
}  // namespace mindspore::ops
