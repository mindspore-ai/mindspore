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

#include "ops/ops_func_impl/binary_ext_op.h"
#include <vector>
#include <map>
#include <string>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore::ops {
static inline bool isIntegralType(TypeId t) {
  return t == kNumberTypeInt8 || t == kNumberTypeInt16 || t == kNumberTypeInt32 || t == kNumberTypeInt64 ||
         t == kNumberTypeUInt8 || t == kNumberTypeUInt16 || t == kNumberTypeUInt32 || t == kNumberTypeUInt64;
}

TypePtr BinaryExtOpFuncImpl::InferType(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]->GetType());
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[kInputIndex0]->GetType());
  (void)types.emplace("y", input_args[kInputIndex1]->GetType());

  auto dtype1 = input_args[kInputIndex0]->GetType();
  auto dtype2 = input_args[kInputIndex1]->GetType();

  auto element1 = dtype1->cast<TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(element1);
  auto type_id1 = element1->type_id();

  auto element2 = dtype2->cast<TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(element2);
  auto type_id2 = element2->type_id();

  auto alpha_type = input_args[kInputIndex2]->GetType();

  if (alpha_type == kFloat32 && (isIntegralType(type_id1) || isIntegralType(type_id2))) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', floating alpha need floating input and other, but got " << dtype1->ToString()
                             << " and " << dtype2->ToString();
  }

  if (alpha_type == kBool && (type_id1 != kNumberTypeBool || type_id2 != kNumberTypeBool)) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', boolean alpha need boolean input and other, but got " << dtype1->ToString()
                             << " and " << dtype2->ToString();
  }

  return CheckAndConvertUtils::CheckMathBinaryOpTensorType(types, common_valid_types_with_complex_and_bool,
                                                           primitive->name());
}
}  // namespace mindspore::ops
