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

#include "ops/ops_func_impl/sub_ext.h"
#include <vector>
#include <map>
#include <string>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore::ops {
static inline bool isIntegralType(TypePtr t) {
  return t == kInt8 || t == kInt16 || t == kInt32 || t == kInt64 || t == kUInt8 || t == kUInt16 || t == kUInt32 ||
         t == kUInt64;
}

TypePtr SubExtFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]->GetType());
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[kInputIndex0]->GetType());
  (void)types.emplace("y", input_args[kInputIndex1]->GetType());

  auto dtype = input_args[kInputIndex0]->GetType();
  auto alpha_type = input_args[kInputIndex2]->GetType();

  if (isIntegralType(dtype) && alpha_type == kFloat32) {
    MS_EXCEPTION(ValueError) << "For integral input tensors, argument alpha must not be a floating point number.";
  }

  return CheckAndConvertUtils::CheckMathBinaryOpTensorType(types, common_valid_types, primitive->name());
}
}  // namespace mindspore::ops
