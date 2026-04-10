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

#include "infer/ops_func_impl/clamp_scalar.h"
#include <vector>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore::ops {
namespace {
constexpr int kTypeLevelNone = -1;
constexpr int kTypeLevelBool = 0;
constexpr int kTypeLevelInt = 1;
constexpr int kTypeLevelFloat = 2;
constexpr int kTypeLevelComplex = 3;

bool IsIntegralType(TypeId t) {
  return t == kNumberTypeInt8 || t == kNumberTypeInt16 || t == kNumberTypeInt32 || t == kNumberTypeInt64 ||
         t == kNumberTypeUInt8 || t == kNumberTypeUInt16 || t == kNumberTypeUInt32 || t == kNumberTypeUInt64;
}

bool IsFloatingType(TypeId t) {
  return t == kNumberTypeBFloat16 || t == kNumberTypeFloat16 || t == kNumberTypeFloat32 || t == kNumberTypeFloat64;
}

int TypeToLevel(TypeId t) {
  if (t == kMetaTypeNone) {
    return kTypeLevelNone;
  } else if (t == kNumberTypeBool) {
    return kTypeLevelBool;
  } else if (IsIntegralType(t)) {
    return kTypeLevelInt;
  } else if (IsFloatingType(t)) {
    return kTypeLevelFloat;
  }
  return kTypeLevelComplex;
}
}  // namespace

TypeId ClampScalarPromoteType(TypeId input_type, TypeId min_type, TypeId max_type) {
  const auto scalar_type = (TypeToLevel(min_type) < TypeToLevel(max_type)) ? max_type : min_type;
  return (TypeToLevel(input_type) < TypeToLevel(scalar_type)) ? scalar_type : input_type;
}

ShapeArray ClampScalarFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  return {input_infos[kInputIndex0]->GetShape()};
}

std::vector<TypeId> ClampScalarFuncImpl::InferType(const PrimitivePtr &primitive,
                                                   const InferInfoPtrList &input_infos) const {
  TypeId input0_type = input_infos[kInputIndex0]->GetType();
  TypeId input1_type = input_infos[kInputIndex1]->IsNone() ? kMetaTypeNone : input_infos[kInputIndex1]->GetType();
  TypeId input2_type = input_infos[kInputIndex2]->IsNone() ? kMetaTypeNone : input_infos[kInputIndex2]->GetType();

  if (input1_type == kMetaTypeNone && input2_type == kMetaTypeNone) {
    MS_EXCEPTION(ValueError) << "For Clamp, at least one of 'min' or 'max' must not be None.";
  }

  if (input0_type == kNumberTypeBool || input1_type == kNumberTypeBool || input2_type == kNumberTypeBool) {
    MS_EXCEPTION(ValueError) << "For Clamp, the dtype of 'input', 'min' and 'max' must not be bool.";
  }

  return {ClampScalarPromoteType(input0_type, input1_type, input2_type)};
}
}  // namespace mindspore::ops
