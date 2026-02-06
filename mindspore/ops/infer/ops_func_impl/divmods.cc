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

#include "infer/ops_func_impl/divmods.h"

#include "mindapi/base/type_id.h"
#include "ops_utils/op_utils.h"

namespace mindspore::ops {
namespace divmods_internal {
constexpr int kTypeLevelBool = 0;
constexpr int kTypeLevelInt = 1;
constexpr int kTypeLevelFloat = 2;
constexpr int kTypeLevelComplex = 3;

static inline bool IsBoolType(TypeId t) { return t == kNumberTypeBool; }
static inline bool IsComplexType(TypeId t) {
  return t == kNumberTypeComplex || t == kNumberTypeComplex64 || t == kNumberTypeComplex128;
}

static inline bool IsIntegralType(TypeId t) {
  return t == kNumberTypeInt8 || t == kNumberTypeInt16 || t == kNumberTypeInt32 || t == kNumberTypeInt64 ||
         t == kNumberTypeUInt8 || t == kNumberTypeUInt16 || t == kNumberTypeUInt32 || t == kNumberTypeUInt64;
}

static inline bool IsFloatingType(TypeId t) {
  return t == kNumberTypeFloat || t == kNumberTypeFloat16 || t == kNumberTypeFloat32 || t == kNumberTypeFloat64 ||
         t == kNumberTypeBFloat16 || t == kNumberTypeDouble;
}

static inline int TypeToLevel(TypeId t) {
  if (IsBoolType(t)) {
    return kTypeLevelBool;
  } else if (IsIntegralType(t)) {
    return kTypeLevelInt;
  } else if (IsFloatingType(t)) {
    return kTypeLevelFloat;
  } else if (IsComplexType(t)) {
    return kTypeLevelComplex;
  } else {
    MS_EXCEPTION(TypeError) << "For DivMods, unsupported dtype: " << TypeIdToString(t);
  }
}
}  // namespace divmods_internal

ShapeArray DivModsFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  return {input_infos[kIndex0]->GetShape()};
}

std::vector<TypeId> DivModsFuncImpl::InferType(const PrimitivePtr &primitive,
                                               const InferInfoPtrList &input_infos) const {
  auto input_type = input_infos[kIndex0]->GetType();
  auto other_type = input_infos[kIndex1]->GetType();
  auto promote_type =
    (divmods_internal::TypeToLevel(input_type) < divmods_internal::TypeToLevel(other_type)) ? other_type : input_type;

  // If rounding_mode is None (true division), integers promote to float.
  if (input_infos[kIndex2]->IsNone()) {
    if (divmods_internal::IsIntegralType(promote_type) || divmods_internal::IsBoolType(promote_type)) {
      return {kNumberTypeFloat32};
    }
  }
  return {promote_type};
}
}  // namespace mindspore::ops
