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

#include "infer/ops_func_impl/divs.h"

#include <set>
#include <unordered_set>

#include "mindapi/base/type_id.h"
#include "ops_utils/op_utils.h"

namespace mindspore::ops {
namespace divs_internal {
constexpr int typeLevelBool = 0;
constexpr int typeLevelInt = 1;
constexpr int typeLevelFloat = 2;
constexpr int typeLevelComplex = 3;

inline bool IsBoolType(TypeId t) { return t == kNumberTypeBool; }
inline bool IsComplexType(TypeId t) { return t == kNumberTypeComplex64 || t == kNumberTypeComplex128; }

inline bool IsIntegralType(TypeId t) {
  return t == kNumberTypeInt8 || t == kNumberTypeInt16 || t == kNumberTypeInt32 || t == kNumberTypeInt64 ||
         t == kNumberTypeUInt8 || t == kNumberTypeUInt16 || t == kNumberTypeUInt32 || t == kNumberTypeUInt64;
}

inline bool IsFloatingType(TypeId t) {
  return t == kNumberTypeFloat16 || t == kNumberTypeFloat32 || t == kNumberTypeFloat64 || t == kNumberTypeBFloat16;
}

inline int TypeToLevel(TypeId t) {
  if (IsBoolType(t)) {
    return typeLevelBool;
  } else if (IsIntegralType(t)) {
    return typeLevelInt;
  } else if (IsFloatingType(t)) {
    return typeLevelFloat;
  } else {
    // Do not silently treat unsupported types as complex.
    if (IsComplexType(t)) {
      return typeLevelComplex;
    }
    MS_EXCEPTION(TypeError) << "For Divs, unsupported dtype: " << TypeIdToString(t);
  }
}
}  // namespace divs_internal

ShapeArray DivsFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  return {input_infos[kIndex0]->GetShape()};
}

std::vector<TypeId> DivsFuncImpl::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  TypeId input_type_id = input_infos[kInputIndex0]->GetType();
  TypeId other_type_id = input_infos[kInputIndex1]->GetType();

  auto promote_type_id = (divs_internal::TypeToLevel(input_type_id) < divs_internal::TypeToLevel(other_type_id))
                           ? other_type_id
                           : input_type_id;
  if (divs_internal::TypeToLevel(promote_type_id) < divs_internal::typeLevelFloat) {
    promote_type_id = kNumberTypeFloat32;
  }
  return {promote_type_id};
}
}  // namespace mindspore::ops
