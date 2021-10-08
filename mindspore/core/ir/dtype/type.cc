/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "ir/dtype/type.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <climits>

#include "ir/dtype/number.h"
#include "utils/log_adapter.h"
#include "utils/convert_utils_base.h"

namespace mindspore {
#define MS_TYPE2LABLE(type_id) #type_id
static std::unordered_map<TypeId, std::string> g_type_2_lable{
  {kTypeUnknown, MS_TYPE2LABLE(kTypeUnknown)},
  {kMetaTypeType, MS_TYPE2LABLE(kMetaTypeType)},
  {kMetaTypeAnything, MS_TYPE2LABLE(kMetaTypeAnything)},
  {kMetaTypeObject, MS_TYPE2LABLE(kMetaTypeObject)},
  {kMetaTypeTypeType, MS_TYPE2LABLE(kMetaTypeTypeType)},
  {kMetaTypeProblem, MS_TYPE2LABLE(kMetaTypeProblem)},
  {kMetaTypeExternal, MS_TYPE2LABLE(kMetaTypeExternal)},
  {kMetaTypeNone, MS_TYPE2LABLE(kMetaTypeNone)},
  {kMetaTypeNull, MS_TYPE2LABLE(kMetaTypeNull)},
  {kMetaTypeEllipsis, MS_TYPE2LABLE(kMetaTypeEllipsis)},
  {kMetaTypeEnd, MS_TYPE2LABLE(kMetaTypeEnd)},
  {kObjectTypeNumber, MS_TYPE2LABLE(kObjectTypeNumber)},
  {kObjectTypeString, MS_TYPE2LABLE(kObjectTypeString)},
  {kObjectTypeList, MS_TYPE2LABLE(kObjectTypeList)},
  {kObjectTypeTuple, MS_TYPE2LABLE(kObjectTypeTuple)},
  {kObjectTypeSlice, MS_TYPE2LABLE(kObjectTypeSlice)},
  {kObjectTypeKeyword, MS_TYPE2LABLE(kObjectTypeKeyword)},
  {kObjectTypeTensorType, MS_TYPE2LABLE(kObjectTypeTensorType)},
  {kObjectTypeRowTensorType, MS_TYPE2LABLE(kObjectTypeRowTensorType)},
  {kObjectTypeSparseTensorType, MS_TYPE2LABLE(kObjectTypeSparseTensorType)},
  {kObjectTypeUndeterminedType, MS_TYPE2LABLE(kObjectTypeUndeterminedType)},
  {kObjectTypeClass, MS_TYPE2LABLE(kObjectTypeClass)},
  {kObjectTypeDictionary, MS_TYPE2LABLE(kObjectTypeDictionary)},
  {kObjectTypeFunction, MS_TYPE2LABLE(kObjectTypeFunction)},
  {kObjectTypeJTagged, MS_TYPE2LABLE(kObjectTypeJTagged)},
  {kObjectTypeSymbolicKeyType, MS_TYPE2LABLE(kObjectTypeSymbolicKeyType)},
  {kObjectTypeEnvType, MS_TYPE2LABLE(kObjectTypeEnvType)},
  {kObjectTypeRefKey, MS_TYPE2LABLE(kObjectTypeRefKey)},
  {kObjectTypeRef, MS_TYPE2LABLE(kObjectTypeRef)},
  {kObjectTypeEnd, MS_TYPE2LABLE(kObjectTypeEnd)},
  {kNumberTypeBool, MS_TYPE2LABLE(kNumberTypeBool)},
  {kNumberTypeInt, MS_TYPE2LABLE(kNumberTypeInt)},
  {kNumberTypeInt8, MS_TYPE2LABLE(kNumberTypeInt8)},
  {kNumberTypeInt16, MS_TYPE2LABLE(kNumberTypeInt16)},
  {kNumberTypeInt32, MS_TYPE2LABLE(kNumberTypeInt32)},
  {kNumberTypeInt64, MS_TYPE2LABLE(kNumberTypeInt64)},
  {kNumberTypeUInt, MS_TYPE2LABLE(kNumberTypeUInt)},
  {kNumberTypeUInt8, MS_TYPE2LABLE(kNumberTypeUInt8)},
  {kNumberTypeUInt16, MS_TYPE2LABLE(kNumberTypeUInt16)},
  {kNumberTypeUInt32, MS_TYPE2LABLE(kNumberTypeUInt32)},
  {kNumberTypeUInt64, MS_TYPE2LABLE(kNumberTypeUInt64)},
  {kNumberTypeFloat, MS_TYPE2LABLE(kNumberTypeFloat)},
  {kNumberTypeFloat16, MS_TYPE2LABLE(kNumberTypeFloat16)},
  {kNumberTypeFloat32, MS_TYPE2LABLE(kNumberTypeFloat32)},
  {kNumberTypeFloat64, MS_TYPE2LABLE(kNumberTypeFloat64)},
  {kNumberTypeComplex64, MS_TYPE2LABLE(kNumberTypeComplex64)},
  {kNumberTypeEnd, MS_TYPE2LABLE(kNumberTypeEnd)},
  {kObjectTypeMonad, MS_TYPE2LABLE(kObjectTypeMonad)},
  {kObjectTypeUMonad, MS_TYPE2LABLE(kObjectTypeUMonad)},
  {kObjectTypeIOMonad, MS_TYPE2LABLE(kObjectTypeIOMonad)},
  {kMonadTypeEnd, MS_TYPE2LABLE(kMonadTypeEnd)}};

TypeId IntBitsToTypeId(const int nbits) {
  switch (nbits) {
    case static_cast<int>(BitsNum::eBits8):
      return kNumberTypeInt8;
    case static_cast<int>(BitsNum::eBits16):
      return kNumberTypeInt16;
    case static_cast<int>(BitsNum::eBits32):
      return kNumberTypeInt32;
    case static_cast<int>(BitsNum::eBits64):
      return kNumberTypeInt64;
    default:
      MS_LOG(EXCEPTION) << "Wrong number of bits:" << nbits;
  }
}

TypeId UIntBitsToTypeId(const int nbits) {
  switch (nbits) {
    case static_cast<int>(BitsNum::eBits8):
      return kNumberTypeUInt8;
    case static_cast<int>(BitsNum::eBits16):
      return kNumberTypeUInt16;
    case static_cast<int>(BitsNum::eBits32):
      return kNumberTypeUInt32;
    case static_cast<int>(BitsNum::eBits64):
      return kNumberTypeUInt64;
    default:
      MS_LOG(EXCEPTION) << "Wrong number of bits:" << nbits;
  }
}

TypeId FloatBitsToTypeId(const int nbits) {
  switch (nbits) {
    case static_cast<int>(BitsNum::eBits16):
      return kNumberTypeFloat16;
    case static_cast<int>(BitsNum::eBits32):
      return kNumberTypeFloat32;
    case static_cast<int>(BitsNum::eBits64):
      return kNumberTypeFloat64;
    default:
      MS_LOG(EXCEPTION) << "Wrong number of bits:" << nbits;
  }
}

TypeId ComplexBitsToTypeId(const int nbits) {
  switch (nbits) {
    case static_cast<int>(BitsNum::eBits64):
      return kNumberTypeComplex64;
    case static_cast<int>(BitsNum::eBits128):
      return kNumberTypeComplex128;
    default:
      MS_LOG(EXCEPTION) << "Wrong number of bits:" << nbits;
  }
}

const std::string &TypeIdLabel(const TypeId &v) {
  static const std::string unknown("[Unknown Type Id]");
  auto iter = g_type_2_lable.find(v);
  if (iter != g_type_2_lable.end()) {
    return iter->second;
  } else {
    return unknown;
  }
}

TypeId NormalizeTypeId(const TypeId type_id) {
  if ((type_id == kNumberTypeInt) || (type_id == kNumberTypeInt8) || (type_id == kNumberTypeInt16) ||
      (type_id == kNumberTypeInt32) || (type_id == kNumberTypeInt64)) {
    return kNumberTypeInt;
  } else if ((type_id == kNumberTypeFloat) || (type_id == kNumberTypeFloat16) || (type_id == kNumberTypeFloat32) ||
             (type_id == kNumberTypeFloat64)) {
    return kNumberTypeFloat;
  } else {
    return type_id;
  }
}

bool IsSameObjectType(const Type &lhs, const Type &rhs) {
  if ((lhs.meta_type() != kMetaTypeObject) || (rhs.meta_type() != kMetaTypeObject)) {
    return false;
  }
  return lhs.object_type() == rhs.object_type();
}

size_t GetTypeByte(const TypePtr &type_ptr) {
  if (type_ptr && type_ptr->isa<Number>()) {
    auto number = dyn_cast<Number>(type_ptr);
    if (!number) {
      MS_LOG(DEBUG) << "Invalid TypePtr got from ApplyKernel.";
      return 0;
    } else {
      return IntToSize(number->nbits() / CHAR_BIT);
    }
  } else {
    MS_LOG(DEBUG) << "Invalid TypePtr got from ApplyKernel.";
    return 0;
  }
}

bool Type::operator==(const Value &other) const {
  if (other.isa<Type>()) {
    auto other_type = static_cast<const Type *>(&other);
    return *this == *other_type;
  } else {
    return false;
  }
}

std::ostream &operator<<(std::ostream &os, const Type &type) {
  os << type.ToString();
  return os;
}

std::ostream &operator<<(std::ostream &os, const TypePtr type) {
  os << type->ToString();
  return os;
}

bool Object::equal(const TypePtr other) const {
  auto same_other = dyn_cast<Object>(other);
  if (same_other != nullptr) {
    return *this == *same_other;
  }
  return false;
}

std::ostream &operator<<(std::ostream &os, const Object &obj) {
  os << obj.ToString();
  return os;
}

std::ostream &operator<<(std::ostream &os, const std::shared_ptr<Object> obj) {
  os << obj->ToString();
  return os;
}

std::ostream &operator<<(std::ostream &os, const TypePtrList &types) {
  os << "[";
  for (size_t i = 0; i < types.size(); ++i) {
    if (i > 0) {
      os << ", ";
    }
    os << (types[i] == nullptr ? "nullptr" : types[i]->ToString());
  }
  os << "]";
  return os;
}
}  // namespace mindspore
