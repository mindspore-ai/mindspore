/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include <climits>

#include "ir/dtype/number.h"
#include "utils/log_adapter.h"
#include "utils/convert_utils_base.h"

namespace mindspore {
static mindspore::HashMap<TypeId, std::string> g_type_2_lable{{kTypeUnknown, "Unknown"},
                                                              {kMetaTypeType, "Type"},
                                                              {kMetaTypeAny, "Any"},
                                                              {kMetaTypeObject, "Object"},
                                                              {kMetaTypeTypeType, "TypeType"},
                                                              {kMetaTypeProblem, "Problem"},
                                                              {kMetaTypeExternal, "External"},
                                                              {kMetaTypeNone, "None"},
                                                              {kMetaTypeNull, "Null"},
                                                              {kMetaTypeEllipsis, "Ellipsis"},
                                                              {kObjectTypeNumber, "Number"},
                                                              {kObjectTypeString, "String"},
                                                              {kObjectTypeList, "List"},
                                                              {kObjectTypeTuple, "Tuple"},
                                                              {kObjectTypeSlice, "Slice"},
                                                              {kObjectTypeKeyword, "Keyword"},
                                                              {kObjectTypeTensorType, "Tensor"},
                                                              {kObjectTypeMapTensorType, "MapTensor"},
                                                              {kObjectTypeRowTensorType, "RowTensor"},
                                                              {kObjectTypeCOOTensorType, "COOTensor"},
                                                              {kObjectTypeCSRTensorType, "CSRTensor"},
                                                              {kObjectTypeUndeterminedType, "Undetermined"},
                                                              {kObjectTypeClass, "Class"},
                                                              {kObjectTypeDictionary, "Dictionary"},
                                                              {kObjectTypeFunction, "Function"},
                                                              {kObjectTypeJTagged, "JTagged"},
                                                              {kObjectTypeSymbolicKeyType, "SymbolicKey"},
                                                              {kObjectTypeEnvType, "EnvType"},
                                                              {kObjectTypeRefKey, "RefKey"},
                                                              {kObjectTypeRef, "Ref"},
                                                              {kNumberTypeBool, "Bool"},
                                                              {kNumberTypeInt, "Int"},
                                                              {kNumberTypeInt8, "Int8"},
                                                              {kNumberTypeInt16, "Int16"},
                                                              {kNumberTypeInt32, "Int32"},
                                                              {kNumberTypeInt64, "Int64"},
                                                              {kNumberTypeUInt, "UInt"},
                                                              {kNumberTypeUInt8, "UInt8"},
                                                              {kNumberTypeUInt16, "UInt16"},
                                                              {kNumberTypeUInt32, "UInt32"},
                                                              {kNumberTypeUInt64, "UInt64"},
                                                              {kNumberTypeFloat, "Float"},
                                                              {kNumberTypeFloat16, "Float16"},
                                                              {kNumberTypeFloat32, "Float32"},
                                                              {kNumberTypeFloat64, "Float64"},
                                                              {kNumberTypeComplex, "Complex"},
                                                              {kNumberTypeComplex64, "Complex64"},
                                                              {kNumberTypeComplex128, "Complex128"},
                                                              {kNumberTypeInt4, "Int4"},
                                                              {kNumberTypeGLUInt, "GLUInt"},
                                                              {kObjectTypeMonad, "Monad"},
                                                              {kObjectTypeUMonad, "UMonad"},
                                                              {kObjectTypeIOMonad, "IOMonad"}};

const mindspore::HashMap<TypeId, int> &type_priority_map() {
  static const mindspore::HashMap<TypeId, int> type_priority_map = {
    {kNumberTypeBool, 0},    {kNumberTypeUInt8, 1},   {kNumberTypeInt8, 2},
    {kNumberTypeInt16, 3},   {kNumberTypeInt32, 4},   {kNumberTypeInt64, 5},
    {kNumberTypeFloat16, 6}, {kNumberTypeFloat32, 7}, {kNumberTypeFloat64, 8}};
  return type_priority_map;
}

const mindspore::HashMap<TypeId, std::string> &type_name_map() {
  static const mindspore::HashMap<TypeId, std::string> type_name_map = {
    {kNumberTypeBool, "bool_"},      {kNumberTypeInt8, "int8"},       {kNumberTypeUInt8, "uint8"},
    {kNumberTypeInt16, "int16"},     {kNumberTypeInt32, "int32"},     {kNumberTypeInt64, "int64"},
    {kNumberTypeFloat16, "float16"}, {kNumberTypeFloat32, "float32"}, {kNumberTypeFloat64, "float64"}};
  return type_name_map;
}

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
      MS_LOG(EXCEPTION) << "For Int type only support number of 8bits, 16bits, 32bits and 64bits, but got " << nbits
                        << "bits";
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
      MS_LOG(EXCEPTION) << "For UInt type only support number of 8bits, 16bits, 32bits and 64bits, but got " << nbits
                        << "bits";
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
      MS_LOG(EXCEPTION) << "For Float type only support number of 16bits, 32bits and 64bits, but got " << nbits
                        << "bits";
  }
}

TypeId ComplexBitsToTypeId(const int nbits) {
  switch (nbits) {
    case static_cast<int>(BitsNum::eBits64):
      return kNumberTypeComplex64;
    case static_cast<int>(BitsNum::eBits128):
      return kNumberTypeComplex128;
    default:
      MS_LOG(EXCEPTION) << "For Complex type only support number of 64bits and 128bits, but got " << nbits << "bits";
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
  if (!other.isa<Type>()) {
    return false;
  }
  auto other_type = static_cast<const Type *>(&other);
  return *this == *other_type;
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
