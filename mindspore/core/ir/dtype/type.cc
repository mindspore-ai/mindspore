/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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
#include "utils/ms_utils.h"

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
                                                              {kNumberTypeInt4, "QInt4x2"},
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
                                                              {kNumberTypeBFloat16, "BFloat16"},
                                                              {kNumberTypeComplex, "Complex"},
                                                              {kNumberTypeComplex64, "Complex64"},
                                                              {kNumberTypeComplex128, "Complex128"},
                                                              {kNumberTypeGLUInt, "GLUInt"},
                                                              {kObjectTypeMonad, "Monad"},
                                                              {kObjectTypeUMonad, "UMonad"},
                                                              {kObjectTypeIOMonad, "IOMonad"},
                                                              {kObjectTypeEventType, "Event"}};

const mindspore::HashMap<TypeId, int> &type_priority_map() {
  static const mindspore::HashMap<TypeId, int> type_priority_map = {
    {kNumberTypeBool, 0},    {kNumberTypeUInt8, 1},   {kNumberTypeInt8, 2},    {kNumberTypeInt16, 3},
    {kNumberTypeInt32, 4},   {kNumberTypeInt64, 5},   {kNumberTypeFloat16, 6}, {kNumberTypeFloat32, 7},
    {kNumberTypeFloat64, 8}, {kNumberTypeBFloat16, 9}};
  return type_priority_map;
}

const mindspore::HashMap<TypeId, std::string> &type_name_map() {
  static const mindspore::HashMap<TypeId, std::string> type_name_map = {
    {kNumberTypeBool, "bool_"},        {kNumberTypeInt8, "int8"},       {kNumberTypeUInt8, "uint8"},
    {kNumberTypeInt16, "int16"},       {kNumberTypeInt32, "int32"},     {kNumberTypeInt64, "int64"},
    {kNumberTypeFloat16, "float16"},   {kNumberTypeFloat32, "float32"}, {kNumberTypeFloat64, "float64"},
    {kNumberTypeBFloat16, "bfloat16"}, {kNumberTypeInt4, "int4"}};
  return type_name_map;
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
  } else if (type_id == kNumberTypeBFloat16) {
    return kNumberTypeBFloat16;
  } else if ((type_id == kNumberTypeUInt) || (type_id == kNumberTypeUInt8) || (type_id == kNumberTypeUInt16) ||
             (type_id == kNumberTypeUInt32) || (type_id == kNumberTypeUInt64)) {
    return kNumberTypeUInt;
  } else if ((type_id == kNumberTypeComplex) || (type_id == kNumberTypeComplex64) ||
             (type_id == kNumberTypeComplex128)) {
    return kNumberTypeComplex;
  } else {
    return type_id;
  }
}

size_t GetTypeByte(const TypePtr &type_ptr) {
  if (type_ptr && type_ptr->isa<Number>()) {
    auto number = dyn_cast<Number>(type_ptr);
    if (!number) {
      MS_LOG(DEBUG) << "Invalid TypePtr got from ApplyKernel.";
      return 0;
    } else {
      if (number->nbits() < CHAR_BIT) {
        MS_LOG(DEBUG) << "Number of bit " << number->nbits() << " is less than CHAR_BIT " << CHAR_BIT << ", return 1.";
        return 1;
      }
      return IntToSize(number->nbits() / CHAR_BIT);
    }
  } else {
    MS_LOG(DEBUG) << "Invalid TypePtr got from ApplyKernel:" << (type_ptr == nullptr ? "null" : type_ptr->ToString());
    return 0;
  }
}

int64_t GetTypeId(const TypeId &type_id) { return static_cast<int64_t>(type_id); }

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

bool TypeListEqual::operator()(TypePtrList const &lhs, TypePtrList const &rhs) const {
  const auto size = lhs.size();
  if (size != rhs.size()) {
    return false;
  }
  for (std::size_t i = 0; i < size; ++i) {
    if (!common::IsEqual(lhs[i], rhs[i])) {
      return false;
    }
  }
  return true;
}
}  // namespace mindspore
