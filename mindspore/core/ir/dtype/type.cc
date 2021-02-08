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

#include "ir/dtype/number.h"
#include "utils/log_adapter.h"
#include "utils/convert_utils_base.h"

namespace mindspore {
TypeId IntBitsToTypeId(const int nbits) {
  switch (nbits) {
    case 8:
      return kNumberTypeInt8;
    case 16:
      return kNumberTypeInt16;
    case 32:
      return kNumberTypeInt32;
    case 64:
      return kNumberTypeInt64;
    default:
      MS_LOG(EXCEPTION) << "Wrong number of bits.";
  }
}

TypeId UIntBitsToTypeId(const int nbits) {
  switch (nbits) {
    case 8:
      return kNumberTypeUInt8;
    case 16:
      return kNumberTypeUInt16;
    case 32:
      return kNumberTypeUInt32;
    case 64:
      return kNumberTypeUInt64;
    default:
      MS_LOG(EXCEPTION) << "Wrong number of bits.";
  }
}

TypeId FloatBitsToTypeId(const int nbits) {
  switch (nbits) {
    case 16:
      return kNumberTypeFloat16;
    case 32:
      return kNumberTypeFloat32;
    case 64:
      return kNumberTypeFloat64;
    default:
      MS_LOG(EXCEPTION) << "Wrong number of bits.";
  }
}

const char *MetaIdLabel(const TypeId &v) {
  switch (v) {
    case kTypeUnknown:
      return "kTypeUnknown";
    case kMetaTypeType:
      return "kMetaTypeType";
    case kMetaTypeAnything:
      return "kMetaTypeAnything";
    case kMetaTypeObject:
      return "kMetaTypeObject";
    case kMetaTypeTypeType:
      return "kMetaTypeTypeType";
    case kMetaTypeProblem:
      return "kMetaTypeProblem";
    case kMetaTypeExternal:
      return "kMetaTypeExternal";
    case kMetaTypeNone:
      return "kMetaTypeNone";
    case kMetaTypeNull:
      return "kMetaTypeNull";
    case kMetaTypeEllipsis:
      return "kMetaTypeEllipsis";
    case kMetaTypeEnd:
      return "kMetaTypeEnd";
    default:
      return "[Unknown Type Id]";
  }
}

const char *ObjectIdLabel(const TypeId &v) {
  switch (v) {
    case kObjectTypeNumber:
      return "kObjectTypeNumber";
    case kObjectTypeString:
      return "kObjectTypeString";
    case kObjectTypeList:
      return "kObjectTypeList";
    case kObjectTypeTuple:
      return "kObjectTypeTuple";
    case kObjectTypeSlice:
      return "kObjectTypeSlice";
    case kObjectTypeKeyword:
      return "kObjectTypeKeyword";
    case kObjectTypeTensorType:
      return "kObjectTypeTensorType";
    case kObjectTypeRowTensorType:
      return "kObjectTypeRowTensorType";
    case kObjectTypeSparseTensorType:
      return "kObjectTypeSparseTensorType";
    case kObjectTypeUndeterminedType:
      return "kObjectTypeUndeterminedType";
    case kObjectTypeDictionary:
      return "kObjectTypeDictionary";
    case kObjectTypeClass:
      return "kObjectTypeClass";
    case kObjectTypeFunction:
      return "kObjectTypeFunction";
    case kObjectTypeJTagged:
      return "kObjectTypeJTagged";
    case kObjectTypeSymbolicKeyType:
      return "kObjectTypeSymbolicKeyType";
    case kObjectTypeEnvType:
      return "kObjectTypeEnvType";
    case kObjectTypeRefKey:
      return "kObjectTypeRefKey";
    case kObjectTypeRef:
      return "kObjectTypeRef";
    case kObjectTypeMonad:
      return "kObjectTypeMonad";
    case kObjectTypeUMonad:
      return "kObjectTypeUMonad";
    case kObjectTypeIOMonad:
      return "kObjectTypeIOMonad";
    default:
      return "[Unknown Type Id]";
  }
}

const char *NumberIdLabel(const TypeId &v) {
  switch (v) {
    case kNumberTypeBool:
      return "kNumberTypeBool";
    case kNumberTypeInt:
      return "kNumberTypeInt";
    case kNumberTypeInt8:
      return "kNumberTypeInt8";
    case kNumberTypeInt16:
      return "kNumberTypeInt16";
    case kNumberTypeInt32:
      return "kNumberTypeInt32";
    case kNumberTypeInt64:
      return "kNumberTypeInt64";
    case kNumberTypeUInt:
      return "kNumberTypeUInt";
    case kNumberTypeUInt8:
      return "kNumberTypeUInt8";
    case kNumberTypeUInt16:
      return "kNumberTypeUInt16";
    case kNumberTypeUInt32:
      return "kNumberTypeUInt32";
    case kNumberTypeUInt64:
      return "kNumberTypeUInt64";
    case kNumberTypeFloat:
      return "kNumberTypeFloat";
    case kNumberTypeFloat16:
      return "kNumberTypeFloat16";
    case kNumberTypeFloat32:
      return "kNumberTypeFloat32";
    case kNumberTypeFloat64:
      return "kNumberTypeFloat64";
    default:
      return "[Unknown Type Id]";
  }
}

const char *TypeIdLabel(const TypeId &v) {
  if (v < kMetaTypeEnd) {
    return MetaIdLabel(v);
  } else {
    if (v < kObjectTypeEnd) {
      return ObjectIdLabel(v);
    } else if (v > kMonadTypeBegin && v < kMonadTypeEnd) {
      // Monad Types is ObjectType
      return ObjectIdLabel(v);
    } else {
      return NumberIdLabel(v);
    }
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
      return IntToSize(number->nbits() / 8);
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
