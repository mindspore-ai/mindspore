/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "ir/dtype.h"
#include <string>
#include <cstdlib>
#include <algorithm>
#include "utils/log_adapter.h"
#include "abstract/abstract_value.h"

namespace mindspore {
TypePtr TypeAnything::DeepCopy() const { return kAnyType; }

std::size_t TypeHasher::operator()(TypePtr const &type) const {
  MS_EXCEPTION_IF_NULL(type);
  std::size_t hash = std::hash<size_t>()(type->type_id());
  return hash;
}

std::size_t TypeListHasher::operator()(const TypePtrList &type_list) const {
  std::size_t hash_sum = 0;
  for (auto &type : type_list) {
    auto type_id = static_cast<std::size_t>(type->type_id());
    hash_sum = hash_combine(hash_sum, type_id);
  }
  return hash_sum;
}

bool TypeEqual::operator()(TypePtr const &t1, TypePtr const &t2) const {
  MS_EXCEPTION_IF_NULL(t1);
  MS_EXCEPTION_IF_NULL(t2);
  return t1->type_id() == t2->type_id();
}

bool TypeListEqual::operator()(TypePtrList const &lhs, TypePtrList const &rhs) const {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  std::size_t size = lhs.size();
  for (std::size_t i = 0; i < size; ++i) {
    MS_EXCEPTION_IF_NULL(lhs[i]);
    MS_EXCEPTION_IF_NULL(rhs[i]);
    if (*lhs[i] != *rhs[i]) {
      return false;
    }
  }
  return true;
}

TypePtr TypeIdToType(TypeId id) {
  switch (id) {
    case kNumberTypeFloat16:
      return kFloat16;
    case kNumberTypeFloat:
    case kNumberTypeFloat32:
      return kFloat32;
    case kNumberTypeFloat64:
      return kFloat64;
    case kNumberTypeComplex64:
      return kComplex64;
    case kNumberTypeInt8:
      return kInt8;
    case kNumberTypeInt16:
      return kInt16;
    case kNumberTypeInt32:
      return kInt32;
    case kNumberTypeInt:
      return kInt32;
    case kNumberTypeInt64:
      return kInt64;
    case kNumberTypeUInt8:
      return kUInt8;
    case kNumberTypeUInt16:
      return kUInt16;
    case kNumberTypeUInt32:
      return kUInt32;
    case kNumberTypeUInt64:
      return kUInt64;
    case kNumberTypeBool:
      return kBool;
    case kMetaTypeExternal:
      return kTypeExternal;
    case kMetaTypeAnything:
      return kAnyType;
    case kMetaTypeNone:
      return kTypeNone;
    case kMetaTypeNull:
      return kTypeNull;
    case kMetaTypeEllipsis:
      return kTypeEllipsis;
    case kObjectTypeEnvType:
      return kTypeEnv;
    case kObjectTypeRefKey:
      return kRefKeyType;
    case kObjectTypeRef:
      return kRefType;
    case kMetaTypeTypeType:
      return kTypeType;
    case kObjectTypeString:
      return kString;
    case kObjectTypeList:
      return kList;
    case kObjectTypeTuple:
      return kTuple;
    case kObjectTypeDictionary:
      return kDict;
    case kObjectTypeSlice:
      return kSlice;
    case kObjectTypeKeyword:
      return kKeyword;
    case kObjectTypeTensorType:
      return kTensorType;
    case kObjectTypeUMonad:
      return kUMonadType;
    case kObjectTypeIOMonad:
      return kIOMonadType;
    case kTypeUnknown:
    case kMetaTypeProblem:
      return kTypeNone;
    default:
      MS_LOG(EXCEPTION) << "Not support the type: " << id;
  }
}

namespace {
template <typename T>
TypePtr StringToNumberType(const std::string &type_name, const std::string &num_type_name) {
  TypePtr type = nullptr;
  if (type_name == num_type_name) {
    type = std::make_shared<T>();
  } else {
    try {
      if (num_type_name.size() >= type_name.size()) {
        MS_LOG(EXCEPTION) << "Convert type is error, type_name(" << type_name << "), num_type_name(" << num_type_name
                          << ")";
      }
      auto bits = std::stoi(type_name.substr(num_type_name.size()));
      type = std::make_shared<T>(bits);
    } catch (const std::exception &e) {
      MS_LOG(EXCEPTION) << num_type_name << " convert from string error " << e.what();
    }
  }
  return type;
}

std::vector<TypePtr> StringToVectorOfType(const std::string &type_names) {
  std::vector<TypePtr> types;
  if (type_names.length() == 0) {
    return types;
  }
  std::string::size_type start = 0;
  std::string::size_type end = type_names.find_first_of(',');
  while (end != std::string::npos) {
    types.push_back(StringToType(type_names.substr(start, end)));
    // Skip ',' to find the next element.
    start = end + 1;
    end = type_names.find_first_of(',', start);
  }
  if (start >= type_names.size()) {
    MS_LOG(EXCEPTION) << "Type name is empty string.";
  }
  types.push_back(StringToType(type_names.substr(start)));
  return types;
}

TypePtr TensorStrToType(const std::string &type_name) {
  TypePtr type = nullptr;
  if (type_name == "Tensor") {
    type = std::make_shared<TensorType>();
  } else {
    try {
      auto start = type_name.find_first_of('[') + 1;
      auto end = type_name.find_last_of(']');
      if (start >= type_name.size()) {
        return nullptr;
      }
      auto element_str = type_name.substr(start, end - start);
      auto element_type = StringToType(element_str);
      if (element_type == nullptr) {
        return nullptr;
      }
      type = std::make_shared<TensorType>(element_type);
    } catch (const std::exception &e) {
      MS_LOG(EXCEPTION) << type_name << " convert from string error " << e.what();
    }
  }

  return type;
}

TypePtr RowTensorStrToType(const std::string &type_name) {
  if (type_name == "RowTensor") {
    return std::make_shared<RowTensorType>();
  }
  auto start = type_name.find_first_of('[') + 1;
  auto end = type_name.find_last_of(']');
  if (start >= type_name.size()) {
    return nullptr;
  }
  auto element_str = type_name.substr(start, end - start);
  auto element_type = StringToType(element_str);
  if (element_type == nullptr) {
    return nullptr;
  }
  return std::make_shared<RowTensorType>(element_type);
}

TypePtr SparseTensorStrToType(const std::string &type_name) {
  if (type_name == "SparseTensor") {
    return std::make_shared<SparseTensorType>();
  }
  auto start = type_name.find_first_of('[') + 1;
  auto end = type_name.find_last_of(']');
  if (start >= type_name.size()) {
    return nullptr;
  }
  auto element_str = type_name.substr(start, end - start);
  auto element_type = StringToType(element_str);
  if (element_type == nullptr) {
    return nullptr;
  }
  return std::make_shared<SparseTensorType>(element_type);
}

TypePtr UndeterminedStrToType(const std::string &type_name) {
  if (type_name == "Undetermined") {
    return std::make_shared<UndeterminedType>();
  }
  auto start = type_name.find_first_of('[') + 1;
  auto end = type_name.find_last_of(']');
  if (start >= type_name.size()) {
    return nullptr;
  }
  auto element_str = type_name.substr(start, end - start);
  auto element_type = StringToType(element_str);
  if (element_type == nullptr) {
    return nullptr;
  }
  return std::make_shared<UndeterminedType>(element_type);
}

TypePtr ListStrToType(const std::string &type_name) {
  TypePtr type = nullptr;
  if (type_name == "List") {
    type = std::make_shared<List>();
  } else {
    try {
      auto start = type_name.find_first_of('[') + 1;
      auto end = type_name.find_last_of(']');
      if (start >= type_name.size()) {
        return nullptr;
      }
      std::string element_strs = type_name.substr(start, end - start);
      std::vector<TypePtr> element_types = StringToVectorOfType(element_strs);
      bool wrong =
        std::any_of(element_types.begin(), element_types.end(), [](const TypePtr &x) { return x == nullptr; });
      if (wrong) {
        return nullptr;
      }
      type = std::make_shared<List>(element_types);
    } catch (const std::exception &e) {
      MS_LOG(EXCEPTION) << type_name << " convert from string error " << e.what();
    }
  }

  return type;
}

TypePtr TupleStrToType(const std::string &type_name) {
  TypePtr type = nullptr;
  if (type_name == "Tuple") {
    type = std::make_shared<Tuple>();
  } else {
    try {
      size_t start = type_name.find_first_of('[') + 1;
      size_t end = type_name.find_last_of(']');
      if (start >= type_name.size()) {
        return nullptr;
      }
      std::string element_strs = type_name.substr(start, end - start);
      std::vector<TypePtr> element_types = StringToVectorOfType(element_strs);
      bool wrong =
        std::any_of(element_types.begin(), element_types.end(), [](const TypePtr &x) { return x == nullptr; });
      if (wrong) {
        return nullptr;
      }
      type = std::make_shared<Tuple>(element_types);
    } catch (const std::exception &e) {
      MS_LOG(EXCEPTION) << type_name << " convert from string error " << e.what();
    }
  }
  return type;
}

TypePtr FunctionStrToType(const std::string &type_name) {
  TypePtr type = nullptr;

  if (type_name == "Function") {
    type = std::make_shared<Function>();
  } else {
    try {
      // format: [(para1, para2, para3, ...) retval]
      size_t start = type_name.find_first_of('[') + 1;
      size_t end = type_name.find_last_of(']');
      if (start >= type_name.size()) {
        return nullptr;
      }
      std::string str_all = type_name.substr(start, end - start);
      size_t start_a = str_all.find_first_of('(') + 1;
      size_t end_a = str_all.find_last_of(')');
      if (start_a >= str_all.size()) {
        return nullptr;
      }
      std::string str_args = str_all.substr(start_a, end_a - start_a);
      // bypass " " between ")" and retval
      start = end_a + 2;
      if (start >= str_all.size()) {
        return nullptr;
      }
      std::string str_retval = str_all.substr(start);

      std::vector<TypePtr> args_type = StringToVectorOfType(str_args);
      TypePtr retval = StringToType(str_retval);
      bool wrong = std::any_of(args_type.begin(), args_type.end(), [](const TypePtr &x) { return x == nullptr; });
      if (retval == nullptr || wrong) {
        return nullptr;
      }
      type = std::make_shared<Function>(args_type, retval);
    } catch (const std::exception &e) {
      MS_LOG(EXCEPTION) << type_name << " convert from string error " << e.what();
    }
  }
  return type;
}
}  // namespace

TypePtr StringToType(const std::string &type_name) {
  TypePtr type = nullptr;
  if (type_name == "None") {
    type = std::make_shared<TypeNone>();
  } else if (type_name == "Ellipsis") {
    type = std::make_shared<TypeEllipsis>();
  } else if (type_name == "TypeType") {
    type = std::make_shared<TypeType>();
  } else if (type_name == "SymbolicKeyType") {
    type = std::make_shared<SymbolicKeyType>();
  } else if (type_name == "RefKeyType") {
    type = std::make_shared<RefKeyType>();
  } else if (type_name == "EnvType") {
    type = std::make_shared<EnvType>();
  } else if (type_name == "Number") {
    type = std::make_shared<Number>();
  } else if (type_name == "Bool") {
    type = std::make_shared<Bool>();
  } else if (type_name.compare(0, strlen("Int"), "Int") == 0) {
    type = StringToNumberType<Int>(type_name, "Int");
  } else if (type_name.compare(0, strlen("UInt"), "UInt") == 0) {
    type = StringToNumberType<UInt>(type_name, "UInt");
  } else if (type_name.compare(0, strlen("Float"), "Float") == 0) {
    type = StringToNumberType<Float>(type_name, "Float");
  } else if (type_name.compare(0, strlen("Tensor"), "Tensor") == 0) {
    type = TensorStrToType(type_name);
  } else if (type_name.compare(0, strlen("Undetermined"), "Undetermined") == 0) {
    type = UndeterminedStrToType(type_name);
  } else if (type_name.compare(0, strlen("RowTensor"), "RowTensor") == 0) {
    type = RowTensorStrToType(type_name);
  } else if (type_name.compare(0, strlen("SparseTensor"), "SparseTensor") == 0) {
    type = SparseTensorStrToType(type_name);
  } else if (type_name.compare(0, strlen("List"), "List") == 0) {
    type = ListStrToType(type_name);
  } else if (type_name.compare(0, strlen("Tuple"), "Tuple") == 0) {
    type = TupleStrToType(type_name);
  } else if (type_name == "Slice") {
    type = std::make_shared<Slice>();
  } else if (type_name == "Dictionary") {
    type = std::make_shared<Dictionary>();
  } else if (type_name == "String") {
    type = std::make_shared<String>();
  } else if (type_name == "Problem") {
    type = std::make_shared<Problem>();
  } else if (type_name.compare(0, strlen("Function"), "Function") == 0) {
    type = FunctionStrToType(type_name);
  } else if (type_name == "mstype") {
    type = std::make_shared<TypeType>();
  } else if (type_name == "UMonad") {
    type = kUMonadType;
  } else if (type_name == "IOMonad") {
    type = kIOMonadType;
  } else {
    // - unsupported to convert
    // Class
    // SymbolicType
    // JTagged
    // Anything
    // External
    MS_LOG(EXCEPTION) << "Unsupported type name: " << type_name << "!";
  }
  return type;
}

bool IsParentOrChildrenType(TypePtr const &x, TypePtr const &base_type) {
  if (x == nullptr || base_type == nullptr) {
    MS_LOG(ERROR) << "Type is nullptr.";
    return false;
  }
  if (base_type->type_id() == kTypeUnknown || x->type_id() == kTypeUnknown) {
    return false;
  }
  return base_type->type_id() == x->parent_type() || x->type_id() == base_type->parent_type();
}

bool IsIdentidityOrSubclass(TypePtr const &x, TypePtr const &base_type) {
  if (x == nullptr || base_type == nullptr) {
    MS_LOG(ERROR) << "Type is nullptr.";
    return false;
  }
  if (base_type->type_id() == kTypeUnknown || x->type_id() == kTypeUnknown) {
    return false;
  } else if (!(base_type->IsGeneric())) {
    return *(base_type) == *(x);
  } else if (base_type->type_id() == x->type_id()) {
    return true;
  } else if (base_type->type_id() == x->generic_type_id()) {
    return true;
  } else if (base_type->type_id() == x->object_type()) {
    return true;
  } else if (base_type->type_id() == x->meta_type()) {
    return true;
  } else {
    return false;
  }
}

bool IsSubType(TypePtr const &t1, TypePtr const &t2) {
  MS_EXCEPTION_IF_NULL(t1);
  if (t1->type_id() == kTypeUnknown) {
    return false;
  } else if (t2 != nullptr) {
    return IsIdentidityOrSubclass(t1, t2);
  } else {
    return true;
  }
}

const TypePtr kTypeExternal = std::make_shared<External>();
const TypePtr kTypeEnv = std::make_shared<EnvType>();
const TypePtr kTypeType = std::make_shared<TypeType>();
const TypePtr kTensorType = std::make_shared<TensorType>();
const TypePtr kRowTensorType = std::make_shared<RowTensorType>();
const TypePtr kSparseTensorType = std::make_shared<SparseTensorType>();
const TypePtr kUndeterminedType = std::make_shared<UndeterminedType>();
const TypePtr kString = std::make_shared<String>();
const TypePtr kList = std::make_shared<List>();
const TypePtr kTuple = std::make_shared<Tuple>();
const TypePtr kDict = std::make_shared<Dictionary>();
const TypePtr kSlice = std::make_shared<Slice>();
const TypePtr kKeyword = std::make_shared<Keyword>();
}  // namespace mindspore
