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
  static std::unordered_map<TypeId, TypePtr> type_id_to_type = {
    {kNumberTypeFloat16, kFloat16},     {kNumberTypeFloat, kFloat32},         {kNumberTypeFloat32, kFloat32},
    {kNumberTypeFloat64, kFloat64},     {kNumberTypeComplex64, kComplex64},   {kNumberTypeInt8, kInt8},
    {kNumberTypeInt16, kInt16},         {kNumberTypeInt32, kInt32},           {kNumberTypeInt, kInt32},
    {kNumberTypeInt64, kInt64},         {kNumberTypeUInt8, kUInt8},           {kNumberTypeUInt16, kUInt16},
    {kNumberTypeUInt32, kUInt32},       {kNumberTypeUInt64, kUInt64},         {kNumberTypeBool, kBool},
    {kNumberTypeComplex64, kComplex64}, {kNumberTypeComplex128, kComplex128}, {kMetaTypeExternal, kTypeExternal},
    {kMetaTypeAnything, kAnyType},      {kMetaTypeNone, kTypeNone},           {kMetaTypeNull, kTypeNull},
    {kMetaTypeEllipsis, kTypeEllipsis}, {kObjectTypeEnvType, kTypeEnv},       {kObjectTypeRefKey, kRefKeyType},
    {kObjectTypeRef, kRefType},         {kMetaTypeTypeType, kTypeType},       {kObjectTypeString, kString},
    {kObjectTypeList, kList},           {kObjectTypeTuple, kTuple},           {kObjectTypeDictionary, kDict},
    {kObjectTypeSlice, kSlice},         {kObjectTypeKeyword, kKeyword},       {kObjectTypeTensorType, kTensorType},
    {kObjectTypeUMonad, kUMonadType},   {kObjectTypeIOMonad, kIOMonadType},   {kTypeUnknown, kTypeNone},
    {kMetaTypeProblem, kTypeNone}};
  const auto &it = type_id_to_type.find(id);
  if (it == type_id_to_type.end()) {
    MS_LOG(EXCEPTION) << "Not support the type: " << id;
  }
  return it->second;
}

namespace {
template <typename T>
TypePtr StringToNumberType(const std::string &type_name, const std::string &num_type_name) {
  TypePtr type = nullptr;
  if (type_name == num_type_name) {
    type = std::make_shared<T>();
  } else {
    if (num_type_name.size() >= type_name.size()) {
      MS_LOG(EXCEPTION) << "Convert type is error, type_name(" << type_name << "), num_type_name(" << num_type_name
                        << ")";
    }
    auto bits = std::stoi(type_name.substr(num_type_name.size()));
    type = std::make_shared<T>(bits);
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
    auto start = type_name.find_first_of('[') + 1;
    auto end = type_name.find_last_of(']');
    if (start >= type_name.size()) {
      return nullptr;
    }
    std::string element_strs = type_name.substr(start, end - start);
    std::vector<TypePtr> element_types = StringToVectorOfType(element_strs);
    bool wrong = std::any_of(element_types.begin(), element_types.end(), [](const TypePtr &x) { return x == nullptr; });
    if (wrong) {
      return nullptr;
    }
    type = std::make_shared<List>(element_types);
  }

  return type;
}

TypePtr TupleStrToType(const std::string &type_name) {
  TypePtr type = nullptr;
  if (type_name == "Tuple") {
    type = std::make_shared<Tuple>();
  } else {
    size_t start = type_name.find_first_of('[') + 1;
    size_t end = type_name.find_last_of(']');
    if (start >= type_name.size()) {
      return nullptr;
    }
    std::string element_strs = type_name.substr(start, end - start);
    std::vector<TypePtr> element_types = StringToVectorOfType(element_strs);
    bool wrong = std::any_of(element_types.begin(), element_types.end(), [](const TypePtr &x) { return x == nullptr; });
    if (wrong) {
      return nullptr;
    }
    type = std::make_shared<Tuple>(element_types);
  }
  return type;
}

TypePtr FunctionStrToType(const std::string &type_name) {
  TypePtr type = nullptr;

  if (type_name == "Function") {
    type = std::make_shared<Function>();
  } else {
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
  }
  return type;
}
}  // namespace

TypePtr GetTypeByFullString(const std::string &type_name) {
  static std::map<std::string, TypePtr> type_map = {{"None", std::make_shared<TypeNone>()},
                                                    {"Ellipsis", std::make_shared<TypeEllipsis>()},
                                                    {"TypeType", std::make_shared<TypeType>()},
                                                    {"SymbolicKeyType", std::make_shared<SymbolicKeyType>()},
                                                    {"RefKeyType", std::make_shared<RefKeyType>()},
                                                    {"EnvType", std::make_shared<EnvType>()},
                                                    {"Number", std::make_shared<Number>()},
                                                    {"Bool", std::make_shared<Bool>()},
                                                    {"Slice", std::make_shared<Slice>()},
                                                    {"Dictionary", std::make_shared<Dictionary>()},
                                                    {"String", std::make_shared<String>()},
                                                    {"Problem", std::make_shared<Problem>()},
                                                    {"mstype", std::make_shared<TypeType>()},
                                                    {"UMonad", kUMonadType},
                                                    {"IOMonad", kIOMonadType}};

  auto iter = type_map.find(type_name);
  return iter == type_map.end() ? nullptr : iter->second;
}

TypePtr GetTypeByStringStarts(const std::string &type_name) {
  struct name_cmp {
    bool operator()(const std::string &l, const std::string &r) const {
      auto cmp_len = std::min(l.length(), r.length());
      return r.compare(0, cmp_len, l, 0, cmp_len) < 0;
    }
  };
  static std::map<std::string, std::function<TypePtr(const std::string &type_name)>, name_cmp> type_map = {
    {"Int", [](const std::string &type_name) -> TypePtr { return StringToNumberType<Int>(type_name, "Int"); }},
    {"UInt", [](const std::string &type_name) -> TypePtr { return StringToNumberType<UInt>(type_name, "UInt"); }},
    {"Float", [](const std::string &type_name) -> TypePtr { return StringToNumberType<Float>(type_name, "Float"); }},
    {"Tensor", [](const std::string &type_name) -> TypePtr { return TensorStrToType(type_name); }},
    {"Undetermined", [](const std::string &type_name) -> TypePtr { return UndeterminedStrToType(type_name); }},
    {"RowTensor", [](const std::string &type_name) -> TypePtr { return RowTensorStrToType(type_name); }},
    {"SparseTensor", [](const std::string &type_name) -> TypePtr { return SparseTensorStrToType(type_name); }},
    {"List", [](const std::string &type_name) -> TypePtr { return ListStrToType(type_name); }},
    {"Tuple", [](const std::string &type_name) -> TypePtr { return TupleStrToType(type_name); }},
    {"Function", [](const std::string &type_name) -> TypePtr { return FunctionStrToType(type_name); }}};
  auto iter = type_map.find(type_name);
  return iter == type_map.end() ? nullptr : iter->second(type_name);
}

TypePtr StringToType(const std::string &type_name) {
  auto type = GetTypeByFullString(type_name);
  if (type == nullptr) {
    type = GetTypeByStringStarts(type_name);
  }
  if (type == nullptr) {
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

bool IsIdentidityOrSubclass(TypePtr const &x, TypePtr const &base_type) {
  if (x == nullptr || base_type == nullptr) {
    MS_LOG(ERROR) << "Type is nullptr.";
    return false;
  }
  auto type_id = base_type->type_id();
  if (type_id == kTypeUnknown || x->type_id() == kTypeUnknown) {
    return false;
  } else if (!(base_type->IsGeneric())) {
    return *(base_type) == *(x);
  } else if (type_id == x->type_id() || type_id == x->generic_type_id() || type_id == x->object_type() ||
             type_id == x->meta_type()) {
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
}  // namespace mindspore
