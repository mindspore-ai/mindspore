/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include <cstdlib>
#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ir/dtype.h"
#include "mindapi/base/type_id.h"
#include "utils/log_adapter.h"
#include "base/base.h"
#include "include/robin_hood.h"
#include "ir/dtype/container.h"
#include "ir/dtype/empty.h"
#include "ir/dtype/monad_type.h"
#include "ir/dtype/number.h"
#include "ir/dtype/ref.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "utils/hash_map.h"

namespace mindspore {
TypePtr TypeAny::DeepCopy() const { return kTypeAny; }

std::string GetExcptionTypeString(TypeId id) {
  static mindspore::HashMap<TypeId, std::string> type_id_to_string = {{kMetaTypeType, "MetaType"},
                                                                      {kMetaTypeObject, "MetaTypeObject"},
                                                                      {kObjectTypeNumber, "Number"},
                                                                      {kObjectTypeRowTensorType, "RowTensor"},
                                                                      {kObjectTypeCOOTensorType, "COOTensor"},
                                                                      {kObjectTypeUndeterminedType, "Undetermined"},
                                                                      {kObjectTypeClass, "Class"},
                                                                      {kObjectTypeFunction, "Function"},
                                                                      {kObjectTypeJTagged, "JTagged"},
                                                                      {kObjectTypeSymbolicKeyType, "SymbolicKey"},
                                                                      {kNumberTypeUInt, "uint"},
                                                                      {kNumberTypeComplex, "Complex"},
                                                                      {kNumberTypeInt4, "Int4"},
                                                                      {kNumberTypeGLUInt, "GLUInt"},
                                                                      {kObjectTypeMonad, "Monad"},
                                                                      {kObjectTypeCSRTensorType, "CSRTensor"},
                                                                      {kObjectTypeMapTensorType, "MapTensor"}};

  auto it = type_id_to_string.find(id);
  std::string type = "";
  if (it != type_id_to_string.end()) {
    type = it->second;
  } else {
    type = std::to_string(id);
  }

  return type;
}

TypePtr TypeIdToType(TypeId id) {
  static mindspore::HashMap<TypeId, TypePtr> type_id_to_type = {{kNumberTypeFloat16, kFloat16},
                                                                {kNumberTypeFloat, kFloat32},
                                                                {kNumberTypeFloat32, kFloat32},
                                                                {kNumberTypeFloat64, kFloat64},
                                                                {kNumberTypeComplex64, kComplex64},
                                                                {kNumberTypeInt8, kInt8},
                                                                {kNumberTypeInt16, kInt16},
                                                                {kNumberTypeInt32, kInt32},
                                                                {kNumberTypeInt, kInt32},
                                                                {kNumberTypeInt64, kInt64},
                                                                {kNumberTypeUInt8, kUInt8},
                                                                {kNumberTypeUInt16, kUInt16},
                                                                {kNumberTypeUInt32, kUInt32},
                                                                {kNumberTypeUInt64, kUInt64},
                                                                {kNumberTypeBool, kBool},
                                                                {kNumberTypeComplex64, kComplex64},
                                                                {kNumberTypeComplex128, kComplex128},
                                                                {kMetaTypeExternal, kTypeExternal},
                                                                {kMetaTypeAny, kTypeAny},
                                                                {kMetaTypeNone, kTypeNone},
                                                                {kMetaTypeNull, kTypeNull},
                                                                {kMetaTypeEllipsis, kTypeEllipsis},
                                                                {kObjectTypeEnvType, kTypeEnv},
                                                                {kObjectTypeRefKey, kRefKeyType},
                                                                {kObjectTypeRef, kRefType},
                                                                {kMetaTypeTypeType, kTypeType},
                                                                {kObjectTypeString, kString},
                                                                {kObjectTypeList, kList},
                                                                {kObjectTypeTuple, kTuple},
                                                                {kObjectTypeNumber, kNumber},
                                                                {kObjectTypeDictionary, kDict},
                                                                {kObjectTypeSlice, kSlice},
                                                                {kObjectTypeKeyword, kKeyword},
                                                                {kObjectTypeTensorType, kTensorType},
                                                                {kObjectTypeUMonad, kUMonadType},
                                                                {kObjectTypeIOMonad, kIOMonadType},
                                                                {kTypeUnknown, kTypeNone},
                                                                {kMetaTypeProblem, kTypeNone},
                                                                {kObjectTypeCSRTensorType, kCSRTensorType},
                                                                {kObjectTypeCOOTensorType, kCOOTensorType},
                                                                {kObjectTypeRowTensorType, kRowTensorType},
                                                                {kObjectTypeMapTensorType, kMapTensorType}};
  const auto &it = type_id_to_type.find(id);
  if (it == type_id_to_type.end()) {
    MS_LOG(EXCEPTION) << "Not support the type: " << GetExcptionTypeString(id);
  }
  return it->second;
}

std::string TypeIdToString(TypeId id, bool to_lower) {
  switch (id) {
    case TypeId::kNumberTypeFloat:
      return "float";
    case TypeId::kNumberTypeInt:
      return "int";
    case TypeId::kNumberTypeUInt:
      return "uint";
    default:
      break;
  }
  auto type = TypeIdToType(id)->ToString();
  if (to_lower) {
    (void)std::transform(type.begin(), type.end(), type.begin(),
                         [](auto c) { return static_cast<char>(std::tolower(c)); });
  }
  return type;
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

/// Cnvert a string(like "type1, type2, type3") to Vector(TypeID_1, TypeID_1, TypeID_1)
/// \param type_names
/// \param types  The return types
/// \return : true, convert success;  false, format error
bool StringToVectorOfType(const std::string &type_names, std::vector<TypePtr> *types) {
  if (type_names.length() == 0) {
    return true;
  }
  std::string::size_type start = 0;
  std::string::size_type end = type_names.find_first_of(',');
  while (end != std::string::npos) {
    types->push_back(StringToType(type_names.substr(start, end)));
    // Skip ',' to find the next element.
    start = end + 1;
    end = type_names.find_first_of(',', start);
  }
  if (start >= type_names.size()) {
    return false;
  }
  types->push_back(StringToType(type_names.substr(start)));
  return true;
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

TypePtr COOTensorStrToType(const std::string &type_name) {
  TypePtr type = nullptr;
  if (type_name == "COOTensor") {
    type = std::make_shared<COOTensorType>();
  } else {
    size_t start = type_name.find_first_of('[');
    size_t end = type_name.find_last_of(']');
    // It's better to using regular expression, now just do simple check.
    if (start == std::string::npos || end == std::string::npos || end < start) {
      MS_EXCEPTION(NotSupportError) << "Expect format like 'COOTensor[type1, type2, ...]', but got '" << type_name
                                    << "' that not provide pair of ('[', ']').";
    }
    start = start + 1;
    std::string element_strs = type_name.substr(start, end - start);
    std::vector<TypePtr> element_types;
    auto ret = StringToVectorOfType(element_strs, &element_types);
    if (!ret) {
      MS_EXCEPTION(NotSupportError) << "Expect format like 'COOTensor[type1, type2, ...]', but got '" << type_name
                                    << "' that miss typename after ','.";
    }
    type = std::make_shared<COOTensorType>(element_types);
  }
  return type;
}

TypePtr CSRTensorStrToType(const std::string &type_name) {
  TypePtr type = nullptr;
  if (type_name == "CSRTensor") {
    type = std::make_shared<CSRTensorType>();
  } else {
    size_t start = type_name.find_first_of('[');
    size_t end = type_name.find_last_of(']');
    // It's better to using regular expression, now just do simple check.
    if (start == std::string::npos || end == std::string::npos || end < start) {
      MS_EXCEPTION(NotSupportError) << "Expect format like 'CSRTensor[type1, type2, ...]', but got '" << type_name
                                    << "' that not provide pair of ('[', ']').";
    }
    start = start + 1;
    std::string element_strs = type_name.substr(start, end - start);
    std::vector<TypePtr> element_types;
    auto ret = StringToVectorOfType(element_strs, &element_types);
    if (!ret) {
      MS_EXCEPTION(NotSupportError) << "Expect format like 'CSRTensor[type1, type2, ...]', but got '" << type_name
                                    << "' that miss typename after ','.";
    }
    type = std::make_shared<CSRTensorType>(element_types);
  }
  return type;
}

TypePtr MapTensorStrToType(const std::string &type_name) {
  TypePtr type = nullptr;
  if (type_name == "MapTensor") {
    return std::make_shared<MapTensorType>();
  }
  size_t start = type_name.find_first_of('[');
  size_t end = type_name.find_last_of(']');
  // It's better to using regular expression, now just do simple check.
  if (start == std::string::npos || end == std::string::npos || end < start) {
    MS_EXCEPTION(NotSupportError) << "Expect format like 'MapTensor[key_dtype, value_dtype]', but got '" << type_name
                                  << "' that not provide pair of ('[', ']').";
  }
  start = start + 1;
  std::string element_strs = type_name.substr(start, end - start);
  std::vector<TypePtr> element_types;
  auto ret = StringToVectorOfType(element_strs, &element_types);
  constexpr size_t num_of_elements = 2;
  if (!ret || element_types.size() != num_of_elements) {
    MS_EXCEPTION(NotSupportError) << "Expect format like 'MapTensor[key_dtype, value_dtype]', but got '" << type_name
                                  << "' that miss typename after ','.";
  }
  return std::make_shared<MapTensorType>(element_types[0], element_types[1]);
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
    auto start = type_name.find_first_of('[');
    auto end = type_name.find_last_of(']');
    // It's better to using regular expression, now just do simple check.
    if (start == std::string::npos || end == std::string::npos || end < start) {
      MS_EXCEPTION(NotSupportError) << "Expect format like 'List[type1, type2, ...]', but got '" << type_name
                                    << "' that not provide pair of ('[', ']').";
    }
    start = start + 1;
    std::string element_strs = type_name.substr(start, end - start);
    std::vector<TypePtr> element_types;
    auto ret = StringToVectorOfType(element_strs, &element_types);
    if (!ret) {
      MS_EXCEPTION(NotSupportError) << "Expect format like 'List[type1, type2, ...]', but got '" << type_name
                                    << "' that miss typename after ','.";
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
    size_t start = type_name.find_first_of('[');
    size_t end = type_name.find_last_of(']');
    // It's better to using regular expression, now just do simple check.
    if (start == std::string::npos || end == std::string::npos || end < start) {
      MS_EXCEPTION(NotSupportError) << "Expect format like 'Tuple[type1, type2, ...]', but got '" << type_name
                                    << "' that not provide pair of ('[', ']').";
    }
    start = start + 1;
    std::string element_strs = type_name.substr(start, end - start);
    std::vector<TypePtr> element_types;
    auto ret = StringToVectorOfType(element_strs, &element_types);
    if (!ret) {
      MS_EXCEPTION(NotSupportError) << "Expect format like 'Tuple[type1, type2, ...]', but got '" << type_name
                                    << "' that miss typename after ','.";
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
    size_t start = type_name.find_first_of('[');
    size_t end = type_name.find_last_of(']');
    // It's better to using regular expression, now just do simple check.
    if (start == std::string::npos || end == std::string::npos || end < start) {
      MS_EXCEPTION(NotSupportError) << "Expect format like 'Function[(type1, type2, ...), ret_type]', but got '"
                                    << type_name << "' that not provide pair of ('[', ']').";
    }
    start = start + 1;
    std::string str_all = type_name.substr(start, end - start);
    size_t start_a = str_all.find_first_of('(');
    size_t end_a = str_all.find_last_of(')');
    // It's better to using regular expression, now just do simple check.
    if (start_a == std::string::npos || end_a == std::string::npos || end_a < start_a) {
      MS_EXCEPTION(NotSupportError) << "Expect format like 'Function[(type1, type2, ...), ret_type]', but got '"
                                    << type_name << "' that not provide pair of ('(', ')').";
    }
    start_a = start_a + 1;
    std::string str_args = str_all.substr(start_a, end_a - start_a);
    // bypass " " between ")" and retval
    start = end_a + 2;
    if (start >= str_all.size()) {
      MS_EXCEPTION(NotSupportError) << "Expect format like 'Function[(type1, type2, ...), ret_type]', but got '"
                                    << type_name;
    }
    std::string str_retval = str_all.substr(start);
    std::vector<TypePtr> args_type;
    auto ret = StringToVectorOfType(str_args, &args_type);
    if (!ret) {
      MS_EXCEPTION(NotSupportError) << "Expect format like 'Function[(type1, type2, ...), ret_type]', but got '"
                                    << type_name;
    }
    TypePtr retval = StringToType(str_retval);
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
                                                    {"bool", std::make_shared<Bool>()},
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
  static std::map<std::string, std::function<TypePtr(const std::string &)>, name_cmp> type_map = {
    {"Int", [](const std::string &type_name) -> TypePtr { return StringToNumberType<Int>(type_name, "Int"); }},
    {"int", [](const std::string &type_name) -> TypePtr { return StringToNumberType<Int>(type_name, "int"); }},
    {"UInt", [](const std::string &type_name) -> TypePtr { return StringToNumberType<UInt>(type_name, "UInt"); }},
    {"uint", [](const std::string &type_name) -> TypePtr { return StringToNumberType<UInt>(type_name, "uint"); }},
    {"Float", [](const std::string &type_name) -> TypePtr { return StringToNumberType<Float>(type_name, "Float"); }},
    {"float", [](const std::string &type_name) -> TypePtr { return StringToNumberType<Float>(type_name, "float"); }},
    {"Complex", [](const std::string &tname) -> TypePtr { return StringToNumberType<Complex>(tname, "Complex"); }},
    {"complex", [](const std::string &tname) -> TypePtr { return StringToNumberType<Complex>(tname, "complex"); }},
    {"Tensor", [](const std::string &type_name) -> TypePtr { return TensorStrToType(type_name); }},
    {"Undetermined", [](const std::string &type_name) -> TypePtr { return UndeterminedStrToType(type_name); }},
    {"RowTensor", [](const std::string &type_name) -> TypePtr { return RowTensorStrToType(type_name); }},
    {"COOTensor", [](const std::string &type_name) -> TypePtr { return COOTensorStrToType(type_name); }},
    {"CSRTensor", [](const std::string &type_name) -> TypePtr { return CSRTensorStrToType(type_name); }},
    {"MapTensor", [](const std::string &type_name) -> TypePtr { return MapTensorStrToType(type_name); }},
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
    // Any
    // External
    MS_LOG(EXCEPTION) << "Unsupported type name: " << type_name << "!";
  }
  return type;
}

TypeId StringToTypeId(const std::string &type_name) { return StringToType(type_name)->type_id(); }

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
