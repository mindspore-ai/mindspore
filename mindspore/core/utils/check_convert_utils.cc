/**
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

#include <utility>
#include <vector>
#include <algorithm>
#include <typeinfo>
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"
#include "ir/dtype/type.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype.h"

namespace mindspore {
bool CheckAndConvertUtils::IsEqualVector(const std::vector<int64_t> &vec_1, const std::vector<int64_t> &vec_2) {
  if (vec_1.size() != vec_2.size()) {
    return false;
  }
  for (size_t index = 0; index < vec_1.size(); ++index) {
    if (vec_1[index] != vec_2[index]) {
      return false;
    }
  }
  return true;
}

std::vector<int64_t> CheckAndConvertUtils::CheckPositiveVector(const std::string &arg_name,
                                                               const std::vector<int64_t> &arg_value,
                                                               const std::string &prim_name, bool allow_four,
                                                               bool ret_four) {
  auto raise_message = [allow_four, prim_name, arg_value, arg_name]() -> void {
    std::ostringstream buffer;
    buffer << "For " << prim_name << " attr " << arg_name << " should be a positive vector of size two ";
    //    if (allow_four) {
    //      buffer << "or four ";
    //    }
    buffer << " positive int64_t numbers , but got [";
    for (auto item : arg_value) {
      buffer << item << ",";
    }
    buffer << "]";
    MS_EXCEPTION(ValueError) << buffer.str();
  };
  for (auto item : arg_value) {
    if (item < 0) {
      raise_message();
    }
  }
  //  if (arg_value.size() == 1) {
  //    return ret_four ? std::vector<int64_t>{1, 1, arg_value[0], arg_value[0]}
  //                    : std::vector<int64_t>{arg_value[0], arg_value[0]};
  //  }
  //  if (arg_value.size() == 2) {
  //    return ret_four ? std::vector<int64_t>{1, 1, arg_value[0], arg_value[1]} : arg_value;
  //  } else if (arg_value.size() == 4 && allow_four) {
  //    return ret_four ? arg_value : std::vector<int64_t>{arg_value[2], arg_value[3]};
  //  }
  //  raise_message();
  return arg_value;
}

std::string CheckAndConvertUtils::CheckString(const std::string &arg_name, const std::string &arg_value,
                                              const std::set<std::string> &check_list, const std::string &prim_name) {
  if (check_list.find(arg_value) != check_list.end()) {
    return arg_value;
  }
  std::ostringstream buffer;
  buffer << "For " << prim_name << " the " << arg_name << " should be str and must be ";
  if (check_list.size() == 1) {
    buffer << (*check_list.begin()) << "but got " << arg_value;
    MS_EXCEPTION(ValueError) << buffer.str();
  }
  buffer << "one of {";
  for (const auto &item : check_list) {
    buffer << item << " ,";
  }
  buffer << " }"
         << " but got " << arg_value;
  MS_EXCEPTION(ValueError) << buffer.str();
}

int64_t CheckAndConvertUtils::CheckInteger(const std::string &arg_name, int64_t arg_value, CompareEnum compare_operator,
                                           int64_t match_value, const std::string &prim_name) {
  auto iter = kCompareMap<float>.find(compare_operator);
  if (iter == kCompareMap<float>.end()) {
    MS_EXCEPTION(NotExistsError) << "compare_operator " << compare_operator << " cannot find in the compare map";
  }
  if (iter->second(arg_value, match_value)) {
    return arg_value;
  }
  std::ostringstream buffer;
  if (prim_name.empty()) {
    buffer << "The ";
  } else {
    buffer << "For " << prim_name << " the ";
  }
  buffer << arg_name << " must ";
  auto iter_to_string = kCompareToString.find(compare_operator);
  if (iter_to_string == kCompareToString.end()) {
    MS_EXCEPTION(NotExistsError) << "compare_operator " << compare_operator << " cannot find in the compare string map";
  }
  buffer << iter_to_string->second << match_value << " , but got " << arg_value;
  MS_EXCEPTION(ValueError) << buffer.str();
}

std::vector<int64_t> CheckAndConvertUtils::ConvertShapePtrToShape(const std::string &arg_name,
                                                                  const BaseShapePtr &shape,
                                                                  const std::string &prim_name) {
  MS_EXCEPTION_IF_NULL(shape);
  if (!shape->isa<abstract::Shape>()) {
    MS_EXCEPTION(ValueError) << "The " << arg_name << "'s shape is " << shape->ToString()
                             << "should be a common shape!";
  }
  auto shape_element = shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_element);
  return shape_element->shape();
}

void CheckAndConvertUtils::Check(const string &arg_name, int64_t arg_value, CompareEnum compare_type,
                                 const string &value_name, int64_t value, const string &prim_name,
                                 ExceptionType exception_type) {
  auto iter = kCompareMap<float>.find(compare_type);
  if (iter == kCompareMap<float>.end()) {
    MS_EXCEPTION(NotExistsError) << "the compare type :" << compare_type << " is not in the compare map";
  }
  if (iter->second(arg_value, value)) {
    return;
  }
  std::ostringstream buffer;
  if (prim_name.empty()) {
    buffer << "The ";
  } else {
    buffer << "For " << prim_name << " the ";
  }
  auto iter_to_string = kCompareToString.find(compare_type);
  if (iter_to_string == kCompareToString.end()) {
    MS_EXCEPTION(NotExistsError) << "compare_operator " << compare_type << " cannot find in the compare string map";
  }
  MS_EXCEPTION(exception_type) << buffer.str() << arg_name << " should be " << iter_to_string->second << value
                               << " but got " << arg_value;
}

TypeId CheckAndConvertUtils::CheckTensorTypeSame(const std::map<std::string, TypePtr> &types,
                                                 const std::set<TypeId> &check_list, const std::string &prim_name) {
  if (types.empty()) {
    MS_EXCEPTION(ArgumentError) << "Trying to use the function to check a empty types map!";
  }
  std::set<TypeId> types_id;
  std::ostringstream buffer;
  buffer << "For " << prim_name;
  for (const auto &type : types) {
    MS_EXCEPTION_IF_NULL(type.second);
    if (!type.second->isa<TensorType>()) {
      MS_EXCEPTION(TypeError) << "The " << prim_name << "'s" << type.first << " input must be tensor type but got "
                              << type.second->ToString();
    }
    auto tensor_type = type.second->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor_type);
    auto element = tensor_type->element();
    MS_EXCEPTION_IF_NULL(element);
    types_id.emplace(element->type_id());
  }
  if (types_id.size() > 1) {
    buffer << "'s input type is not same : ";
    for (const auto &item : types) {
      buffer << "[ name : " << item.first << " ,type : " << item.second->ToString() << "]";
    }
    MS_EXCEPTION(TypeError) << buffer.str();
  }
  if (check_list.find(*types_id.begin()) == check_list.end()) {
    buffer << " type of ";
    for (const auto &elem : types) {
      buffer << elem.first << " should be in [";
      for (auto type_elem : check_list) {
        buffer << TypeIdToType(type_elem)->ToString() << " ,";
      }
      buffer << "] , but got " << types.begin()->second->ToString();
    }
    MS_EXCEPTION(TypeError) << buffer.str();
  }
  return *types_id.begin();
}

void CheckAndConvertUtils::CheckTensorTypeValid(const std::string &type_name, const TypePtr type,
                                                const std::set<TypeId> &check_list, const std::string &prim_name) {
  MS_EXCEPTION_IF_NULL(type);
  if (!type->isa<TensorType>()) {
    MS_EXCEPTION(TypeError) << "The " << prim_name << "'s " << type_name << " input must be tensor type but got "
                            << type->ToString();
  }
  auto tensor_type = type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto element = tensor_type->element();
  MS_EXCEPTION_IF_NULL(element);
  std::ostringstream buffer;
  if (check_list.find(element->type_id()) == check_list.end()) {
    buffer << "type of " << type_name << " should be in [";
    for (auto type_elem : check_list) {
      buffer << TypeIdToType(type_elem)->ToString() << " ,";
    }
    buffer << "], but got " << type->ToString();
    MS_EXCEPTION(TypeError) << buffer.str();
  }
}

void CheckAndConvertUtils::CheckSubClass(const std::string &type_name, const TypePtr type_,
                                         const std::set<TypePtr> &template_types, const std::string &prim_name) {
  MS_EXCEPTION_IF_NULL(type_);
  bool hit = false;
  for (auto template_type : template_types) {
    if (type_->isa<Type>()) {
      if (IsIdentidityOrSubclass(type_, template_type)) {
        hit = true;
        break;
      }
    } else if (type_->type_id() == template_type->type_id()) {
      hit = true;
      break;
    }
  }
  if (!hit) {
    std::string type_str = type_->ToString();
    std::ostringstream buffer;
    buffer << "For '" << prim_name << "', the type of `" << type_name << "` should be subclass of ";
    for (auto template_type : template_types) {
      buffer << template_type->ToString() << ",";
    }
    buffer << " but got " << type_str << ".";
    MS_EXCEPTION(TypeError) << buffer.str();
  }
}

void CheckAndConvertUtils::CheckScalarOrTensorTypesSame(const std::map<std::string, TypePtr> &args,
                                                        const std::set<TypePtr> &valid_values,
                                                        const std::string &prim_name, const bool allow_mix) {
  std::vector<std::map<std::string, TypePtr>> check_results;
  for (auto &iter : args) {
    std::map<std::string, TypePtr> arg = {{iter.first, iter.second}};
    check_results.push_back(_CheckArgumentType(arg, valid_values, prim_name));
  }

  std::map<std::string, TypePtr> &arg_ = check_results[0];
  int64_t size = check_results.size();
  for (int64_t it = 1; it != size; it++) {
    arg_ = _CheckTypeSame(arg_, check_results[it], prim_name, allow_mix);
  }
}

std::map<std::string, TypePtr> CheckAndConvertUtils::_CheckArgumentType(const std::map<std::string, TypePtr> &arg,
                                                                        const std::set<TypePtr> &valid_values,
                                                                        const std::string &prim_name) {
  std::string arg_key = arg.begin()->first;
  TypePtr arg_val = arg.begin()->second;

  if (arg_val->isa<TensorType>()) {
    auto arg_val_ = std::static_pointer_cast<TensorType>(arg_val);
    arg_val = arg_val_->element();
  }

  auto it = valid_values.find(arg_val);
  if (it == valid_values.end()) {
    std::ostringstream buffer;
    buffer << "For '" << prim_name << "' , the `" << arg_key << "` should be in { ";
    for (auto valid_value : valid_values) {
      buffer << valid_value->ToString() << " },";
      buffer << "but `" << arg_key << "`"
             << "is" << arg_val->ToString() << ".";
    }
    MS_EXCEPTION(TypeError) << buffer.str();
  }
  return arg;
}

std::map<std::string, TypePtr> CheckAndConvertUtils::_CheckTypeSame(const std::map<std::string, TypePtr> &arg1,
                                                                    const std::map<std::string, TypePtr> &arg2,
                                                                    const std::string &prim_name,
                                                                    const bool allow_mix) {
  std::string arg1_name = arg1.begin()->first;
  TypePtr arg1_type = arg1.begin()->second;
  std::string arg2_name = arg2.begin()->first;
  TypePtr arg2_type = arg2.begin()->second;
  bool except_flag = false;

  if (arg1_type->isa<TensorType>() && arg2_type->isa<TensorType>()) {
    arg1_type = std::static_pointer_cast<TensorType>(arg1_type)->element();
    arg2_type = std::static_pointer_cast<TensorType>(arg2_type)->element();
  } else if (allow_mix) {
    arg1_type = arg1_type->isa<TensorType>() ? std::static_pointer_cast<TensorType>(arg1_type)->element() : arg1_type;
    arg2_type = arg2_type->isa<TensorType>() ? std::static_pointer_cast<TensorType>(arg2_type)->element() : arg2_type;
  } else {
    except_flag = true;
  }

  if (except_flag || arg1_type != arg2_type) {
    std::ostringstream buffer;
    buffer << "For '" << prim_name << "'"
           << "type of "
           << "`" << arg2_name << "` should be same as "
           << "`" << arg1_name << "`,";
    buffer << "but `" << arg1_name << "` is " << arg1_type->ToString() << "and `" << arg2_name << "` is "
           << arg2_type->ToString() << ".";
    MS_EXCEPTION(TypeError) << buffer.str();
  }
  return arg1;
}

TypeId CheckAndConvertUtils::CheckTypeSame(const std::string &arg_name, const TypePtr arg_type,
                                           const std::set<TypeId> &valid_type, const std::string &prim_name) {
  if (valid_type.empty()) {
    MS_EXCEPTION(ArgumentError) << "Trying to use the function to check a empty valid_type!";
  }
  // std::set<TypeId> types_id;
  std::ostringstream buffer;
  TypeId arg_type_;
  arg_type_ = arg_type->isa<TensorType>() ? std::static_pointer_cast<TensorType>(arg_type)->generic_type_id()
                                          : arg_type->type_id();

  auto it = valid_type.find(arg_type_);
  if (it == valid_type.end()) {
    buffer << "For" << prim_name << ", the '" << arg_name << "' should be {' one of '" << valid_type.size() << "'}";
    for (auto type : valid_type) {
      buffer << "{" << TypeIdLabel(type);
    }
    buffer << "},";
    buffer << "but got " << arg_type->ToString() << ".";
    MS_EXCEPTION(TypeError) << buffer.str();
  }
  return arg_type_;
}
}  // namespace mindspore
