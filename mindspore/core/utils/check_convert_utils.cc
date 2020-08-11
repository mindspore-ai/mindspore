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

#include "utils/check_convert_utils.h"
#include <utility>
#include "abstract/abstract_value.h"

namespace mindspore {
namespace {
const std::map<CompareEnum, std::function<bool(int, int)>> kCompareMap = {
  {kEqual, [](int num1, int num2) -> bool { return num1 == num2; }},
  {kNotEqual, [](int num1, int num2) -> bool { return num1 != num2; }},
  {kLessThan, [](int num1, int num2) -> bool { return num1 < num2; }},
  {kLessEqual, [](int num1, int num2) -> bool { return num1 <= num2; }},
  {kGreaterThan, [](int num1, int num2) -> bool { return num1 > num2; }},
  {kGreaterEqual, [](int num1, int num2) -> bool { return num1 >= num2; }}};

const std::map<CompareRange, std::function<bool(int, std::pair<int, int>)>> kCompareRangeMap = {
  {kIncludeNeither,
   [](int num1, std::pair<int, int> range) -> bool { return num1 > range.first && num1 < range.second; }},
  {kIncludeLeft,
   [](int num1, std::pair<int, int> range) -> bool { return num1 >= range.first && num1 < range.second; }},
  {kIncludeRight,
   [](int num1, std::pair<int, int> range) -> bool { return num1 > range.first && num1 <= range.second; }},
  {kIncludeBoth,
   [](int num1, std::pair<int, int> range) -> bool { return num1 >= range.first && num1 <= range.second; }}};

const std::map<CompareEnum, std::string> kCompareToString = {
  {kEqual, "equal "},          {kNotEqual, "not equal "},       {kLessThan, "less than "},
  {kLessEqual, "less eqaul "}, {kGreaterThan, "greater than "}, {kGreaterEqual, "greate equal "}};

const std::map<CompareRange, std::pair<std::string, std::string>> kCompareRangeToString = {
  {kIncludeNeither, {"in (", ")"}},
  {kIncludeLeft, {" in [", ")"}},
  {kIncludeRight, {"in (", "]"}},
  {kIncludeBoth, {"in [", "]"}}};
}  // namespace
bool CheckAndConvertUtils::IsEqualVector(const std::vector<int> &vec_1, const std::vector<int> &vec_2) {
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

std::vector<int> CheckAndConvertUtils::CheckPositiveVector(const std::string &arg_name,
                                                           const std::vector<int> &arg_value,
                                                           const std::string &prim_name, bool allow_four,
                                                           bool ret_four) {
  if (arg_value.size() == 2) {
    return ret_four ? std::vector<int>{1, 1, arg_value[0], arg_value[1]} : arg_value;
  } else if (arg_value.size() == 4 && allow_four) {
    return ret_four ? arg_value : std::vector<int>{arg_value[2], arg_value[3]};
  }
  std::ostringstream buffer;
  buffer << "For " << prim_name << " attr " << arg_name << " should be a positive vector of size two ";
  if (allow_four) {
    buffer << "or four ";
  }
  buffer << " positive int numbers , but got [";
  for (auto item : arg_value) {
    buffer << item << ",";
  }
  buffer << "]";
  MS_EXCEPTION(ValueError) << buffer.str();
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

int CheckAndConvertUtils::CheckInteger(const std::string &arg_name, int arg_value, CompareEnum compare_operator,
                                       int match_value, const std::string &prim_name) {
  auto iter = kCompareMap.find(compare_operator);
  if (iter == kCompareMap.end()) {
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

void CheckAndConvertUtils::CheckInRange(const std::string &arg_name, int arg_value, CompareRange compare_operator,
                                        const std::pair<int, int> &range, const std::string &prim_name) {
  auto iter = kCompareRangeMap.find(compare_operator);
  if (iter == kCompareRangeMap.end()) {
    MS_EXCEPTION(NotExistsError) << "compare_operator " << compare_operator << " cannot find in the compare map";
  }
  if (iter->second(arg_value, range)) {
    return;
  }
  std::ostringstream buffer;
  if (prim_name.empty()) {
    buffer << "The ";
  } else {
    buffer << "For " << prim_name << " the ";
  }
  buffer << arg_name << " must ";
  auto iter_to_string = kCompareRangeToString.find(compare_operator);
  if (iter_to_string == kCompareRangeToString.end()) {
    MS_EXCEPTION(NotExistsError) << "compare_operator " << compare_operator << " cannot find in the compare string map";
  }
  auto range_strng = iter_to_string->second;
  buffer << range_strng.first << range.first << "," << range_strng.second << " , but got " << arg_value;
  MS_EXCEPTION(ValueError) << buffer.str();
}

std::vector<int> CheckAndConvertUtils::ConvertShapePtrToShape(const std::string &arg_name, const BaseShapePtr &shape,
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

void CheckAndConvertUtils::Check(const string &arg_name, int arg_value, CompareEnum compare_type,
                                 const string &value_name, int value, const string &prim_name,
                                 ExceptionType exception_type) {
  auto iter = kCompareMap.find(compare_type);
  if (iter == kCompareMap.end()) {
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
void CheckAndConvertUtils::Check(const string &arg_name, const std::vector<int> &arg_value, CompareEnum compare_type,
                                 const string &value_name, const std::vector<int> &value, const string &prim_name,
                                 ExceptionType exception_type) {
  if (compare_type != kEqual) {
    auto iter = kCompareToString.find(compare_type);
    if (iter != kCompareToString.end()) {
      MS_EXCEPTION(NotSupportError) << "Only supported equal to compare two vectors but got " << iter->second;
    }
    MS_EXCEPTION(UnknownError) << "Cannot find the operator " << compare_type << "in the compare map!";
  }
  if (arg_value == value) {
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
  buffer << arg_name << "should be " << iter_to_string->second << " [";
  for (auto item : value) {
    buffer << item << ",";
  }
  buffer << "] "
         << "but got [";
  for (auto item : arg_value) {
    buffer << item << " ,";
  }
  buffer << "]";
  MS_EXCEPTION(exception_type) << buffer.str();
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
}  // namespace mindspore
