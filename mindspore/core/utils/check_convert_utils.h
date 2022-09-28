/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_UTILS_CHECK_CONVERT_UTILS_H_
#define MINDSPORE_CORE_UTILS_CHECK_CONVERT_UTILS_H_
#include <vector>
#include <string>
#include <map>
#include <set>
#include <utility>
#include <typeinfo>
#include <memory>
#include "abstract/param_validator.h"
#include "base/base.h"
#include "ir/anf.h"
#include "include/api/format.h"
#include "utils/log_adapter.h"
#if __has_include("include/mindapi/base/types.h")
#include "include/mindapi/base/types.h"
#else
#include "mindapi/base/types.h"
#endif

namespace mindspore {
typedef std::pair<std::map<std::string, int64_t>, std::map<int64_t, std::string>> AttrConverterPair;
typedef std::map<std::string, std::vector<int64_t>> ShapeMap;
constexpr auto kShape = "shape";
constexpr auto kMinShape = "min_shape";
constexpr auto kMaxShape = "max_shape";

enum CompareEnum : int64_t {
  kEqual = 1,         // ==
  kNotEqual = 2,      // !=
  kLessThan = 3,      // <
  kLessEqual = 4,     // <=
  kGreaterThan = 5,   // >
  kGreaterEqual = 6,  // >=
};

enum CompareRange {
  kIncludeNeither = 1,  // (a,b)
  kIncludeLeft = 2,     // [a,b)
  kIncludeRight = 3,    // (a,b]
  kIncludeBoth = 4,     // [a,b]
};

enum ReduceType : int64_t {
  REDUCE_MAX = 0,
  REDUCE_MEAN = 1,
  REDUCE_ALL = 2,
  REDUCE_ANY = 3,
  REDUCE_LOG_SUM_EXP = 4,
  REDUCE_PROD = 5,
  REDUCE_SUM = 6,
  REDUCE_UNKNOW = 7,
};

enum GateOrderMode : int64_t { RZH = 0, ZRH = 1 };

template <typename T>
const std::map<CompareEnum, std::function<bool(T, T)>> kCompareMap = {
  {kEqual, [](T num1, T num2) -> bool { return num1 == num2; }},
  {kNotEqual, [](T num1, T num2) -> bool { return num1 != num2; }},
  {kLessThan, [](T num1, T num2) -> bool { return num1 < num2; }},
  {kLessEqual, [](T num1, T num2) -> bool { return num1 <= num2; }},
  {kGreaterThan, [](T num1, T num2) -> bool { return num1 > num2; }},
  {kGreaterEqual, [](T num1, T num2) -> bool { return num1 >= num2; }}};

template <typename T>
const std::map<CompareRange, std::function<bool(T, std::pair<T, T>)>> kCompareRangeMap = {
  {kIncludeNeither, [](T num1, std::pair<T, T> range) -> bool { return num1 > range.first && num1 < range.second; }},
  {kIncludeLeft, [](T num1, std::pair<T, T> range) -> bool { return num1 >= range.first && num1 < range.second; }},
  {kIncludeBoth, [](T num1, std::pair<T, T> range) -> bool { return num1 >= range.first && num1 <= range.second; }},
  {kIncludeRight, [](T num1, std::pair<T, T> range) -> bool { return num1 > range.first && num1 <= range.second; }}};

const std::map<CompareEnum, std::string> kCompareToString = {
  {kEqual, "be equal to "},           {kNotEqual, "be not equal to "},
  {kLessThan, "be less than "},       {kLessEqual, "be less than or equal to "},
  {kGreaterThan, "be greater than "}, {kGreaterEqual, "be greater than or equal to "}};

const std::map<CompareRange, std::pair<std::string, std::string>> kCompareRangeToString = {
  {kIncludeNeither, {"in (", ")"}},
  {kIncludeLeft, {"in [", ")"}},
  {kIncludeRight, {"in (", "]"}},
  {kIncludeBoth, {"in [", "]"}}};

class MS_CORE_API CheckAndConvertUtils {
 public:
  template <typename T>
  static std::vector<T> CheckPositiveVector(const std::string &arg_name, const std::vector<T> &arg_value,
                                            const std::string &prim_name) {
    std::ostringstream buffer;
    buffer << "For primitive[" << prim_name << "], the attribute[" << arg_name
           << "] should be a vector with all positive item. but got [";
    if (std::any_of(arg_value.begin(), arg_value.end(), [](T item) { return item < T(0); })) {
      for (auto item : arg_value) {
        buffer << item << ", ";
      }
      buffer << "].";
      MS_EXCEPTION(ValueError) << buffer.str();
    }

    return arg_value;
  }
  static std::string CheckString(const std::string &arg_name, const std::string &arg_value,
                                 const std::set<std::string> &check_list, const std::string &prim_name);

  // CheckValue should replace CheckInteger
  static int64_t CheckInteger(const std::string &arg_name, int64_t arg_value, CompareEnum compare_operator,
                              int64_t match_value, const std::string &prim_name = "");

  template <typename T>
  static std::vector<T> CheckPositiveVectorExcludeZero(const std::string &arg_name, const std::vector<T> &arg_value,
                                                       const std::string &prim_name) {
    std::ostringstream buffer;
    buffer << "For primitive[" << prim_name << "], the attribute[" << arg_name
           << "] should be a vector with all positive item. but got [";
    if (std::any_of(arg_value.begin(), arg_value.end(), [](T item) { return item <= T(0); })) {
      for (auto item : arg_value) {
        buffer << item << ", ";
      }
      buffer << "].";
      MS_EXCEPTION(ValueError) << buffer.str();
    }

    return arg_value;
  }

  template <typename T>
  static T CheckValue(const std::string &arg_name, T arg_value, CompareEnum compare_operator, T match_value,
                      const std::string &prim_name) {
    auto iter = kCompareMap<T>.find(compare_operator);
    if (iter == kCompareMap<T>.end()) {
      MS_EXCEPTION(NotExistsError) << "compare_operator " << compare_operator << " cannot find in the compare map";
    }
    if (iter->second(arg_value, match_value)) {
      return arg_value;
    }
    std::ostringstream buffer;
    if (prim_name.empty()) {
      buffer << "The attribute[" << arg_name << "] must ";
    } else {
      buffer << "For primitive[" << prim_name << "], the " << arg_name << " must ";
    }
    auto iter_to_string = kCompareToString.find(compare_operator);
    if (iter_to_string == kCompareToString.end()) {
      MS_EXCEPTION(NotExistsError) << "compare_operator " << compare_operator
                                   << " cannot find in the compare string map";
    }
    buffer << iter_to_string->second << match_value << " , but got " << arg_value << ".";
    MS_EXCEPTION(ValueError) << buffer.str();
  }

  template <typename T>
  static T CheckValue(const std::string &arg_name, T arg_value, CompareEnum compare_operator,
                      const std::string &match_name, T match_value, const std::string &prim_name) {
    auto iter = kCompareMap<T>.find(compare_operator);
    if (iter == kCompareMap<T>.end()) {
      MS_EXCEPTION(NotExistsError) << "compare_operator " << compare_operator << " cannot find in the compare map";
    }
    if (iter->second(arg_value, match_value)) {
      return arg_value;
    }
    std::ostringstream buffer;
    if (prim_name.empty()) {
      buffer << "The attribute[" << arg_name << "] must ";
    } else {
      buffer << "For primitive[" << prim_name << "], the " << arg_name << " must ";
    }
    auto iter_to_string = kCompareToString.find(compare_operator);
    if (iter_to_string == kCompareToString.end()) {
      MS_EXCEPTION(NotExistsError) << "compare_operator " << compare_operator
                                   << " cannot find in the compare string map";
    }
    buffer << iter_to_string->second << match_name << " which is " << match_value << " , but got " << arg_value << ".";
    MS_EXCEPTION(ValueError) << buffer.str();
  }

  template <typename T>
  static void CheckInRange(const std::string &arg_name, T arg_value, CompareRange compare_operator,
                           const std::pair<T, T> &range, const std::string &prim_name) {
    auto iter = kCompareRangeMap<T>.find(compare_operator);
    if (iter == kCompareRangeMap<T>.end()) {
      MS_EXCEPTION(NotExistsError) << "For " << prim_name << ", compare_operator " << compare_operator
                                   << " cannot find in the compare map";
    }
    if (range.first >= range.second) {
      MS_EXCEPTION(ValueError) << "For " << prim_name
                               << ", the check range left must be smaller than right number but got left: "
                               << range.first << " and right: " << range.second << ".";
    }
    if (iter->second(arg_value, range)) {
      return;
    }
    std::ostringstream buffer;
    if (prim_name.empty()) {
      buffer << "The attribute[" << arg_name << "] must be ";
    } else {
      buffer << "For primitive[" << prim_name << "], the " << arg_name << " must be ";
    }
    auto iter_to_string = kCompareRangeToString.find(compare_operator);
    if (iter_to_string == kCompareRangeToString.end()) {
      MS_EXCEPTION(NotExistsError) << "For " << prim_name << ", compare_operator " << compare_operator
                                   << " cannot find in the compare string map";
    }
    auto range_strng = iter_to_string->second;
    buffer << range_strng.first << range.first << "," << range.second << range_strng.second << ", but got " << arg_value
           << ".";
    MS_EXCEPTION(ValueError) << buffer.str();
  }

  static ShapeMap ConvertShapePtrToShapeMap(const BaseShapePtr &shape);
  static abstract::ShapePtr GetTensorInputShape(const std::string &prim_name,
                                                const std::vector<AbstractBasePtr> &input_args, size_t index);
  static TypePtr GetTensorInputType(const std::string &prim_name, const std::vector<AbstractBasePtr> &input_args,
                                    size_t index);
  static void Check(const std::string &arg_name, int64_t arg_value, CompareEnum compare_type, int64_t value,
                    const std::string &prim_name = "", ExceptionType exception_type = ValueError);

  template <typename T>
  static void Check(const std::string &arg_name, const std::vector<T> &arg_value, CompareEnum compare_type,
                    const std::vector<T> &value, const std::string &prim_name = "",
                    ExceptionType exception_type = ValueError) {
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
      buffer << "The attribute[" << arg_name << "]:";
    } else {
      buffer << "For primitive[" << prim_name << "], the " << arg_name << ":";
    }
    auto iter_to_string = kCompareToString.find(compare_type);
    if (iter_to_string == kCompareToString.end()) {
      MS_EXCEPTION(NotExistsError) << "compare_operator " << compare_type << " cannot find in the compare string map";
    }

    buffer << " [";
    for (auto item : arg_value) {
      buffer << item << ",";
    }
    buffer << "]";
    buffer << " must " << iter_to_string->second << "[";
    for (auto item : value) {
      buffer << item << ",";
    }
    buffer << "]";
    MS_EXCEPTION(exception_type) << buffer.str();
  }

  template <typename T>
  static std::shared_ptr<T> CheckArgs(const std::string &op, const AbstractBasePtrList &args_spec_list, size_t index) {
    if (index >= args_spec_list.size()) {
      MS_EXCEPTION(ValueError) << op << " evaluator arguments list index out of bound, size " << args_spec_list.size()
                               << ", index " << index;
    }
    auto args_abs = args_spec_list[index];
    MS_EXCEPTION_IF_NULL(args_abs);
    auto arg = dyn_cast<T>(args_abs);
    if (arg == nullptr) {
      MS_EXCEPTION(TypeError) << "For primitive[" << op << "], the input[" << index << "] should be a "
                              << abstract::ReportNameTraits<T>::name << ", but got "
                              << args_spec_list[index]->BuildType()->ToString() << ".";
    }
    return arg;
  }

  static ShapeVector CheckTensorShapeSame(const std::map<std::string, BaseShapePtr> &shapes,
                                          const std::vector<int64_t> &check_shape, const std::string &prim_name);
  static TypePtr CheckTensorTypeSame(const std::map<std::string, TypePtr> &types, const std::set<TypePtr> &check_list,
                                     const std::string &prim_name);
  static ShapeVector CheckTensorIntValue(const std::string &type_name, const ValuePtr &value,
                                         const std::string &prim_name);
  static TypePtr CheckTensorTypeValid(const std::string &type_name, const TypePtr &type,
                                      const std::set<TypePtr> &check_list, const std::string &prim_name);
  static TypePtr CheckSparseTensorTypeValid(const std::string &type_name, const TypePtr &type,
                                            const std::set<TypePtr> &check_list, const std::string &prim_name);
  static TypePtr CheckSubClass(const std::string &type_name, const TypePtr &type,
                               const std::set<TypePtr> &template_types, const std::string &prim_name);
  static TypePtr CheckScalarOrTensorTypesSame(const std::map<std::string, TypePtr> &args,
                                              const std::set<TypePtr> &valid_values, const std::string &prim_name,
                                              bool allow_mix = false);
  static TypePtr CheckTypeValid(const std::string &arg_name, const TypePtr &arg_type,
                                const std::set<TypePtr> &valid_type, const std::string &prim_name);
  static bool ConvertAttrValueToInt(const std::string &op_type, const std::string &attr_name, ValuePtr *const value);
  static bool ConvertAttrValueToString(const std::string &op_type, const std::string &attr_name, ValuePtr *const value);
  static void ConvertAttrValueInExport(const std::string &op_type, const std::string &attr_name, ValuePtr *const value);
  static void ConvertAttrValueInLoad(const std::string &op_type, const std::string &attr_name, ValuePtr *const value);
  static AttrConverterPair GetAttrConvertPair(const std::string &op_type, const std::string &attr_name);
  static bool GetDataFormatEnumValue(const ValuePtr &value, int64_t *enum_value);
  static void GetPadModEnumValue(const ValuePtr &value, int64_t *enum_value, bool is_upper = false);
  static void GetReductionEnumValue(const ValuePtr &value, int64_t *enum_value);
  static bool CheckIrAttrtoOpAttr(const std::string &op_type, const std::string &attr_name, ValuePtr *const value);
  static void CheckSummaryParam(const AbstractBasePtr &name, const AbstractBasePtr &value,
                                const std::string &class_name);
  static void CheckMode(const std::string &class_name);
  static std::vector<int64_t> CheckIntOrTupleInt(const std::string &arg_name, const ValuePtr &attr,
                                                 const std::string &prim_name);
  static std::vector<int64_t> CheckTupleInt(const std::string &arg_name, const ValuePtr &attr,
                                            const std::string &prim_name);
  static std::vector<int64_t> CheckListInt(const std::string &arg_name, const ValuePtr &attr,
                                           const std::string &prim_name);
  static void CheckMinMaxShape(const ShapeVector &shape, ShapeVector *min_shape, ShapeVector *max_shape);
  static int64_t GetAndCheckFormat(const ValuePtr &value);
  static size_t GetRemoveMonadAbsNum(const AbstractBasePtrList &abs_list);
  static void CheckInputArgs(const std::vector<AbstractBasePtr> &input_args, const CompareEnum compare_operator,
                             const int64_t match_value, const std::string &prim_name);
  static bool HasDynamicShapeInput(const AbstractBasePtrList &abs_list);
  static void GetFormatStringVal(const PrimitivePtr &prim, std::string *format);

 private:
  static TypePtr _CheckTypeSame(const std::map<std::string, TypePtr> &args, const std::string &prim_name,
                                const bool allow_mix);
  static TypePtr CheckTensorSubClass(const std::string &type_name, const TypePtr &type,
                                     const std::set<TypePtr> &template_types, const std::string &prim_name,
                                     bool is_mix = false);
};
}  // namespace mindspore
#endif  // MINDSPORE_CORE_UTILS_CHECK_CONVERT_UTILS_H_
