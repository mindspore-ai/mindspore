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

#ifndef MINDSPORE_CORE_UTILS_CHECK_CONVERT_UTILS_H_
#define MINDSPORE_CORE_UTILS_CHECK_CONVERT_UTILS_H_
#include <vector>
#include <string>
#include <map>
#include <set>
#include <utility>
#include <typeinfo>
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/type_id.h"
#include "utils/log_adapter.h"
namespace mindspore {
typedef std::pair<std::map<std::string, int64_t>, std::map<int64_t, std::string>> AttrConverterPair;

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
enum Format : int64_t {
  NCHW = 0,
  NHWC = 1,
  NHWC4 = 2,
  HWKC = 3,
  HWCK = 4,
  KCHW = 5,
  CKHW = 6,
  KHWC = 7,
  CHWK = 8,
  HW = 9,
  HW4 = 10,
  NC = 11,
  NC4 = 12,
  NC4HW4 = 13,
  NUM_OF_FORMAT = 14,
  NCDHW = 15
};
enum ActivationType : int64_t {
  NO_ACTIVATION = 0,
  RELU = 1,
  SIGMOID = 2,
  RELU6 = 3,
  ELU = 4,
  LEAKY_RELU = 5,
  ABS = 6,
  RELU1 = 7,
  SOFTSIGN = 8,
  SOFTPLUS = 9,
  TANH = 10,
  SELU = 11,
  HSWISH = 12,
  HSIGMOID = 13,
  THRESHOLDRELU = 14,
  LINEAR = 15,
  HARD_TANH = 16,
  SIGN = 17,
  SWISH = 18,
  GELU = 19,
  UNKNOWN = 20
};
enum ReduceMode : int64_t {
  Reduce_Mean = 0,
  Reduce_Max = 1,
  Reduce_Min = 2,
  Reduce_Prod = 3,
  Reduce_Sum = 4,
  Reduce_Sum_Square = 5,
  Reduce_ASum = 6,
  Reduce_All = 7
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
enum EltwiseMode : int64_t { PROD = 0, SUM = 1, MAXIMUM = 2, ELTWISEMODE_UNKNOW = 3 };

enum Reduction : int64_t { REDUCTION_SUM = 0, MEAN = 1, NONE = 2 };

enum PadMode : int64_t { PAD = 0, SAME = 1, VALID = 2 };

enum RoundMode : int64_t {
  FLOOR = 0,
  CEIL = 1,
};

enum PoolMode : int64_t {
  MAX_POOLING = 0,
  MEAN_POOLING = 1,
};

enum GateOrderMode : int64_t { RZH = 0, ZRH = 1 };

enum class LshProjectionType : int64_t { UNKNOWN = 0, SPARSE = 1, DENSE = 2 };

enum PaddingMode : int64_t { CONSTANT = 0, REFLECT = 1, SYMMETRIC = 2, MODE_RESERVED = 3 };

enum class ResizeMethod : int64_t { UNKNOWN = -1, LINEAR = 0, NEAREST = 1, CUBIC = 2 };

enum CoordinateTransformMode : int64_t { ASYMMETRIC = 0, ALIGN_CORNERS = 1, HALF_PIXEL = 2, CROP_AND_RESIZE = 3 };

enum class NearestMode : int64_t { NORMAL = 0, ROUND_HALF_DOWN = 1, ROUND_HALF_UP = 2, FLOOR = 3, CEIL = 4 };

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
  {kEqual, "equal "},          {kNotEqual, "not equal "},       {kLessThan, "less than "},
  {kLessEqual, "less equal "}, {kGreaterThan, "greater than "}, {kGreaterEqual, "greater equal "}};

const std::map<CompareRange, std::pair<std::string, std::string>> kCompareRangeToString = {
  {kIncludeNeither, {"in (", ")"}},
  {kIncludeLeft, {" in [", ")"}},
  {kIncludeRight, {"in (", "]"}},
  {kIncludeBoth, {"in [", "]"}}};

class CheckAndConvertUtils {
 public:
  static std::vector<int64_t> CheckPositiveVector(const std::string &arg_name, const std::vector<int64_t> &arg_value,
                                                  const std::string &prim_name, bool allow_four = false,
                                                  bool ret_four = false);
  static std::string CheckString(const std::string &arg_name, const std::string &arg_value,
                                 const std::set<std::string> &check_list, const std::string &prim_name);

  // CheckValue should replace CheckInteger
  static int64_t CheckInteger(const std::string &arg_name, int64_t arg_value, CompareEnum compare_operator,
                              int64_t match_value, const std::string &prim_name);

  template <typename T>
  static T CheckValue(const std::string &arg_name, T arg_value, CompareEnum compare_operator, T match_value,
                      const std::string &prim_name) {
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
      MS_EXCEPTION(NotExistsError) << "compare_operator " << compare_operator
                                   << " cannot find in the compare string map";
    }
    buffer << iter_to_string->second << match_value << " , but got " << arg_value;
    MS_EXCEPTION(ValueError) << buffer.str();
  }

  template <typename T>
  static void CheckInRange(const std::string &arg_name, T arg_value, CompareRange compare_operator,
                           const std::pair<T, T> &range, const std::string &prim_name) {
    auto iter = kCompareRangeMap<float>.find(compare_operator);
    if (iter == kCompareRangeMap<float>.end()) {
      MS_EXCEPTION(NotExistsError) << "compare_operator " << compare_operator << " cannot find in the compare map";
    }
    if (range.first >= range.second) {
      MS_EXCEPTION(ArgumentError) << "the check range left must be larger than right number bug got [ " << range.first
                                  << "," << range.second;
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
      MS_EXCEPTION(NotExistsError) << "compare_operator " << compare_operator
                                   << " cannot find in the compare string map";
    }
    auto range_strng = iter_to_string->second;
    buffer << range_strng.first << range.first << "," << range_strng.second << " , but got " << arg_value;
    MS_EXCEPTION(ValueError) << buffer.str();
  }

  static std::vector<int64_t> ConvertShapePtrToShape(const std::string &arg_name, const BaseShapePtr &shape,
                                                     const std::string &prim_name);
  static void Check(const std::string &arg_name, int64_t arg_value, CompareEnum compare_type,
                    const std::string &value_name, int64_t value, const std::string &prim_name = "",
                    ExceptionType exception_type = ValueError);

  template <typename T>
  static void Check(const std::string &arg_name, const std::vector<T> &arg_value, CompareEnum compare_type,
                    const std::string &value_name, const std::vector<T> &value, const std::string &prim_name = "",
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

  static TypeId CheckTensorTypeSame(const std::map<std::string, TypePtr> &types, const std::set<TypeId> &check_list,
                                    const std::string &prim_name);
  static void CheckTensorTypeValid(const std::string &type_name, const TypePtr type, const std::set<TypeId> &check_list,
                                   const std::string &prim_name);
  static void CheckSubClass(const std::string &type_name, const TypePtr type, const std::set<TypePtr> &template_types,
                            const std::string &prim_name);
  static void CheckScalarOrTensorTypesSame(const std::map<std::string, TypePtr> &args,
                                           const std::set<TypeId> &valid_values, const std::string &prim_name,
                                           bool allow_mix = false);
  static TypeId CheckTypeSame(const std::string &arg_name, const TypePtr arg_type, const std::set<TypeId> &valid_type,
                              const std::string &prim_name);
  static bool ConvertAttrValueToInt(const std::string &op_type, const std::string &attr_name, ValuePtr *const value);
  static bool ConvertAttrValueToString(const std::string &op_type, const std::string &attr_name, ValuePtr *const value);
  static void ConvertAttrValueInExport(const std::string &op_type, const std::string &attr_name, ValuePtr *const value);
  static void ConvertAttrValueInLoad(const std::string &op_type, const std::string &attr_name, ValuePtr *const value);
  static AttrConverterPair GetAttrConvertPair(const std::string &op_type, const std::string &attr_name);
  static bool GetDataFormatEnumValue(const ValuePtr &value, int64_t *enum_value);
  static void GetPadModEnumValue(const ValuePtr &value, int64_t *enum_value, bool is_upper = false);
  static bool CheckIrAttrtoOpAttr(const std::string &op_type, const std::string &attr_name, ValuePtr *const value);

 private:
  static bool IsEqualVector(const std::vector<int64_t> &vec_1, const std::vector<int64_t> &vec_2);
  static std::map<std::string, TypePtr> _CheckArgumentType(const std::map<std::string, TypePtr> &arg,
                                                           const std::set<TypeId> &valid_values,
                                                           const std::string &prim_name);
  static std::map<std::string, TypePtr> _CheckTypeSame(const std::map<std::string, TypePtr> &arg1,
                                                       const std::map<std::string, TypePtr> &arg2,
                                                       const std::string &prim_name, const bool allow_mix);
};
}  // namespace mindspore
#endif  // MINDSPORE_CORE_UTILS_CHECK_CONVERT_UTILS_H_
