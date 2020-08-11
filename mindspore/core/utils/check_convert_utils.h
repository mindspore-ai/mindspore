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

#ifndef MINDSPORE_CORE_UTILS_CHECK_CONVERT_UTILS_H
#define MINDSPORE_CORE_UTILS_CHECK_CONVERT_UTILS_H
#include <vector>
#include <string>
#include <map>
#include <set>
#include <utility>
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/type_id.h"
#include "utils/log_adapter.h"
namespace mindspore {
enum CompareEnum : int {
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

class CheckAndConvertUtils {
 public:
  static std::vector<int> CheckPositiveVector(const std::string &arg_name, const std::vector<int> &arg_value,
                                              const std::string &prim_name, bool allow_four = false,
                                              bool ret_four = false);
  static std::string CheckString(const std::string &arg_name, const std::string &arg_value,
                                 const std::set<std::string> &check_list, const std::string &prim_name);
  static int CheckInteger(const std::string &arg_name, int arg_value, CompareEnum compare_operator, int match_value,
                          const std::string &prim_name);
  static void CheckInRange(const std::string &arg_name, int arg_value, CompareRange compare_operator,
                           const std::pair<int, int> &range, const std::string &prim_name);
  static std::vector<int> ConvertShapePtrToShape(const std::string &arg_name, const BaseShapePtr &shape,
                                                 const std::string &prim_name);
  static void Check(const std::string &arg_name, int arg_value, CompareEnum compare_type, const std::string &value_name,
                    int value, const std::string &prim_name = "", ExceptionType exception_type = ValueError);
  static void Check(const std::string &arg_name, const std::vector<int> &arg_value, CompareEnum compare_type,
                    const std::string &value_name, const std::vector<int> &value, const std::string &prim_name = "",
                    ExceptionType exception_type = ValueError);
  static TypeId CheckTensorTypeSame(const std::map<std::string, TypePtr> &types, const std::set<TypeId> &check_list,
                                    const std::string &prim_name);

 private:
  static bool IsEqualVector(const std::vector<int> &vec_1, const std::vector<int> &vec_2);
};
}  // namespace mindspore
#endif  // MINDSPORE_CORE_UTILS_CHECK_CONVERT_UTILS_H
