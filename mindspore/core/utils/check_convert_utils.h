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
  UNKNOW = 19,
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
  NUM_OF_FORMAT = 14
};
enum EltwiseMode : int64_t { PROD = 0, SUM = 1, MAXIMUM = 2, ELTWISEMODE_UNKNOW = 3 };

class CheckAndConvertUtils {
 public:
  static std::vector<int64_t> CheckPositiveVector(const std::string &arg_name, const std::vector<int64_t> &arg_value,
                                                  const std::string &prim_name, bool allow_four = false,
                                                  bool ret_four = false);
  static std::string CheckString(const std::string &arg_name, const std::string &arg_value,
                                 const std::set<std::string> &check_list, const std::string &prim_name);
  static int CheckInteger(const std::string &arg_name, int arg_value, CompareEnum compare_operator, int match_value,
                          const std::string &prim_name);
  static void CheckInRange(const std::string &arg_name, int arg_value, CompareRange compare_operator,
                           const std::pair<int, int> &range, const std::string &prim_name);
  static std::vector<int64_t> ConvertShapePtrToShape(const std::string &arg_name, const BaseShapePtr &shape,
                                                     const std::string &prim_name);
  static void Check(const std::string &arg_name, int arg_value, CompareEnum compare_type, const std::string &value_name,
                    int value, const std::string &prim_name = "", ExceptionType exception_type = ValueError);
  static void Check(const std::string &arg_name, const std::vector<int64_t> &arg_value, CompareEnum compare_type,
                    const std::string &value_name, const std::vector<int64_t> &value, const std::string &prim_name = "",
                    ExceptionType exception_type = ValueError);
  static TypeId CheckTensorTypeSame(const std::map<std::string, TypePtr> &types, const std::set<TypeId> &check_list,
                                    const std::string &prim_name);

 private:
  static bool IsEqualVector(const std::vector<int64_t> &vec_1, const std::vector<int64_t> &vec_2);
};
}  // namespace mindspore
#endif  // MINDSPORE_CORE_UTILS_CHECK_CONVERT_UTILS_H_
