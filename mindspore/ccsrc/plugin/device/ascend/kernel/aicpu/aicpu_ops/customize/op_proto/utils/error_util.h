/**
 * Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
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

/*!
 * \file error_util.h
 * \brief
 */
#ifndef CUSTOMIZE_OP_PROTO_UTILS_ERROR_UTIL_H_
#define CUSTOMIZE_OP_PROTO_UTILS_ERROR_UTIL_H_

#include <sstream>
#include <string>
#include <vector>
#include "error_code.h"
#include "graph/operator.h"
#include "op_proto/utils/op_log.h"

#define AICPU_INFER_SHAPE_CALL_ERR_REPORT(op_name, err_msg)   \
  do {                                                        \
    OP_LOGE_WITHOUT_REPORT(op_name, "%s", get_cstr(err_msg)); \
  } while (0)

#define AICPU_INFER_SHAPE_INNER_ERR_REPORT(op_name, err_msg)  \
  do {                                                        \
    OP_LOGE_WITHOUT_REPORT(op_name, "%s", get_cstr(err_msg)); \
  } while (0)

#define VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op_name, err_msg) \
  do {                                                        \
    OP_LOGE_WITHOUT_REPORT(op_name, "%s", get_cstr(err_msg)); \
  } while (0)

#define INFER_AXIS_TYPE_ERR_REPORT(op_name, err_msg, ...)    \
  do {                                                       \
    OP_LOGE_WITHOUT_REPORT(op_name, err_msg, ##__VA_ARGS__); \
  } while (0)

#define VECTOR_FUSION_INNER_ERR_REPORT(op_name, err_msg, ...) \
  do {                                                        \
    OP_LOGE_WITHOUT_REPORT(op_name, err_msg, ##__VA_ARGS__);  \
  } while (0)

#define VECTOR_CHECK_NULLPTR_RETURN_WITH_REPORT(op_name, ptr, ret, err_msg, ...) \
  do {                                                                           \
    if ((ptr) == nullptr) {                                                      \
      VECTOR_FUSION_INNER_ERR_REPORT(op_name, err_msg, ##__VA_ARGS__);           \
      return (ret);                                                              \
    }                                                                            \
  } while (0)

#define CUBE_INNER_ERR_REPORT(op_name, err_msg, ...)         \
  do {                                                       \
    OP_LOGE_WITHOUT_REPORT(op_name, err_msg, ##__VA_ARGS__); \
  } while (0)

#define CUBE_CALL_ERR_REPORT(op_name, err_msg, ...)          \
  do {                                                       \
    OP_LOGE_WITHOUT_REPORT(op_name, err_msg, ##__VA_ARGS__); \
  } while (0)

#define CUBE_INNER_ERR_REPORT_PLUGIN(op_name, err_msg, ...)  \
  do {                                                       \
    OP_LOGE_WITHOUT_REPORT(op_name, err_msg, ##__VA_ARGS__); \
  } while (0)

#define CUBE_CALL_ERR_REPORT_PLUGIN(op_name, err_msg, ...)   \
  do {                                                       \
    OP_LOGE_WITHOUT_REPORT(op_name, err_msg, ##__VA_ARGS__); \
  } while (0)

namespace optiling {
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) \
  do {                                                         \
    OP_LOGE_WITHOUT_REPORT(op_name, err_msg, ##__VA_ARGS__);   \
  } while (0)

#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)
}  // namespace optiling

namespace ge {
/*
 * get debug string of vector
 * param[in] v vector
 * return vector's debug string
 */
template <typename T>
std::string DebugString(const std::vector<T> &v) {
  std::ostringstream oss;
  oss << "[";
  if (v.size() > 0) {
    for (size_t i = 0; i < v.size() - 1; ++i) {
      oss << v[i] << ", ";
    }
    oss << v[v.size() - 1];
  }
  oss << "]";
  return oss.str();
}

template <typename T>
std::string DebugString(const std::vector<std::pair<T, T>> &v) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i != 0) {
      oss << ", ";
    }
    oss << "(" << v[i].first << ", " << v[i].second << ")";
  }
  oss << "]";
  return oss.str();
}

inline std::ostream &operator<<(std::ostream &os, const ge::Operator &op) { return os << get_op_info(op); }

/*
 * str cat util function
 * param[in] params need concat to string
 * return concatted string
 */
template <typename T>
std::string ConcatString(const T &arg) {
  std::ostringstream oss;
  oss << arg;
  return oss.str();
}

template <typename T, typename... Ts>
std::string ConcatString(const T &arg, const Ts &... arg_left) {
  std::ostringstream oss;
  oss << arg;
  oss << ConcatString(arg_left...);
  return oss.str();
}

template <typename T>
std::string Shape2String(const T &shape) {
  std::ostringstream oss;
  oss << "[";
  if (shape.GetDimNum() > 0) {
    for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
      oss << shape.GetDim(i) << ", ";
    }
    oss << shape.GetDim(shape.GetDimNum() - 1);
  }
  oss << "]";
  return oss.str();
}

std::string GetViewErrorCodeStr(ge::ViewErrorCode errCode);

std::string GetShapeErrMsg(uint32_t index, const std::string &wrong_shape, const std::string &correct_shape);

std::string GetAttrValueErrMsg(const std::string &attr_name, const std::string &wrong_val,
                               const std::string &correct_val);

std::string GetAttrSizeErrMsg(const std::string &attr_name, const std::string &wrong_size,
                              const std::string &correct_size);

std::string GetInputInvalidErrMsg(const std::string &param_name);
std::string GetShapeSizeErrMsg(uint32_t index, const std::string &wrong_shape_size,
                               const std::string &correct_shape_size);

std::string GetInputFormatNotSupportErrMsg(const std::string &param_name, const std::string &expected_format_list,
                                           const std::string &data_format);

std::string GetInputDtypeNotSupportErrMsg(const std::string &param_name, const std::string &expected_dtype_list,
                                          const std::string &data_dtype);

std::string GetInputDTypeErrMsg(const std::string &param_name, const std::string &expected_dtype,
                                const std::string &data_dtype);

std::string GetInputFormatErrMsg(const std::string &param_name, const std::string &expected_format,
                                 const std::string &data_format);

std::string SetAttrErrMsg(const std::string &param_name);
std::string UpdateParamErrMsg(const std::string &param_name);

template <typename T>
std::string GetParamOutRangeErrMsg(const std::string &param_name, const T &real_value, const T &min, const T &max);

std::string OtherErrMsg(const std::string &error_detail);
}  // namespace ge

#endif  // CUSTOMIZE_OP_PROTO_UTILS_ERROR_UTIL_H_
