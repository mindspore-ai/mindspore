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
 * \file op_util.h
 * \brief
 */

#ifndef CANN_CUSTOMIZE_OP_UTIL_H_
#define CANN_CUSTOMIZE_OP_UTIL_H_

#include <memory>
#include <utility>
#include <vector>
#include <string>

#include "error_util.h"

namespace ops {
template <typename _T, typename... _Args>
inline std::shared_ptr<_T> make_shared_nothrow(_Args &&... _args) noexcept(
  noexcept(_T(std::forward<_Args>(_args)...))) {
  try {
    return std::make_shared<_T>(std::forward<_Args>(_args)...);
  } catch (...) {
    return std::shared_ptr<_T>();
  }
}

template <typename T1, typename T2>
bool IsDimValid(const T1 shape_size, const T2 dim_value) {
  int64_t minimum_num = static_cast<int64_t>(shape_size) * (-1);
  int64_t maximum_num = static_cast<int64_t>(shape_size) - 1;

  return static_cast<int64_t>(dim_value) >= minimum_num && static_cast<int64_t>(dim_value) <= maximum_num;
}

template <typename T1, typename T2>
std::string GenInvalidDimMsg(const std::string &dim_name, const T1 shape_size, const T2 dim_value) {
  std::string wrong_val = ge::ConcatString(static_cast<int64_t>(dim_value));
  // will be "[-rank, rank)"
  std::string neg_rank = ge::ConcatString(static_cast<int64_t>(shape_size) * (-1));
  std::string expect_val =
    ge::ConcatString("[", neg_rank, ", ", ge::ConcatString(static_cast<int64_t>(shape_size)), ")");

  return ge::GetAttrValueErrMsg(dim_name, wrong_val, expect_val);
}

template <typename T1, typename T2>
std::string GenInvalidDimMsg(const std::string &dim_name, const size_t dim_idx, const T1 shape_size,
                             const T2 dim_value) {
  std::string invalid_dim_name = ge::ConcatString(dim_name, "[", ge::ConcatString(dim_idx), "]");

  return GenInvalidDimMsg(invalid_dim_name, shape_size, dim_value);
}

template <typename T>
T CeilDiv(T x, T y) {
  return y == 0 ? x : (x + y - 1) / y;
}

/*
 * @brief: get datatype string from enum
 * @param [in] type: enum datatype
 * @return string: datatype string
 */
std::string ToString(const ge::DataType &type);

/*
 * @brief: get format string from enum
 * @param [in] format: enum format
 * @return string: format string
 */
std::string ToString(const ge::Format &format);

/*
 * @brief: get shape string from ge::Shape, for debug
 * @param [in] shape: reference of ge::Shape
 * @return string: shape string
 */
std::string ToString(const ge::Shape &shape);

/*
 * @brief: get shape string from ge::Shape, for debug
 * @param [in] shape: ptr of ge::Shape
 * @return string: shape string
 */
std::string ToString(const ge::Shape *shape);

std::string ToString(const std::vector<int64_t> &shape);
std::string ToString(const std::vector<ge::Shape> &shapes);

/*
 * @brief: trans the ge::Shape to vector<int64_t>
 * @param [in] format: ge::Shape
 * @return vector<int64_t>: the vector shape
 */
std::vector<int64_t> ToVector(const ge::Shape &shape);

/*
 * @brief: floor(u_value/d_value)
 *         eg. GetFloorDiv(7,3) -> 2, GetFloorDiv(4,2) -> 2, GetFloorDiv(4,0) -> 4
 *
 * @param [in] u_value: int64_t
 * @param [in] d_value: int64_t
 * @return int64: floor
 */
int64_t GetFloorDiv(int64_t u_value, int64_t d_value);

/*
 * @brief: ceil(u_value/d_value)
 *         eg. GetCeilDiv(7,3) -> 3, GetFloorDiv(4,2) -> 2, GetFloorDiv(4,0) -> 4
 *
 * @param [in] u_value: int64_t
 * @param [in] d_value: int64_t
 * @return int64: ceil
 */
int64_t GetCeilDiv(int64_t u_value, int64_t d_value);

/*
 * @brief: floor(u_value/d_value) * d_value
 *         eg. GetDivisorAlign(7,3) -> 6, GetDivisorAlign(4,2) -> 4, GetDivisorAlign(4,0) -> 4
 *
 * @param [in] u_value: int64_t
 * @param [in] d_value: int64_t
 * @return int64: floor*d_value
 */
int64_t GetDivisorAlign(const int64_t u_value, const int64_t d_value);

/*
 * @brief: l_value % r_value
 *
 * @param [in] l_value: int64_t
 * @param [in] r_value: int64_t
 * @return int64: mod
 */
int64_t GetMod(const int64_t l_value, const int64_t r_value);

/*
 * @brief: trans the const buffer to string
 * @param [in] value: const T*
 * @param [in] size: size_t
 * @return string
 */
template <typename T>
std::string ToString(const T *value, size_t size) {
  std::string r = "[";
  for (size_t i = 0; i < size; i++) {
    r = r + std::to_string(value[i]) + ", ";
  }
  r = r + "]";
  return r;
}
}  // namespace ops
#endif  // CANN_CUSTOMIZE_OP_UTIL_H_
