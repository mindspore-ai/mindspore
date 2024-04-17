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
 * \file op_common_util.h
 * \brief common util for op, in this file only original type or class in C++ allowed
 */

#ifndef CUSTOMIZE_OP_PROTO_UTIL_OP_COMMON_UTIL_H_
#define CUSTOMIZE_OP_PROTO_UTIL_OP_COMMON_UTIL_H_

#include <set>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>

template <typename T1, typename T2>
std::ostream &operator<<(std::ostream &os, const std::pair<T1, T2> &values) {
  os << "[" << values.first << ", " << values.second << "]";
  return os;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &values) {
  os << "[";
  for (const auto &item : values) {
    os << item << ", ";
  }
  os << "]";
  return os;
}

namespace ops {
template <typename T>
std::string to_string(const std::vector<T> &items) {
  std::ostringstream oss;
  oss << "[";
  for (const auto &item : items) {
    oss << item << ", ";
  }
  oss << "]";
  return oss.str();
}

template <typename T>
std::string to_string(const std::set<T> &items) {
  std::ostringstream oss;
  oss << "[";
  for (const auto &item : items) {
    oss << item << ", ";
  }
  oss << "]";
  return oss.str();
}
}  // namespace ops

#endif  // CUSTOMIZE_OP_PROTO_UTIL_OP_COMMON_UTIL_H_
