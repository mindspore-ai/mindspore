/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_BIT_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_BIT_H_

namespace mindspore {
namespace dataset {
template <typename Enum>
Enum operator|(Enum lhs, Enum rhs) {
  static_assert(std::is_enum<Enum>::value, "template parameter is not an enum type");
  using underlying = typename std::underlying_type<Enum>::type;
  return static_cast<Enum>(static_cast<underlying>(lhs) | static_cast<underlying>(rhs));
}

template <typename Enum>
Enum operator&(Enum lhs, Enum rhs) {
  static_assert(std::is_enum<Enum>::value, "template parameter is not an enum type");
  using underlying = typename std::underlying_type<Enum>::type;
  return static_cast<Enum>(static_cast<underlying>(lhs) & static_cast<underlying>(rhs));
}

template <typename Enum>
Enum operator^(Enum lhs, Enum rhs) {
  static_assert(std::is_enum<Enum>::value, "template parameter is not an enum type");
  using underlying = typename std::underlying_type<Enum>::type;
  return static_cast<Enum>(static_cast<underlying>(lhs) ^ static_cast<underlying>(rhs));
}

template <typename Enum>
Enum &operator|=(Enum &lhs, Enum rhs) {
  static_assert(std::is_enum<Enum>::value, "template parameter is not an enum type");
  using underlying = typename std::underlying_type<Enum>::type;
  lhs = static_cast<Enum>(static_cast<underlying>(lhs) | static_cast<underlying>(rhs));
  return lhs;
}

template <typename Enum>
Enum &operator&=(Enum &lhs, Enum rhs) {
  static_assert(std::is_enum<Enum>::value, "template parameter is not an enum type");
  using underlying = typename std::underlying_type<Enum>::type;
  lhs = static_cast<Enum>(static_cast<underlying>(lhs) & static_cast<underlying>(rhs));
  return lhs;
}

template <typename Enum>
Enum &operator^=(Enum &lhs, Enum rhs) {
  static_assert(std::is_enum<Enum>::value, "template parameter is not an enum type");
  using underlying = typename std::underlying_type<Enum>::type;
  lhs = static_cast<Enum>(static_cast<underlying>(lhs) ^ static_cast<underlying>(rhs));
  return lhs;
}

template <typename Enum>
Enum operator~(Enum v) {
  static_assert(std::is_enum<Enum>::value, "template parameter is not an enum type");
  using underlying = typename std::underlying_type<Enum>::type;
  return static_cast<Enum>(~static_cast<underlying>(v));
}
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_BIT_H_
