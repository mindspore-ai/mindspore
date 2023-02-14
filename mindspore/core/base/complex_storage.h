/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CORE_BASE_COMPLEX_STORAGE_H_
#define MINDSPORE_CORE_BASE_COMPLEX_STORAGE_H_

#include "base/float16.h"
#include "utils/ms_utils.h"

namespace mindspore {
constexpr auto kComplexValueUnit = 2;

template <typename T>
struct alignas(sizeof(T) * kComplexValueUnit) ComplexStorage {
  T real_;
  T imag_;

  ComplexStorage() = default;
  ~ComplexStorage() = default;

  ComplexStorage(const ComplexStorage<T> &other) noexcept = default;
  ComplexStorage(ComplexStorage<T> &&other) noexcept = default;

  ComplexStorage &operator=(const ComplexStorage<T> &other) noexcept = default;
  ComplexStorage &operator=(ComplexStorage<T> &&other) noexcept = default;

  inline constexpr ComplexStorage(const T &real, const T &imag = T()) : real_(real), imag_(imag) {}
#ifndef ENABLE_ARM
  inline explicit constexpr ComplexStorage(const float16 &real) : real_(static_cast<T>(real)), imag_(T()) {}
#endif
  template <typename U = T>
  explicit ComplexStorage(const std::enable_if_t<std::is_same<U, float>::value, ComplexStorage<double>> &other)
      : real_(other.real_), imag_(other.imag_) {}

  template <typename U = T>
  explicit ComplexStorage(const std::enable_if_t<std::is_same<U, double>::value, ComplexStorage<float>> &other)
      : real_(other.real_), imag_(other.imag_) {}

  inline explicit operator bool() const { return static_cast<bool>(real_) || static_cast<bool>(imag_); }
  inline explicit operator signed char() const { return static_cast<signed char>(real_); }
  inline explicit operator unsigned char() const { return static_cast<unsigned char>(real_); }
  inline explicit operator double() const { return static_cast<double>(real_); }
  inline explicit operator float() const { return static_cast<float>(real_); }
  inline explicit operator int16_t() const { return static_cast<int16_t>(real_); }
  inline explicit operator uint16_t() const { return static_cast<uint16_t>(real_); }
  inline explicit operator int32_t() const { return static_cast<int32_t>(real_); }
  inline explicit operator uint32_t() const { return static_cast<uint32_t>(real_); }
  inline explicit operator int64_t() const { return static_cast<int64_t>(real_); }
  inline explicit operator uint64_t() const { return static_cast<uint64_t>(real_); }
  inline explicit operator float16() const { return static_cast<float16>(real_); }
};

template <typename T>
inline bool operator==(const ComplexStorage<T> &lhs, const ComplexStorage<T> &rhs) {
  if constexpr (std::is_same_v<T, double>) {
    return common::IsDoubleEqual(lhs.real_, rhs.real_) && common::IsDoubleEqual(lhs.imag_, rhs.imag_);
  } else if constexpr (std::is_same_v<T, float>) {
    return common::IsFloatEqual(lhs.real_, rhs.real_) && common::IsFloatEqual(lhs.imag_, rhs.imag_);
  }
  return (lhs.real_ == rhs.real_) && (lhs.imag_ == rhs.imag_);
}

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const ComplexStorage<T> &v) {
  return (os << std::noshowpos << v.real_ << std::showpos << v.imag_ << 'j');
}

}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_COMPLEX_STORAGE_H_
