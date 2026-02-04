/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_CONTRACT_H
#define MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_CONTRACT_H
#include <string>
#include <exception>
#include <memory>
#include <utility>
#include <type_traits>
#include "utils/log_adapter.h"

namespace mindspore {
class ContractError : public std::logic_error {
 public:
  explicit ContractError(const std::string &msg) : std::logic_error(msg) {}
  explicit ContractError(const char *msg) : std::logic_error(msg) {}
  ~ContractError() override = default;
};

struct Signatory {
  LocationInfo location_info;
  const char *extra_info;
};

struct NotNullRule {
  template <class T>
  constexpr static bool Check(const T &val) {
    return val != nullptr;
  }
  constexpr static const char *Desc() { return " must not be null"; }
};
template <class T, class R, class E = void>
class EnsuresAccess {};

template <class T>
using RemoveCVR = std::remove_cv_t<std::remove_reference_t<T>>;

template <class T, class R>
class Ensures : public EnsuresAccess<T, R> {
 public:
  Ensures(T v, const Signatory &signatory) : value_(v) {
    if (!R::Check(value_)) {
      LogStream contract_stream;
      contract_stream << "contract error: " << signatory.extra_info << R::Desc();
      LogWriter(signatory.location_info, MsLogLevel::kException, SUBMODULE_ID, ArgumentError) ^ contract_stream;
    }
  }
  template <class O, typename = std::enable_if_t<std::is_convertible_v<O, T>>>
  Ensures(const Ensures<O, R> &other) : value_(other.get()) {}
  ~Ensures() = default;

  const T get() const { return value_; }
  T &get() { return value_; }

  operator const T() const { return value_; }

  T value_;
};

template <class T, class R>
class EnsuresAccess<T, R, std::enable_if_t<std::is_pointer_v<std::remove_cv_t<T>>>> {
 public:
  T operator->() const {
    auto ptr = static_cast<const Ensures<T, R> *>(this)->get();
    return ptr;
  }
};

template <typename T>
struct IsSharedPtr : public std::false_type {};
template <typename T>
struct IsSharedPtr<std::shared_ptr<T>> : public std::true_type {};
template <typename T>
struct IsSharedPtr<const std::shared_ptr<T> &> : public std::true_type {};

template <class T, class R>
class EnsuresAccess<T, R, std::enable_if_t<IsSharedPtr<T>::value>> {
  using element_type = typename std::remove_cv_t<std::remove_reference_t<T>>::element_type;

 public:
  element_type *operator->() const {
    auto ptr = static_cast<const Ensures<T, R> *>(this)->get();
    return ptr.get();
  }
};

template <class T, class R>
using Expects = Ensures<T, R>;
template <class T>
using NotNull = Ensures<T, NotNullRule>;
#define ENSURE(_v, _rule) Ensures<decltype(_v), _rule>(_v, {{__FILE__, __LINE__, __FUNCTION__}, #_v})
#define NOT_NULL(_v) ENSURE(_v, NotNullRule)
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_CONTRACT_H
