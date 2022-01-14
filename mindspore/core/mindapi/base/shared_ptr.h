/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_MINDAPI_BASE_SHARED_PTR_H_
#define MINDSPORE_CORE_MINDAPI_BASE_SHARED_PTR_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <ostream>
#include <functional>

namespace mindspore::api {
/// \brief SharedPtr wraps a std::shared_ptr and provides wrapper functions according the underlying implementation.
template <typename T>
class SharedPtr {
 public:
  using element_type = T;
  constexpr SharedPtr() noexcept = default;
  constexpr SharedPtr(std::nullptr_t) noexcept : SharedPtr() {}  // NOLINT
  template <typename U>
  explicit SharedPtr(std::shared_ptr<U> &&ptr) : ptr_(std::move(ptr)) {}
  template <typename U>
  SharedPtr(const SharedPtr<U> &other) : ptr_(other.ptr_) {}
  template <typename U>
  SharedPtr(SharedPtr<U> &&other) : ptr_(std::move(other.ptr_)) {}
  template <typename U>
  SharedPtr &operator=(const SharedPtr<U> &other) {
    ptr_ = other.ptr_;
    return *this;
  }
  template <typename U>
  SharedPtr &operator=(SharedPtr<U> &&other) {
    ptr_ = std::move(other.ptr_);
    return *this;
  }
  ~SharedPtr() = default;

  std::uintptr_t addr() const { return (ptr_ == nullptr) ? 0 : reinterpret_cast<std::uintptr_t>(ptr_->impl().get()); }
  element_type &operator*() const noexcept { return *ptr_; }
  element_type *operator->() const noexcept { return ptr_.get(); }
  element_type *get() const noexcept { return ptr_.get(); }
  explicit operator bool() const { return addr() != 0; }

 private:
  template <typename U>
  friend class SharedPtr;
  std::shared_ptr<element_type> ptr_;
};

template <typename T, typename U>
inline bool operator==(const SharedPtr<T> &a, const SharedPtr<U> &b) noexcept {
  return a.addr() == b.addr();
}

template <typename T>
inline bool operator==(const SharedPtr<T> &a, std::nullptr_t) noexcept {
  return a.addr() == 0;
}

template <typename T>
inline bool operator==(std::nullptr_t, const SharedPtr<T> &a) noexcept {
  return a.addr() == 0;
}

template <typename T, typename U>
inline bool operator!=(const SharedPtr<T> &a, const SharedPtr<U> &b) noexcept {
  return a.addr() != b.addr();
}

template <typename T>
inline bool operator!=(const SharedPtr<T> &a, std::nullptr_t) noexcept {
  return a.addr() != 0;
}

template <typename T>
inline bool operator!=(std::nullptr_t, const SharedPtr<T> &a) noexcept {
  return a.addr() != 0;
}

template <typename T, typename U>
inline bool operator<(const SharedPtr<T> &a, const SharedPtr<U> &b) noexcept {
  return a.addr() < b.addr();
}

template <typename T>
inline bool operator<(const SharedPtr<T> &a, std::nullptr_t) noexcept {
  return a.addr() < 0;
}

template <typename T>
inline bool operator<(std::nullptr_t, const SharedPtr<T> &a) noexcept {
  // 'nullptr < ptr' is false only when ptr is nullptr.
  return a.addr() != 0;
}

template <typename T, typename U>
inline bool operator>(const SharedPtr<T> &a, const SharedPtr<U> &b) noexcept {
  return a.addr() > b.addr();
}

template <typename T>
inline bool operator>(const SharedPtr<T> &a, std::nullptr_t) noexcept {
  return a.addr() > 0;
}

template <typename T>
inline bool operator>(std::nullptr_t, const SharedPtr<T> &) noexcept {
  // 'nullptr > ptr' is always false.
  return false;
}

template <typename T, typename U>
inline bool operator<=(const SharedPtr<T> &a, const SharedPtr<U> &b) noexcept {
  return a.addr() <= b.addr();
}

template <typename T>
inline bool operator<=(const SharedPtr<T> &a, std::nullptr_t) noexcept {
  return a.addr() <= 0;
}

template <typename T>
inline bool operator<=(std::nullptr_t, const SharedPtr<T> &) noexcept {
  // 'nullptr <= ptr' is always true.
  return true;
}

template <typename T, typename U>
inline bool operator>=(const SharedPtr<T> &a, const SharedPtr<U> &b) noexcept {
  return a.addr() >= b.addr();
}

template <typename T>
inline bool operator>=(const SharedPtr<T> &a, std::nullptr_t) noexcept {
  return a.addr() >= 0;
}

template <typename T>
inline bool operator>=(std::nullptr_t, const SharedPtr<T> &a) noexcept {
  // 'nullptr >= ptr' is true only when ptr is nullptr.
  return a.addr() == 0;
}

template <typename T, typename U, typename V>
inline std::basic_ostream<U, V> &operator<<(std::basic_ostream<U, V> &os, const SharedPtr<T> &a) {
  return (os << reinterpret_cast<void *>(a.addr()));
}

/// \brief Constructs an object of type T and wraps it in a SharedPtr.
///
/// \param[in] args The parameter list for the constructor of T.
template <typename T, typename... Args>
inline SharedPtr<T> MakeShared(Args &&... args) {
  auto ptr = std::make_shared<T>(std::forward<Args>(args)...);
  return SharedPtr<T>(std::move(ptr));
}
}  // namespace mindspore::api

namespace std {
template <typename T>
struct hash<mindspore::api::SharedPtr<T>> {
  size_t operator()(const mindspore::api::SharedPtr<T> &ptr) const noexcept { return static_cast<size_t>(ptr.addr()); }
};
}  // namespace std

#endif  // MINDSPORE_CORE_MINDAPI_BASE_SHARED_PTR_H_
