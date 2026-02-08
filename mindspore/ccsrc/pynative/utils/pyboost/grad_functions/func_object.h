/**
 * Copyright 2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF FuncObject KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <exception>
#include <typeinfo>
#include <type_traits>
#include <typeindex>
#include <utility>
#include "utils/log_adapter.h"

namespace mindspore::runtime {
class FuncObject {
 public:
  FuncObject() = default;
  template <typename T>
  explicit FuncObject(const T &value)
      : object_(new Object<typename std::remove_cv_t<typename std::decay_t<const T>>>(value)) {}
  FuncObject(const FuncObject &other) : object_(other.object_ ? other.object_->clone() : nullptr) {}
  FuncObject(FuncObject &&other) : object_(other.object_) { other.object_ = nullptr; }
  template <typename T>
  explicit FuncObject(T &&value) : object_(new Object<typename std::decay_t<T>>(static_cast<T &&>(value))) {}

  ~FuncObject() {
    delete object_;
    object_ = nullptr;
  }

 public:
  FuncObject &swap(FuncObject &rhs) {
    std::swap(object_, rhs.object_);
    return *this;
  }

  template <typename T>
  FuncObject &operator=(T &&rhs) {
    // cppcheck-suppress *
    (void)FuncObject(static_cast<T &&>(rhs)).swap(*this);
    return *this;
  }

  const std::type_info &type() const { return object_ ? object_->type() : typeid(void); }

  template <typename T>
  friend T *Cast(FuncObject *func_object);

 private:
  class Handle {
   public:
    virtual ~Handle() {}
    virtual const std::type_info &type() const = 0;
    virtual Handle *clone() const = 0;
  };

  template <typename T>
  class Object : public Handle {
   public:
    Object &operator=(const Object &) = delete;

    explicit Object(const T &value) : handle_(value) {}
    explicit Object(T &&value) : handle_(static_cast<T &&>(value)) {}
    const std::type_info &type() const override { return typeid(T); }
    Handle *clone() const override { return new Object(handle_); }
    T handle_;
  };

  Handle *object_{nullptr};
};

template <bool, typename T1, typename T2>
struct If {
  using type = T2;
};

template <typename T1, typename T2>
struct If<true, T1, T2> {
  using type = T1;
};

template <bool b, typename T1, typename T2>
using If_t = typename If<b, T1, T2>::type;

template <typename T>
T *Cast(FuncObject *func_object) {
  return func_object && func_object->type() == typeid(T)
           ? &(static_cast<FuncObject::Object<typename std::remove_cv_t<T>> *>(func_object->object_)->handle_)
           : nullptr;
}

template <typename T>
// cppcheck-suppress *
T FuncCast(FuncObject &func_object) {  // NOLINT
  using object_type = typename std::remove_reference_t<T>;
  object_type *result = Cast<object_type>(&func_object);
  if (!result) {
    MS_LOG(EXCEPTION) << "Can not convert from " << func_object.type().name() << " to " << typeid(T).name();
  }
  using ref_type = If_t<std::is_reference_v<T>, T, std::add_lvalue_reference_t<T>>;
  return static_cast<ref_type>(*result);
}

}  // namespace mindspore::runtime
