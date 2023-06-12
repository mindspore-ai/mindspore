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
#ifndef MINDSPORE_CORE_UTILS_ANY_H_
#define MINDSPORE_CORE_UTILS_ANY_H_

#include <iostream>
#include <string>
#include <typeinfo>
#include <typeindex>
#include <memory>
#include <functional>
#include <sstream>
#include <vector>
#include <utility>
#include "utils/overload.h"
#include "utils/log_adapter.h"
#include "utils/misc.h"

namespace mindspore {
// usage:AnyPtr sp = std::make_shared<Any>(aname);
template <class T>
std::string type(const T &t) {
  return demangle(typeid(t).name());
}

class MS_CORE_API Any {
 public:
  // constructors
  Any() : m_ptr(nullptr), m_tpIndex(std::type_index(typeid(void))) {}
  Any(const Any &other) : m_ptr(other.clone()), m_tpIndex(other.m_tpIndex) {}
  Any(Any &&other) : m_ptr(std::move(other.m_ptr)), m_tpIndex(std::move(other.m_tpIndex)) {}

  Any &operator=(Any &&other);
  // right reference constructor
  template <class T, class = typename std::enable_if<!std::is_same<typename std::decay<T>::type, Any>::value, T>::type>
  Any(T &&t) : m_tpIndex(typeid(typename std::decay<T>::type)) {  // NOLINT
    BasePtr new_val = std::make_unique<Derived<typename std::decay<T>::type>>(std::forward<T>(t));
    std::swap(m_ptr, new_val);
  }

  ~Any() = default;

  // judge whether is empty
  bool empty() const { return m_ptr == nullptr; }

  // judge the is relation
  template <class T>
  bool is() const {
    return m_tpIndex == std::type_index(typeid(T));
  }

  const std::type_info &type() const { return m_ptr ? m_ptr->type() : typeid(void); }

  std::size_t Hash() const {
    std::stringstream buffer;
    buffer << m_tpIndex.name();
    if (m_ptr != nullptr) {
      buffer << m_ptr->GetString();
    }
    return std::hash<std::string>()(buffer.str());
  }

  template <typename T>
  bool Apply(const std::function<void(T &)> &fn) {
    if (type() == typeid(T)) {
      T x = cast<T>();
      fn(x);
      return true;
    }
    return false;
  }

  std::string GetString() const {
    if (m_ptr != nullptr) {
      return m_ptr->GetString();
    } else {
      return std::string("");
    }
  }

  friend std::ostream &operator<<(std::ostream &os, const Any &any) {
    os << any.GetString();
    return os;
  }

  // type cast
  template <class T>
  T &cast() const {
    if (!is<T>() || !m_ptr) {
      // Use MS_LOGFATAL replace throw std::bad_cast()
      MS_LOG(EXCEPTION) << "can not cast " << m_tpIndex.name() << " to " << typeid(T).name();
    }
    auto ptr = static_cast<Derived<T> *>(m_ptr.get());
    return ptr->m_value;
  }

  bool operator==(const Any &other) const {
    if (m_tpIndex != other.m_tpIndex) {
      return false;
    }
    if (m_ptr == nullptr && other.m_ptr == nullptr) {
      return true;
    }
    if (m_ptr == nullptr || other.m_ptr == nullptr) {
      return false;
    }
    return *m_ptr == *other.m_ptr;
  }

  bool operator!=(const Any &other) const { return !(operator==(other)); }

  Any &operator=(const Any &other);

  bool operator<(const Any &other) const;

  std::string ToString() const {
    std::ostringstream buffer;
    if (m_tpIndex == typeid(float)) {
      buffer << "<float> " << cast<float>();
    } else if (m_tpIndex == typeid(double)) {
      buffer << "<double> " << cast<double>();
    } else if (m_tpIndex == typeid(int)) {
      buffer << "<int> " << cast<int>();
    } else if (m_tpIndex == typeid(bool)) {
      buffer << "<bool> " << cast<bool>();
    } else if (m_ptr != nullptr) {
      buffer << "<" << demangle(m_tpIndex.name()) << "> " << m_ptr->GetString();
    }
    return buffer.str();
  }
#ifdef _MSC_VER
  void dump() const { std::cout << ToString() << std::endl; }
#else
  __attribute__((used)) void dump() const { std::cout << ToString() << std::endl; }
#endif

 private:
  struct Base;
  using BasePtr = std::unique_ptr<Base>;

  // type base definition
  struct Base {
    virtual const std::type_info &type() const = 0;
    virtual BasePtr clone() const = 0;
    virtual ~Base() = default;
    virtual bool operator==(const Base &other) const = 0;
    virtual std::string GetString() = 0;
  };

  template <typename T>
  struct Derived : public Base {
    template <typename... Args>
    explicit Derived(Args &&... args) : m_value(std::forward<Args>(args)...), serialize_cache_("") {}

    bool operator==(const Base &other) const override {
      if (typeid(*this) != typeid(other)) {
        return false;
      }
      return m_value == static_cast<const Derived<T> &>(other).m_value;
    }

    const std::type_info &type() const override { return typeid(T); }

    BasePtr clone() const override { return std::make_unique<Derived<T>>(m_value); }

    ~Derived() override {}

    std::string GetString() override {
      std::stringstream buffer;
      buffer << m_value;
      return buffer.str();
    }

    T m_value;
    std::string serialize_cache_;
  };

  // clone method
  BasePtr clone() const {
    if (m_ptr != nullptr) {
      return m_ptr->clone();
    }
    return nullptr;
  }

  BasePtr m_ptr;              // point to real data
  std::type_index m_tpIndex;  // type info of data
};

using AnyPtr = std::shared_ptr<Any>;

struct AnyHash {
  std::size_t operator()(const Any &c) const { return c.Hash(); }
};

struct AnyLess {
  bool operator()(const Any &a, const Any &b) const { return a.Hash() < b.Hash(); }
};

bool AnyIsLiteral(const Any &any);
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_ANY_H_
