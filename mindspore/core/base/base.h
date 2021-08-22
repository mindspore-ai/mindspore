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

#ifndef MINDSPORE_CORE_BASE_BASE_H_
#define MINDSPORE_CORE_BASE_BASE_H_

#include <atomic>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>
#include <utility>
#include "utils/visible.h"
#include "utils/log_adapter.h"
#include "utils/ordered_set.h"
#include "utils/ordered_map.h"

namespace mindspore {
template <typename T>
struct is_shared_ptr : public std::false_type {};
template <typename T>
struct is_shared_ptr<std::shared_ptr<T>> : public std::true_type {};

class Base : public std::enable_shared_from_this<Base> {
 public:
  constexpr Base() = default;
  Base(const Base &other) : std::enable_shared_from_this<Base>(other) {}
  virtual bool operator==(const Base &rhs) {
    if (this == &rhs) {
      return true;
    }
    return false;
  }

  virtual Base &operator=(const Base &) { return *this; }
  virtual ~Base() = default;
  virtual std::size_t hash() const { return tid(); }
  virtual std::string ToString() const { return type_name(); }
  virtual void dump() const { std::cout << ToString() << std::endl; }

  virtual std::string DumpText() const { return ToString(); }

  virtual const bool IsFromTypeId(uint32_t tid) const;
  virtual std::string type_name() const { return "Base"; }
  static uint32_t GetTypeId(const char *const type_key);
  virtual uint32_t tid() const {
    static const uint32_t tid = GetTypeId(typeid(Base).name());
    return tid;
  }

  template <typename T,
            typename std::enable_if<!is_shared_ptr<T>::value && std::is_base_of<Base, T>::value, T>::type * = nullptr>
  inline bool isa() const {
    static const uint32_t tid = GetTypeId(typeid(T).name());
    return this->IsFromTypeId(tid);
  }

  template <typename T, typename U = typename std::enable_if<is_shared_ptr<T>::value, typename T::element_type>::type>
  inline T cast() {
    if (isa<U>()) {
      return std::static_pointer_cast<U>(shared_from_this());
    } else {
      return nullptr;
    }
  }

 protected:
  template <typename Derived>
  std::shared_ptr<Derived> shared_from_base() {
    return std::static_pointer_cast<Derived>(shared_from_this());
  }
};

using BasePtr = std::shared_ptr<Base>;
using BaseWeakPtr = std::weak_ptr<Base>;

template <typename T, typename U>
inline T *cast(U *source) {
  if (source != nullptr && source->template isa<T>()) {
    return static_cast<T *>(source);
  } else {
    return nullptr;
  }
}

template <
  typename T, typename U,
  typename std::enable_if<std::is_base_of<Base, T>::value && std::is_base_of<Base, U>::value, T>::type * = nullptr>
inline std::shared_ptr<T> dyn_cast(const std::shared_ptr<U> &r) {
  if (r != nullptr && r->template isa<T>()) {
    return std::static_pointer_cast<T>(r);
  } else {
    return std::shared_ptr<T>();
  }
}

#define MS_DECLARE_PARENT(current_t, parent_t)                             \
  uint32_t tid() const override {                                          \
    static const uint32_t tid = GetTypeId(typeid(current_t).name());       \
    return tid;                                                            \
  }                                                                        \
  const bool IsFromTypeId(uint32_t from_tid) const override {              \
    static const uint32_t tid = Base::GetTypeId(typeid(current_t).name()); \
    if (tid == from_tid) {                                                 \
      return true;                                                         \
    }                                                                      \
    return parent_t::IsFromTypeId(from_tid);                               \
  }                                                                        \
  std::string type_name() const override { return #current_t; }

class Type;
using TypePtr = std::shared_ptr<Type>;

class AnfNode;
using AnfNodePtr = std::shared_ptr<AnfNode>;
using AnfNodePtrList = std::vector<AnfNodePtr>;
using AnfNodeSet = OrderedSet<AnfNodePtr>;
using AnfNodeWeakPtr = std::weak_ptr<AnfNode>;

class FuncGraph;
using FuncGraphPtr = std::shared_ptr<FuncGraph>;
using FuncGraphWeakPtr = std::weak_ptr<FuncGraph>;

namespace abstract {
class AbstractBase;
using AbstractBasePtr = std::shared_ptr<AbstractBase>;
using AbstractAttribute = std::pair<std::string, AbstractBasePtr>;
class AnalysisContext;
using AnalysisContextPtr = std::shared_ptr<AnalysisContext>;
}  // namespace abstract

struct MS_EXPORT TypeIdManager {
  std::mutex mutex;
  std::atomic<uint32_t> type_counter{0};
  std::unordered_map<std::string, uint32_t> map;
  static TypeIdManager *Get();
  TypeIdManager() : mutex(), type_counter(0), map() {}
};
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_BASE_H_
