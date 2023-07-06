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

#ifndef MINDSPORE_CORE_BASE_BASE_H_
#define MINDSPORE_CORE_BASE_BASE_H_

#include <atomic>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <typeinfo>
#include <vector>
#include <utility>
#include <algorithm>
#include "base/user_data.h"
#include "mindapi/base/macros.h"
#include "utils/hashing.h"
#include "utils/log_adapter.h"
#include "utils/ordered_map.h"
#include "utils/ordered_set.h"

namespace mindspore {
template <typename T>
struct is_shared_ptr : public std::false_type {};
template <typename T>
struct is_shared_ptr<std::shared_ptr<T>> : public std::true_type {};

/// \brief Base is a base class of many derived classes, which provides basic interfaces such as hash, and so on.
class MS_CORE_API Base : public std::enable_shared_from_this<Base> {
 public:
  /// \brief The type id of this class.
  static constexpr uint32_t kTypeId = ConstStringHash("Base");

  /// \brief The constructor of Base.
  ///
  /// \return The instance of Base.
  constexpr Base() = default;

  /// \brief The copy constructor of Base.
  ///
  /// \param[in] other Define another instance of Base.
  ///
  /// \return The instance of Base.
  Base(const Base &other);

  /// \brief The operator overloading for "==".
  ///
  /// \param[in] rhs Define the right operand of "==".
  ///
  /// \return The comparison result.
  virtual bool operator==(const Base &rhs);

  /// \brief The copy assignment operator of Base.
  ///
  /// \param[in] Define another instance of Base.
  ///
  /// \return The instance of Base.
  Base &operator=(const Base &other);

  /// \brief The destructor of Base.
  virtual ~Base() = default;

  /// \brief Get the hash value of this object.
  ///
  /// \return The hash value.
  virtual std::size_t hash() const;

  /// \brief Get the string representation of this object.
  ///
  /// \return The string representation.
  virtual std::string ToString() const;

  /// \brief Export the string representation to the standard output stream.
  virtual void dump() const;

  /// \brief Get the text representation of this object.
  ///
  /// \return The text representation.
  virtual std::string DumpText() const;

  /// \brief Judge whether this object is an instance of class with the given type id.
  ///
  /// \param[in] tid Define a type id.
  ///
  /// \return The result of the judgment.
  virtual bool IsFromTypeId(uint32_t tid) const;

  /// \brief Judge whether the type id of this object is same as the given type id.
  ///
  /// \param[in] tid Define a type id.
  ///
  /// \return The result of the judgment.
  virtual bool IsSameTypeId(uint32_t tid) const;

  /// \brief Get the type name of this object.
  ///
  /// \return The type name.
  virtual std::string type_name() const;

  /// \brief Get the type id of this object.
  ///
  /// \return The type id.
  virtual uint32_t tid() const;

  /// \brief Judge whether this class is derived from class with the given type id.
  ///
  /// \param[in] tid Define a type id.
  ///
  /// \return The result of the judgment.

  static bool IsDerivedFrom(uint32_t tid);

  /// \brief Judge whether this object is an instance of a given class which is derived from Base.
  ///
  /// \return The result of the judgment.
  template <typename T,
            typename std::enable_if<!is_shared_ptr<T>::value && std::is_base_of<Base, T>::value, T>::type * = nullptr>
  inline bool isa() const {
    if constexpr (std::is_final<T>::value) {
      return this->IsSameTypeId(T::kTypeId);
    } else {
      return this->IsFromTypeId(T::kTypeId);
    }
  }

  /// \brief Cast a shared_ptr of this object to a given class.
  ///
  /// \return If success, a shared_ptr of the given class will be returned. Otherwise a nullptr will be returned.
  template <typename T, typename U = typename std::enable_if<is_shared_ptr<T>::value, typename T::element_type>::type>
  inline T cast() {
    if (isa<U>()) {
      return std::static_pointer_cast<U>(shared_from_this());
    }
    return nullptr;
  }

  /// \brief Cast to a raw pointer of the given class.
  ///
  /// \return If success, a raw pointer of the given class will be returned. Otherwise a nullptr will be returned.
  template <typename T, typename U = typename std::enable_if<std::is_base_of<Base, T>::value>::type>
  inline T *cast_ptr() {
    if (isa<T>()) {
      return static_cast<T *>(this);
    }
    return nullptr;
  }

  /// \brief Set user data.
  ///
  /// \param[in] key The key of user data.
  /// \param[in] value The value of user data.
  template <typename T>
  void set_user_data(const std::string &key, const std::shared_ptr<T> &value) {
    user_data_.set<T>(key, value);
  }

  /// \brief Set user data.
  ///
  /// \param[in] value The value of user data.
  template <typename T>
  void set_user_data(const std::shared_ptr<T> &value) {
    user_data_.set<T>(T::key, value);
  }

  /// \brief Get user data.
  ///
  /// \param[in] key The key of user data.
  /// \return Pointer to user data.
  template <typename T>
  std::shared_ptr<T> user_data(const std::string &key) const {
    return user_data_.get<T>(key);
  }

  /// \brief Set user data.
  ///
  /// \return Pointer to user data.
  template <typename T>
  std::shared_ptr<T> user_data() const {
    return user_data_.get<T>(T::key);
  }

  /// \brief Check whether there is corresponding user data by the given key.
  ///
  /// \param[in] key The key of user data.
  /// \return True if it exists, otherwise false.
  bool has_user_data(const std::string &key) const;

  /// \brief Check if there is user data.
  ///
  /// \return True if it exists, otherwise false.
  template <typename T>
  bool has_user_data() const {
    return user_data_.has(T::key);
  }

  /// \brief Clone user data.
  ///
  /// \param[in] other used to copy user data.
  void CloneUserData(const std::shared_ptr<Base> &other);

  /// \brief Clone user data.
  ///
  /// \param[in] other used to copy.
  void CloneUserData(const UserData &other);

  /// \brief Get user data.
  ///
  /// \return User data.
  const UserData &GetUserData() const;

 protected:
  /// \brief Get the shared_ptr of Base.
  ///
  /// \return The shared_ptr of Base.
  template <typename Derived>
  std::shared_ptr<Derived> shared_from_base() {
    return std::static_pointer_cast<Derived>(shared_from_this());
  }

  UserData user_data_;
};

using BasePtr = std::shared_ptr<Base>;
using BaseWeakPtr = std::weak_ptr<Base>;

template <typename T, typename U>
T *cast(U *source) {
  if (source != nullptr && source->template isa<T>()) {
    return static_cast<T *>(source);
  }
  return nullptr;
}

template <
  typename T, typename U,
  typename std::enable_if<std::is_base_of<Base, T>::value && std::is_base_of<Base, U>::value, T>::type * = nullptr>
std::shared_ptr<T> dyn_cast(const std::shared_ptr<U> &r) {
  if (r != nullptr && r->template isa<T>()) {
    return std::static_pointer_cast<T>(r);
  }
  return std::shared_ptr<T>();
}

template <
  typename T, typename U,
  typename std::enable_if<std::is_base_of<Base, T>::value && std::is_base_of<Base, U>::value, T>::type * = nullptr>
T *dyn_cast_ptr(const std::shared_ptr<U> &r) {
  if (r != nullptr && r->template isa<T>()) {
    return static_cast<T *>(r.get());
  }
  return nullptr;
}

#define MS_DECLARE_PARENT(current_t, parent_t)                                             \
  static constexpr uint32_t kTypeId = ConstStringHash(#parent_t "_" #current_t);           \
  static bool IsDerivedFrom(uint32_t tid) ALWAYS_INLINE {                                  \
    return (tid == current_t::kTypeId) || parent_t::IsDerivedFrom(tid);                    \
  }                                                                                        \
  uint32_t tid() const override { return current_t::kTypeId; }                             \
  bool IsFromTypeId(uint32_t tid) const override { return current_t::IsDerivedFrom(tid); } \
  bool IsSameTypeId(uint32_t tid) const override { return tid == current_t::kTypeId; }     \
  std::string type_name() const override { return #current_t; }

class Type;
using TypePtr = std::shared_ptr<Type>;

class AnfNode;
using AnfNodePtr = std::shared_ptr<AnfNode>;
using AnfNodePtrList = std::vector<AnfNodePtr>;
using AnfNodeSet = OrderedSet<AnfNodePtr>;
using AnfNodeWeakPtr = std::weak_ptr<AnfNode>;
using AnfNodeWeakPtrList = std::vector<AnfNodeWeakPtr>;

class FuncGraph;
using FuncGraphPtr = std::shared_ptr<FuncGraph>;
using FuncGraphWeakPtr = std::weak_ptr<FuncGraph>;

namespace abstract {
class AbstractBase;
using AbstractBasePtr = std::shared_ptr<AbstractBase>;
using AbstractElementPair = std::pair<AbstractBasePtr, AbstractBasePtr>;
class AnalysisContext;
using AnalysisContextPtr = std::shared_ptr<AnalysisContext>;
}  // namespace abstract
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_BASE_H_
