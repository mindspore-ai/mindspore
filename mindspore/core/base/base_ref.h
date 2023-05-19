/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CORE_UTILS_BASE_REF_H_
#define MINDSPORE_CORE_UTILS_BASE_REF_H_

#include <type_traits>
#include <algorithm>
#include <vector>
#include <set>
#include <string>
#include <memory>
#include <sstream>
#include <utility>
#include <iterator>

#include "ir/value.h"

namespace mindspore {
class BaseRef;
class VectorRef;
class SetRef;
class RunFunctionRef;

using iterator = std::vector<BaseRef>::iterator;
using const_iterator = std::vector<BaseRef>::const_iterator;
using const_reverse_iterator = std::vector<BaseRef>::const_reverse_iterator;

using RunFunc = std::function<VectorRef(const VectorRef &args)>;
using RunFuncPtr = std::shared_ptr<RunFunc>;

template <typename T>
using remove_reference_t = typename std::remove_reference<T>::type;
template <typename T>
using remove_const_t = typename std::remove_const<T>::type;
template <typename T>
using is_base = std::is_base_of<Base, remove_reference_t<T>>;
template <typename T>
using is_value = std::is_base_of<Value, remove_reference_t<T>>;
template <typename T>
using is_base_ref = std::is_base_of<BaseRef, remove_reference_t<T>>;

MS_CORE_API iterator ConstIteratorCast(std::vector<BaseRef> *v, const const_iterator iter);

inline std::shared_ptr<VectorRef> MakeNode(const std::vector<BaseRef> &elements) {
  return std::make_shared<VectorRef>(elements);
}

inline std::shared_ptr<VectorRef> MakeNode(std::initializer_list<BaseRef> elements) {
  return std::make_shared<VectorRef>(elements);
}

// Anfnode, Funcgraph and some not value node class
template <typename T,
          typename std::enable_if<is_shared_ptr<remove_const_t<T>>::value && is_base<typename T::element_type>::value,
                                  int64_t>::type = static_cast<int64_t>(0)>
inline BasePtr MakeNode(const T &v) {
  return v;
}

template <typename T, typename std::enable_if<!is_shared_ptr<remove_const_t<T>>::value && !is_base_ref<T>::value,
                                              int64_t>::type = static_cast<int64_t>(0)>
inline BasePtr MakeNode(const T &v) {
  return MakeValue(v);
}

inline std::shared_ptr<VectorRef> MakeNode(const VectorRef &a) { return std::make_shared<VectorRef>(a); }
inline std::shared_ptr<VectorRef> MakeNode(const AnfNodePtrList &a) {
  std::vector<BaseRef> ret;
  (void)std::transform(a.begin(), a.end(), std::back_inserter(ret), [](const AnfNodePtr &v) { return v; });
  return std::make_shared<VectorRef>(ret);
}
inline std::shared_ptr<SetRef> MakeNode(const SetRef &a) { return std::make_shared<SetRef>(a); }
inline std::shared_ptr<RunFunctionRef> MakeNode(const RunFuncPtr &a) { return std::make_shared<RunFunctionRef>(a); }

/// \brief BaseRef is a base class which store a Base pointer to some real data.
class MS_CORE_API BaseRef : public Base {
 public:
  /// \brief The Constructor of BaseRef.
  ///
  /// \return The instance of BaseRef.
  BaseRef() : m_ptr(nullptr) {}

  /// \brief The copy constructor of BaseRef.
  ///
  /// \param[in] other Define another instance of BaseRef.
  ///
  /// \return The instance of BaseRef.
  BaseRef(const BaseRef &other);

  /// \brief Get the Base pointer to some real data.
  ///
  /// \return The Base pointer.
  virtual std::shared_ptr<Base> copy() const { return m_ptr; }

  /// \brief The move constructor of BaseRef.
  ///
  /// \param[in] other Define another instance of BaseRef.
  ///
  /// \return The instance of BaseRef.
  BaseRef(BaseRef &&other) : Base(other) {
    m_ptr = other.m_ptr;
    other.m_ptr = nullptr;
  }

  /// \brief The move constructor of BaseRef with template.
  ///
  /// \param[in] t Define an instance of T.
  ///
  /// \return The instance of BaseRef.
  template <class T,
            class = typename std::enable_if<!std::is_same<typename std::decay<T>::type, BaseRef>::value, T>::type>
  BaseRef(T &&t) {  // NOLINT
    m_ptr = MakeNode(t);
  }

  /// \brief The destructor of BaseRef.
  ~BaseRef() override { m_ptr = nullptr; }

  MS_DECLARE_PARENT(BaseRef, Base)

  /// \brief The operator overloading for "!=".
  ///
  /// \param[in] other Define the right operand of "!=".
  ///
  /// \return The comparison result.
  bool operator!=(const BaseRef &other) const { return !(operator==(other)); }

  /// \brief The operator overloading for "==".
  ///
  /// \param[in] other Define the right operand of "==".
  ///
  /// \return The comparison result.
  virtual bool operator==(const BaseRef &other) const;

  /// \brief The copy assignment operator of BaseRef.
  ///
  /// \param[in] other Define another instance of BaseRef.
  ///
  /// \return The instance of BaseRef.
  BaseRef &operator=(const BaseRef &other);

  /// \brief The move assignment operator of BaseRef.
  ///
  /// \param[in] other Define another instance of BaseRef.
  ///
  /// \return The instance of BaseRef.
  virtual BaseRef &operator=(BaseRef &&other);

  std::size_t hash() const override {
    if (m_ptr == nullptr) {
      MS_LOG(ERROR) << "Invalid m_ptr";
      return 0;
    }
    return m_ptr->hash();
  }

  std::string ToString() const override;

  /// \brief Judge whether the real data is null.
  ///
  /// \return The result of the judgment.
  bool is_null() const { return m_ptr == nullptr; }

  /// \brief Get the type id of the real data.
  ///
  /// \return The type id of the real data.
  virtual uint32_t type() const;

  BasePtr m_ptr; /**< pointer to the real data */
};
using BaseRefPtr = std::shared_ptr<BaseRef>;

struct BaseRefHash {
  std::size_t operator()(const BaseRef &c) const { return c.hash(); }
};

struct BaseRefLess {
  bool operator()(const BaseRef &a, const BaseRef &b) const { return a.hash() < b.hash(); }
};

namespace utils {
// judge isa relation
// examples: isa<Int32Imm>(handle), isa<FuncGraph>(handle)
template <typename T,
          typename std::enable_if<is_base<T>::value && !is_base_ref<T>::value, int64_t>::type = static_cast<int64_t>(0)>
bool isa(const BaseRef &handle) {
  if (!handle.m_ptr) {
    return false;
  }
  return handle.m_ptr->isa<T>();
}

// noderef isa ptr isa<AnfNodePtr>(x) or isa<SeqPtr>()
template <typename T, typename U = typename std::enable_if<is_shared_ptr<T>::value, typename T::element_type>::type,
          typename std::enable_if<is_base<U>::value || is_base_ref<U>::value, int64_t>::type = static_cast<int64_t>(0)>
bool isa(const BaseRef &handle) {
  if (handle.m_ptr == nullptr) {
    return typeid(handle.m_ptr) == typeid(T);
  }

  if (handle.m_ptr->isa<U>()) {
    return true;
  }

  // constptr isa<anfnodeptr> can be true
  return std::dynamic_pointer_cast<U>(handle.m_ptr) != nullptr;
}

// isa<int32>(handle)
template <typename S, typename U = typename ImmTraits<S>::type::element_type>
bool isa(const BaseRef &handle) {
  if (handle.m_ptr == nullptr) {
    return false;
  }
  return handle.m_ptr->isa<U>();
}

// isa<BaseRef>(handle), judge reference or ptr
template <typename T, typename std::enable_if<is_base_ref<T>::value, int64_t>::type = static_cast<int64_t>(0)>
bool isa(const BaseRef &handle) {
  return handle.isa<T>() || (handle.m_ptr && handle.m_ptr->isa<T>());
}

// valueref -> C++ type
// cast<int64_t>(handle)
template <typename T, typename std::enable_if<!is_base_ref<T>::value && !is_shared_ptr<T>::value, int64_t>::type =
                        static_cast<int64_t>(0)>
T cast(const BaseRef &handle) {
  T ret = GetValue<T>(std::static_pointer_cast<Value>(handle.m_ptr));
  return std::move(ret);
}

// valueref -> valueref type
// cast<VectorRef>(handle)
template <typename T, typename std::enable_if<is_base_ref<T>::value, int64_t>::type = static_cast<int64_t>(0)>
const T &cast(const BaseRef &handle) {
  if (handle.m_ptr) {
    return static_cast<const T &>(*handle.m_ptr);
  }

  return static_cast<const T &>(handle);
}

// valueref -> nodeptr type
// cast<FuncGraphPtr>(handle)
template <typename T, typename U = typename std::enable_if<is_shared_ptr<T>::value, typename T::element_type>::type,
          typename std::enable_if<is_shared_ptr<T>::value && std::is_base_of<Base, typename T::element_type>::value,
                                  int64_t>::type = static_cast<int64_t>(0)>
T cast(const BaseRef &handle) {
  if (!handle.m_ptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Can not cast to " << typeid(T).name() << ", pointer is null";
  }

  auto m = handle.m_ptr->cast<T>();
  if (nullptr != m) {
    return m;
  }
  return std::static_pointer_cast<U>(handle.m_ptr);
}
}  // namespace utils

class MS_CORE_API VectorRef : public BaseRef {
 public:
  using value_type = BaseRef;

  VectorRef() {}
  explicit VectorRef(const std::vector<BaseRef> &elements) : elements_(elements) {}
  VectorRef(const const_iterator &begin, const const_iterator &end) : elements_(begin, end) {}

  // left reference
  VectorRef(const VectorRef &other);
  VectorRef &operator=(const VectorRef &other);

  ~VectorRef() override = default;

  std::shared_ptr<Base> copy() const override { return std::make_shared<VectorRef>(elements_); }

  bool empty() const { return (elements_.size() == 0); }

  std::size_t size() const { return elements_.size(); }
  MS_DECLARE_PARENT(VectorRef, BaseRef)

  const BaseRef &operator[](const std::size_t &dim) const {
    if (dim >= size()) {
      MS_LOG(INTERNAL_EXCEPTION) << "Out of the size of the tuple.";
    }
    return elements_[dim];
  }

  BaseRef &operator[](const std::size_t &dim) {
    if (dim >= size()) {
      MS_LOG(INTERNAL_EXCEPTION) << "Out of the size of the tuple.";
    }
    return elements_[dim];
  }

  uint32_t type() const override { return tid(); }
  std::string ToString() const override;
  std::vector<BaseRef> &elements() { return elements_; }
  void clear() { elements_.clear(); }

  bool operator==(const BaseRef &other) const override;
  bool operator==(const VectorRef &other) const;

  void push_back(const BaseRef &value) { elements_.push_back(value); }
  void push_back(BaseRef &&value) { elements_.push_back(value); }

  void emplace_back(const BaseRef &value) { elements_.emplace_back(value); }
  void emplace_back(BaseRef &&value) { elements_.emplace_back(value); }

  template <class InputIt>
  void insert(const iterator pos, const InputIt first, const InputIt last) {
    (void)elements_.insert(pos, first, last);
  }

  template <class InputIt>
  void insert(const const_iterator cpos, const InputIt first, const InputIt last) {
    auto pos = ConstIteratorCast(&elements_, cpos);
    (void)elements_.insert(pos, first, last);
  }

  const_iterator begin() const { return elements_.begin(); }
  const_iterator end() const { return elements_.end(); }

  const_reverse_iterator rbegin() const { return elements_.rbegin(); }
  const_reverse_iterator rend() const { return elements_.rend(); }

  iterator erase(const const_iterator cpos) {
    auto pos = ConstIteratorCast(&elements_, cpos);
    return elements_.erase(pos);
  }

  iterator erase(const const_iterator cfirst, const const_iterator clast) {
    auto first = ConstIteratorCast(&elements_, cfirst);
    auto last = ConstIteratorCast(&elements_, clast);
    return elements_.erase(first, last);
  }

  std::size_t hash() const override {
    std::stringstream buffer;
    buffer << ToString();
    return std::hash<std::string>()(buffer.str());
  }

  std::vector<BaseRef> elements_;
};

using VectorRefPtr = std::shared_ptr<VectorRef>;

using set_iterator = std::set<BaseRef, BaseRefLess>::iterator;
using const_set_iterator = std::set<BaseRef, BaseRefLess>::const_iterator;

struct VectorRefHash {
  std::size_t operator()(const VectorRef &c) const { return c.hash(); }
};

class MS_CORE_API SetRef : public BaseRef {
 public:
  SetRef() {}
  explicit SetRef(const std::set<BaseRef, BaseRefLess> &elements) : elements_(elements) {}
  SetRef(const std::initializer_list<BaseRef> elements) : elements_(elements.begin(), elements.end()) {}
  SetRef(const const_set_iterator &begin, const const_set_iterator &end) : elements_(begin, end) {}

  // left reference
  SetRef(const SetRef &other);
  SetRef &operator=(const SetRef &other);

  bool operator==(const BaseRef &other) const override;
  bool operator==(const SetRef &other) const;

  ~SetRef() override = default;

  std::shared_ptr<Base> copy() const override { return std::make_shared<SetRef>(elements_); }

  bool empty() const { return (elements_.size() == 0); }

  std::size_t size() const { return elements_.size(); }
  MS_DECLARE_PARENT(SetRef, BaseRef)

  uint32_t type() const override { return tid(); }
  std::string ToString() const override;
  std::set<BaseRef, BaseRefLess> &elements() { return elements_; }
  void clear() { elements_.clear(); }

  void insert(const BaseRef &elem) { (void)elements_.insert(elem); }

  const_set_iterator begin() const { return elements_.begin(); }
  const_set_iterator end() const { return elements_.end(); }

  template <class InputIt>
  void insert(const InputIt first, const InputIt last) {
    (void)elements_.insert(first, last);
  }

  std::size_t count(const BaseRef &elem) const { return elements_.count(elem); }
  const_set_iterator find(const BaseRef &elem) const { return elements_.find(elem); }

  std::set<BaseRef, BaseRefLess> elements_;
};

using SetRefPtr = std::shared_ptr<SetRef>;

class MS_CORE_API RunFunctionRef : public BaseRef {
 public:
  RunFunctionRef() {}
  explicit RunFunctionRef(const RunFuncPtr &ref_func) : func_(ref_func) {}

  ~RunFunctionRef() override = default;
  MS_DECLARE_PARENT(RunFunctionRef, BaseRef)

  uint32_t type() const override { return tid(); }
  std::string ToString() const override { return std::string("RunFunctionRef"); }
  bool operator==(const BaseRef &other) const override;
  bool operator==(const RunFunctionRef &other) const;

  RunFuncPtr func_;
};
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_BASE_REF_H_
