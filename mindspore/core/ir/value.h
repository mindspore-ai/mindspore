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

#ifndef MINDSPORE_CORE_IR_VALUE_H_
#define MINDSPORE_CORE_IR_VALUE_H_

#include <type_traits>
#include <algorithm>
#include <vector>
#include <string>
#include <memory>
#include <sstream>
#include <utility>

#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ir/scalar.h"
#include "ir/dtype/ref.h"
#include "utils/hashing.h"
#include "utils/ms_utils.h"

namespace mindspore {
/// \brief ValueSequence defines a Value class whose type is Sequence.
class MS_CORE_API ValueSequence : public Value {
 public:
  /// \brief ValueSequence display defines a Value class whose type is Sequence, such as Tuple.
  ///
  /// \param[in] elements Define the vector of elements of value.
  explicit ValueSequence(const ValuePtrList &elements) : elements_(elements) {
    TypePtrList t_list;
    (void)std::transform(elements.begin(), elements.end(), std::back_inserter(t_list), [](const ValuePtr &ele) {
      MS_EXCEPTION_IF_NULL(ele);
      return ele->type();
    });
    TypePtr t = std::make_shared<Tuple>(t_list);
    type_ = t;
  }

  /// \brief ValueSequence defines a ValueNode whose type is Sequence.
  ///
  /// \param[in] elements Define the elements of value.
  ValueSequence(const std::initializer_list<ValuePtr> &elements) : elements_(elements.begin(), elements.end()) {
    TypePtrList t_list;
    (void)std::transform(elements_.begin(), elements_.end(), std::back_inserter(t_list), [](const ValuePtr &ele) {
      MS_EXCEPTION_IF_NULL(ele);
      return ele->type();
    });
    TypePtr t = std::make_shared<Tuple>(t_list);
    type_ = t;
  }
  /// \brief Destructor of ValueSequence.
  ~ValueSequence() override = default;
  MS_DECLARE_PARENT(ValueSequence, Value)
  /// \brief Get the Hash value of ValueSequence object.
  ///
  /// \return The hash value.
  std::size_t hash() const override { return hash_combine(tid(), std::hash<std::size_t>{}(elements_.size())); }
  /// \brief Get the element size of ValueSequence.
  ///
  /// \return The size of elements_.
  std::size_t size() const { return elements_.size(); }
  /// \brief Erase the elements in elements_ between 0 and idx.
  ///
  /// \param[in] idx Define the subscript of vector.
  /// \return Whether the erase operation is successful.
  bool erase(size_t idx);
  /// \brief Access elements by subscript.
  ///
  /// \param[in] dim Define the subscript of vector.
  /// \return The element whose subscript is dim.
  const ValuePtr operator[](const std::size_t &dim) const;
  /// \brief The value of ValueSequence object.
  ///
  /// \return The set of ValuePtr objects.
  const ValuePtrList &value() const { return elements_; }
  /// \brief Check whether the input is the current ValueSequence object.
  ///
  /// \param[in] other Define a Value object.
  /// \return Whether the input is the current ValueSequence object.
  bool operator==(const Value &other) const override;
  /// \brief Compares two ValueSequence objects.
  ///
  /// \param[in] other Define a ValueSequence object.
  /// \return Check whether the elements of the current object and the other object are the same.
  bool operator==(const ValueSequence &other) const;
  /// \brief Show each element ToString.
  ///
  /// \return The description of the ValueSequence object.
  std::string ToString() const override;
  /// \brief Show each element DumpText.
  ///
  /// \return The description of the ValueSequence object.
  std::string DumpText() const override;

 protected:
  ValuePtrList elements_;
};
using ValueSequencePtr = std::shared_ptr<ValueSequence>;

// For some old code in lite, still using typo names.
using ValueSequeue = ValueSequence;
using ValueSequeuePtr = ValueSequencePtr;

/// \brief ValueTuple defines a Value class whose type is Tuple.
class MS_CORE_API ValueTuple : public ValueSequence {
 public:
  /// \brief Constructor of ValueTuple.
  ///
  /// \param[in] elements Define the elements of the object.
  explicit ValueTuple(const std::vector<ValuePtr> &elements) : ValueSequence(elements) {}
  /// \brief Constructor of ValueTuple.
  ///
  /// \param[in] elements Define the elements of the object.
  ValueTuple(const std::initializer_list<ValuePtr> &elements) : ValueSequence(elements) {}
  /// \brief Destructor of ValueTuple.
  ~ValueTuple() override = default;
  MS_DECLARE_PARENT(ValueTuple, ValueSequence)
  /// \brief Get abstract of the ValueTuple object.
  ///
  /// \return The abstract of the ValueTuple object.
  abstract::AbstractBasePtr ToAbstract() override;
  /// \brief Show each element DumpText.
  ///
  /// \return The description of the ValueTuple object.
  std::string DumpText() const override { return "(" + ValueSequence::DumpText() + ")"; }
  /// \brief Show each element ToString.
  ///
  /// \return The description of the ValueTuple object.
  std::string ToString() const override { return "(" + ValueSequence::ToString() + ")"; }
};
using ValueTuplePtr = std::shared_ptr<ValueTuple>;

/// \brief ValueList defines a Value class whose type is List.
class MS_CORE_API ValueList : public ValueSequence {
 public:
  /// \brief Constructor of ValueList.
  ///
  /// \param[in] elements Define the elements of the object.
  explicit ValueList(const std::vector<ValuePtr> &elements) : ValueSequence(elements) {}
  /// \brief Constructor of ValueList.
  ///
  /// \param[in] elements Define the elements of the object.
  ValueList(const std::initializer_list<ValuePtr> &elements) : ValueSequence(elements) {}
  /// \brief Destructor of ValueList.
  ~ValueList() override = default;
  MS_DECLARE_PARENT(ValueList, ValueSequence)
  /// \brief Get abstract of the ValueList object.
  ///
  /// \return The abstract of the ValueList object.
  abstract::AbstractBasePtr ToAbstract() override;
  /// \brief Show each element DumpText.
  ///
  /// \return The description of the ValueList object.
  std::string DumpText() const override { return "[" + ValueSequence::DumpText() + "]"; }
  /// \brief Show each element ToString.
  ///
  /// \return The description of the ValueList object.
  std::string ToString() const override { return "[" + ValueSequence::ToString() + "]"; }
};
using ValueListPtr = std::shared_ptr<ValueList>;
inline ValuePtr MakeValue(const std::vector<ValuePtr> &v) { return std::make_shared<ValueTuple>(v); }
inline ValuePtr MakeValue(std::initializer_list<ValuePtr> v) { return std::make_shared<ValueTuple>(v); }

template <typename T>
struct is_vector : public std::false_type {};
template <typename T, typename A>
struct is_vector<std::vector<T, A>> : public std::true_type {};

/// \brief Convert other type to ValueTuple.
///
/// \param[in] vec Define the object to be converted.
/// \return ValuePtr a ValueTuple object.
template <typename T, typename U = typename std::enable_if<is_vector<T>::value, typename T::value_type>::type>
ValuePtr MakeValue(const T &vec) {
  std::vector<ValuePtr> list;
  (void)std::transform(vec.begin(), vec.end(), std::back_inserter(list), [](U ele) { return MakeValue(ele); });
  return std::make_shared<ValueTuple>(list);
}

/// \brief ValueSlice defines a Value class whose type is Slice.
class MS_CORE_API ValueSlice : public Value {
 public:
  /// \brief Constructor of ValueSlice.
  ///
  /// \param[in] start Define the start position of slice.
  /// \param[in] stop Define the stop position of slice.
  /// \param[in] step Define the space of slice.
  ValueSlice(const ValuePtr &start, const ValuePtr &stop, const ValuePtr &step)
      : start_(start), stop_(stop), step_(step) {}
  /// \brief Destructor of ValueSlice.
  ~ValueSlice() override = default;
  MS_DECLARE_PARENT(ValueSlice, Value)
  /// \brief Get the hash value of ValueSlice object.
  ///
  /// \return The hash value.
  std::size_t hash() const override;
  /// \brief Check whether the input is the current ValueSlice object.
  ///
  /// \param[in] other Define a Value object.
  /// \return Whether the input is the current ValueSlice object.
  bool operator==(const Value &other) const override;
  /// \brief Compares two ValueSlice objects.
  ///
  /// \param[in] other Define a ValueSlice object.
  /// \return Check whether the start, stop, step of the current object and the other object are the same.
  bool operator==(const ValueSlice &other) const;
  /// \brief Show the ValueSlice object ToString.
  ///
  /// \return The description of the ValueSlice object.
  std::string ToString() const override;
  /// \brief Get abstract of the ValueSlice object.
  ///
  /// \return The abstract of the ValueSlice object.
  abstract::AbstractBasePtr ToAbstract() override;
  /// \brief Show the ValueSlice object.
  ///
  /// \return The description of the ValueSlice object.
  std::string DumpText() const override { return ToString(); }
  /// \brief Get the start position of the slice object.
  ///
  /// \return The start position of the slice object.
  ValuePtr start() const { return start_; }
  /// \brief Get the stop position of the slice object.
  ///
  /// \return The stop position of the slice object.
  ValuePtr stop() const { return stop_; }
  /// \brief Get the step position of the slice object.
  ///
  /// \return The step position of the slice object.
  ValuePtr step() const { return step_; }

 private:
  ValuePtr start_;
  ValuePtr stop_;
  ValuePtr step_;
};
using ValueSlicePtr = std::shared_ptr<ValueSlice>;

/// \brief KeywordArg defines a Value class which has keyword.
class MS_CORE_API KeywordArg : public Value {
 public:
  /// \brief Constructor of KeywordArg.
  ///
  /// \param[in] key Define the key word.
  /// \param[in] value Define the value associated with the keyword.
  KeywordArg(const std::string &key, const ValuePtr &value) : key_(key), value_(value) {}
  /// \brief Destructor of KeywordArg.
  ~KeywordArg() override = default;
  MS_DECLARE_PARENT(KeywordArg, Value)
  /// \brief The hash value of the KeywordArg object.
  ///
  /// \return The hash value.
  std::size_t hash() const override;
  /// \brief Get value of the KeywordArg object.
  ///
  /// \return The value of the KeywordArg object.
  ValuePtr get_value() const { return value_; }
  /// \brief Check whether the input is the current KeywordArg object.
  ///
  /// \param[in] other Define a Value object.
  /// \return Whether the input is the current KeywordArg object.
  bool operator==(const Value &other) const override;
  /// \brief Compares two KeywordArg objects.
  ///
  /// \param[in] other Define a KeywordArg object.
  /// \return Check whether the key and value of the current object and the other object are the same.
  bool operator==(const KeywordArg &other) const;
  /// \brief Show the KeywordArg object.
  ///
  /// \return The description of the KeywordArg object.
  std::string ToString() const override;
  /// \brief Get abstract of the KeywordArg object.
  ///
  /// \return The abstract of the KeywordArg object.
  abstract::AbstractBasePtr ToAbstract() override;
  /// \brief Show the KeywordArg object DumpText.
  ///
  /// \return The description of the KeywordArg object.
  std::string DumpText() const override { return ToString(); }

 private:
  std::string key_;
  ValuePtr value_;
};
using KeywordArgPtr = std::shared_ptr<KeywordArg>;

/// \brief ValueDictionary defines a Value class whose type is Dictionary.
class MS_CORE_API ValueDictionary : public Value {
 public:
  /// \brief Constructor of ValueDictionary.
  ///
  /// \param[in] key_values Define the set of keys and values of Dictionary.
  explicit ValueDictionary(const std::vector<std::pair<ValuePtr, ValuePtr>> &key_values) : key_values_(key_values) {}
  /// \brief Destructor of ValueDictionary.
  ~ValueDictionary() override = default;
  MS_DECLARE_PARENT(ValueDictionary, Value)
  /// \brief Get the hash value through key_values_ size.
  ///
  /// \return The hash value.
  std::size_t hash() const override { return hash_combine(tid(), std::hash<std::size_t>{}(key_values_.size())); }
  /// \brief Get the size of key_values_.
  ///
  /// \return The size of key_values_.
  std::size_t size() const { return key_values_.size(); }
  /// \brief 'operator[]' which can access value by key.
  ///
  /// \param[in] key Define the keyword.
  /// \return The value associated with the keyword.
  const ValuePtr operator[](const ValuePtr &key) const;
  /// \brief The value of ValueDictionary object.
  ///
  /// \return The value of ValueDictionary object.
  const std::vector<std::pair<ValuePtr, ValuePtr>> &value() const { return key_values_; }
  /// \brief Check whether the input is the current ValueDictionary object.
  ///
  /// \param[in] other Define a Value object.
  /// \return Whether the input is the current ValueDictionary object.
  bool operator==(const Value &other) const override;
  /// \brief Compares two ValueDictionary objects.
  ///
  /// \param[in] other Define a ValueDictionary object.
  /// \return Check whether the keys and values of the current object and the other object are the same.
  bool operator==(const ValueDictionary &other) const;
  /// \brief Show the ValueDictionary object.
  ///
  /// \return The description of the ValueDictionary object.
  std::string ToString() const override {
    std::ostringstream buffer;
    buffer << "{";
    size_t index = 0;
    for (const auto &kv : key_values_) {
      // Only supports the key as string type currently.
      ValuePtr key = kv.first;
      ValuePtr value = kv.second;
      MS_EXCEPTION_IF_NULL(key);
      MS_EXCEPTION_IF_NULL(value);
      buffer << "'" << key->ToString() << "': " << value->ToString();
      if (index != key_values_.size() - 1) {
        buffer << ", ";
      }
      index++;
    }
    buffer << "}";
    return buffer.str();
  }
  /// \brief Get abstract of the ValueDictionary object.
  ///
  /// \return The abstract of the ValueDictionary object.
  abstract::AbstractBasePtr ToAbstract() override;
  /// \brief Show the ValueDictionary object DumpText.
  ///
  /// \return The description of the ValueDictionary object.
  std::string DumpText() const override { return ToString(); }

 private:
  std::vector<std::pair<ValuePtr, ValuePtr>> key_values_;
};
using ValueDictionaryPtr = std::shared_ptr<ValueDictionary>;

/// \brief StringImm defines a Value class whose type is String.
class MS_CORE_API StringImm : public Value {
 public:
  /// \brief Constructor of StringImm.
  ///
  /// \param[in] str Define the string.
  explicit StringImm(const std::string &str) noexcept
      : Value(kString), str_(str), hash_(std::hash<std::string>{}(str_)) {}
  /// \brief Destructor of StringImm.
  ~StringImm() override = default;
  MS_DECLARE_PARENT(StringImm, Value)
  /// \brief The hash value of the StringImm object.
  ///
  /// \return The hash value.
  std::size_t hash() const override { return hash_; }
  /// \brief Get the value of StringImm object.
  ///
  /// \return The value of StringImm object.
  virtual const std::string &value() const { return str_; }
  /// \brief Check whether the input is the current StringImm object.
  ///
  /// \param[in] other Define a Value object.
  /// \return Whether the input is the current StringImm object.
  bool operator==(const Value &other) const override;
  /// \brief Compares two StringImm objects.
  ///
  /// \param[in] other Define a StringImm object.
  /// \return Check whether the string of the current object and the other object are the same.
  bool operator==(const StringImm &other) const;
  /// \brief Get abstract of the StringImm object.
  ///
  /// \return The abstract of the StringImm object.
  abstract::AbstractBasePtr ToAbstract() override;
  /// \brief Show the StringImm object.
  ///
  /// \return The description of the StringImm object.
  std::string ToString() const override { return str_; }
  /// \brief Show the StringImm object DumpText.
  ///
  /// \return The description of the StringImm object.
  std::string DumpText() const override {
    std::ostringstream oss;
    oss << "\"" << str_ << "\"";
    return oss.str();
  }

 private:
  std::string str_;
  std::size_t hash_ = 0;
};
using StringImmPtr = std::shared_ptr<StringImm>;
IMM_TRAITS(StringImmPtr, std::string)
IMM_TRAITS(StringImmPtr, const char *)

/// \brief RefKey defines a class whose real type is String.
/// \brief Notice: RefKey is keep for compatible only, we use RefKey just as StringImm.
class MS_CORE_API RefKey final : public StringImm {
 public:
  /// \brief Constructor of RefKey.
  ///
  /// \param[in] str Define the string value of RefKey object.
  explicit RefKey(const std::string &str) : StringImm(str) {}
  /// \brief Destructor of RefKey.
  ~RefKey() override = default;
  MS_DECLARE_PARENT(RefKey, StringImm)

  abstract::AbstractBasePtr ToAbstract() override {
    MS_LOG(EXCEPTION) << "RefKey can't be converted to abstract, ref_key: " << ToString();
  }
};
using RefKeyPtr = std::shared_ptr<RefKey>;

/// \brief ValueAny defines a Value class which can be any Value type.
class MS_CORE_API ValueAny final : public Value {
 public:
  /// \brief Constructor of ValueAny.
  ValueAny() = default;
  /// \brief Destructor of ValueAny.
  ~ValueAny() override = default;
  MS_DECLARE_PARENT(ValueAny, Value)
  /// \brief The hash value of the ValueAny object.
  ///
  /// \return The hash value.
  std::size_t hash() const override { return tid(); }
  /// \brief Check whether the input is the current ValueAny object.
  ///
  /// \param[in] other Define a Value object.
  /// \return Whether the input is the current ValueAny object.
  bool operator==(const Value &other) const override;
  /// \brief Get abstract of the ValueAny object.
  ///
  /// \return The abstract of the ValueAny object.
  abstract::AbstractBasePtr ToAbstract() override;
};

GVAR_DEF(ValuePtr, kValueAny, std::make_shared<ValueAny>());

enum ValueProblemType : int { kDead = 0, kPoly = 1 };

/// \brief ValueProblem defines a class for DeadNode and PolyNode.
class MS_CORE_API ValueProblem final : public Value {
 public:
  /// \brief Constructor of ValueProblem.
  ///
  /// \param[in] err_type Define the error value type.
  explicit ValueProblem(ValueProblemType err_type) : err_type_(err_type) {}
  /// \brief Destructor of RefKey.
  ~ValueProblem() override = default;
  MS_DECLARE_PARENT(ValueProblem, Value)
  /// \brief The hash value of the ValueProblem object.
  ///
  /// \return The hash value.
  std::size_t hash() const override { return tid(); }
  /// \brief Check whether the input is the current ValueProblem object.
  ///
  /// \param[in] other Define a Value object.
  /// \return Whether the input is the current ValueProblem object.
  bool operator==(const Value &other) const override;
  /// \brief Check whether the input is the current ValueProblem object.
  ///
  /// \param[in] other Define a ValueProblem object.
  /// \return Whether the input is the current ValueProblem object.
  bool operator==(const ValueProblem &other) const;
  /// \brief Get abstract of the ValueProblem object.
  ///
  /// \return The abstract of the ValueProblem object.
  abstract::AbstractBasePtr ToAbstract() override {
    MS_LOG(EXCEPTION) << "ValueProblem(" << ToString() << ") can't be converted to abstract.";
  }
  /// \brief Check whether the value belongs to DeadNode.
  ///
  /// \return Whether the value belongs to DeadNode.
  bool IsDead() const { return err_type_ == kDead; }
  /// \brief Check whether the value belongs to PolyNode.
  ///
  /// \return Whether the value belongs to PolyNode.
  bool IsPoly() const { return err_type_ == kPoly; }
  /// \brief Show the ValueProblem object.
  ///
  /// \return The description of the ValueProblem object.
  std::string ToString() const override { return IsDead() ? "DeadNode" : "PolyNode"; }

 private:
  ValueProblemType err_type_{kDead};
};
using ValueProblemPtr = std::shared_ptr<ValueProblem>;

/// \brief Monad defines a Value class which is used in side effect.
class MS_CORE_API Monad : public Value {
 public:
  /// \brief Destructor of Monad.
  ~Monad() override = default;
  MS_DECLARE_PARENT(Monad, Value)
  /// \brief Get abstract of the Monad object.
  ///
  /// \return The abstract of the Monad object.
  abstract::AbstractBasePtr ToAbstract() override = 0;

 protected:
  /// \brief Constructor of Monad.
  ///
  /// \param[in] type Define the type of Monad object.
  explicit Monad(const TypePtr &type) : Value(type) {}
};

/// \brief UMonad defines a Value class which related to memory side effect.
class MS_CORE_API UMonad final : public Monad {
 public:
  /// \brief Constructor of UMonad.
  UMonad() : Monad(kUMonadType) {}
  /// \brief Destructor of UMonad.
  ~UMonad() override = default;
  MS_DECLARE_PARENT(UMonad, Monad)
  /// \brief The hash value of the UMonad object.
  ///
  /// \return The hash value.
  std::size_t hash() const override { return tid(); }
  /// \brief Check whether the input is UMonad object.
  ///
  /// \param[in] other Define a Value object.
  /// \return Whether the input is UMonad object.
  bool operator==(const Value &other) const override;
  /// \brief Get abstract of the UMonad object.
  ///
  /// \return The abstract of the UMonad object.
  abstract::AbstractBasePtr ToAbstract() override;
  /// \brief Show the UMonad object.
  ///
  /// \return The description of the UMonad object.
  std::string ToString() const override { return "U"; }
};
using UMonadPtr = std::shared_ptr<UMonad>;
MS_CORE_API extern const ValuePtr kUMonad;

/// \brief IOMonad defines a Value class which related to IO side effect.
class MS_CORE_API IOMonad final : public Monad {
 public:
  /// \brief Constructor of IOMonad.
  IOMonad() : Monad(kIOMonadType) {}
  /// \brief Destructor of IOMonad.
  ~IOMonad() override = default;
  MS_DECLARE_PARENT(IOMonad, Monad)
  /// \brief The hash value of the UMonad object.
  ///
  /// \return The hash value.
  std::size_t hash() const override { return tid(); }
  /// \brief Check whether the input is IOMonad object.
  ///
  /// \param[in] other Define a Value object.
  /// \return Whether the input is IOMonad object.
  bool operator==(const Value &other) const override;
  /// \brief Get abstract of the IOMonad object.
  ///
  /// \return The abstract of the IOMonad object.
  abstract::AbstractBasePtr ToAbstract() override;
  /// \brief Show the IOMonad object.
  ///
  /// \return The description of the IOMonad object.
  std::string ToString() const override { return "IO"; }
};
using IOMonadPtr = std::shared_ptr<IOMonad>;
MS_CORE_API extern const ValuePtr kIOMonad;

template <>
inline const char *GetValue(const ValuePtr &value) {
  if (value == nullptr) {
    MS_LOG(EXCEPTION) << "Value is nullptr";
  }
  auto imm = value->cast<StringImmPtr>();
  if (imm == nullptr) {
    MS_LOG(EXCEPTION) << "GetValue:" << value->ToString() << ", Type:" << value->type_name();
  }
  return common::SafeCStr(imm->value());
}

template <typename T, typename S = typename std::decay<T>::type,
          typename U = typename std::enable_if<is_vector<S>::value, typename S::value_type>::type>
std::vector<U> GetValue(const ValuePtr &value) {
  if (value == nullptr) {
    MS_LOG(EXCEPTION) << "Value is nullptr";
  }

  if (!value->isa<ValueSequence>()) {
    MS_LOG(EXCEPTION) << "Error GetValue for value: " << value->ToString() << ", type: vector<" << typeid(U).name()
                      << ">";
  }
  std::vector<U> rets;
  const std::vector<ValuePtr> &vals = value->cast<ValueSequencePtr>()->value();
  (void)std::transform(vals.begin(), vals.end(), std::back_inserter(rets),
                       [](const ValuePtr &v) { return GetValue<U>(v); });
  return rets;
}

inline ValueNodePtr NewValueNode(const ValuePtr &t) { return std::make_shared<ValueNode>(t); }

inline ValueNodePtr NewValueNode(const ValuePtr &t, NodeDebugInfoPtr &&debug_info) {
  return std::make_shared<ValueNode>(t, std::move(debug_info));
}

template <typename T, typename _ = typename std::enable_if<!std::is_base_of<Value, T>::value>::type>
inline ValueNodePtr NewValueNode(const std::shared_ptr<T> &x) {
  return NewValueNode(MakeValue(x));
}

template <typename T, typename _ = typename std::enable_if<!is_shared_ptr<T>::value>::type>
inline ValueNodePtr NewValueNode(const T &x) {
  return NewValueNode(MakeValue(x));
}
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_VALUE_H_
