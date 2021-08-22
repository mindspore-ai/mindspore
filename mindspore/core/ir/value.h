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
class ValueSequeue : public Value {
 public:
  explicit ValueSequeue(const ValuePtrList &elements) : elements_(elements) {
    TypePtrList t_list;
    (void)std::transform(elements.begin(), elements.end(), std::back_inserter(t_list), [](const ValuePtr &ele) {
      MS_EXCEPTION_IF_NULL(ele);
      return ele->type();
    });
    TypePtr t = std::make_shared<Tuple>(t_list);
    type_ = t;
  }
  ValueSequeue(const std::initializer_list<ValuePtr> &elements) : elements_(elements.begin(), elements.end()) {
    TypePtrList t_list;
    (void)std::transform(elements_.begin(), elements_.end(), std::back_inserter(t_list),
                         [](const ValuePtr &ele) { return ele->type(); });
    TypePtr t = std::make_shared<Tuple>(t_list);
    type_ = t;
  }
  ~ValueSequeue() override = default;
  MS_DECLARE_PARENT(ValueSequeue, Value)
  std::size_t hash() const override { return hash_combine(tid(), std::hash<std::size_t>{}(elements_.size())); }
  std::size_t size() const { return elements_.size(); }
  bool erase(size_t idx);
  const ValuePtr operator[](const std::size_t &dim) const;
  const ValuePtrList &value() const { return elements_; }
  bool operator==(const Value &other) const override;
  bool operator==(const ValueSequeue &other) const;
  std::string ToString() const override;
  std::string DumpText() const override;

 protected:
  ValuePtrList elements_;
};
using ValueSequeuePtr = std::shared_ptr<ValueSequeue>;

class ValueTuple : public ValueSequeue {
 public:
  explicit ValueTuple(const std::vector<ValuePtr> &elements) : ValueSequeue(elements) {}
  ValueTuple(const std::initializer_list<ValuePtr> &elements) : ValueSequeue(elements) {}
  ~ValueTuple() override = default;
  MS_DECLARE_PARENT(ValueTuple, ValueSequeue)
  abstract::AbstractBasePtr ToAbstract() override;

  std::string DumpText() const override { return "(" + ValueSequeue::DumpText() + ")"; }
  std::string ToString() const override { return "(" + ValueSequeue::ToString() + ")"; }
};
using ValueTuplePtr = std::shared_ptr<ValueTuple>;

class ValueList : public ValueSequeue {
 public:
  explicit ValueList(const std::vector<ValuePtr> &elements) : ValueSequeue(elements) {}
  ValueList(const std::initializer_list<ValuePtr> &elements) : ValueSequeue(elements) {}
  ~ValueList() override = default;
  MS_DECLARE_PARENT(ValueList, ValueSequeue)
  abstract::AbstractBasePtr ToAbstract() override;

  std::string DumpText() const override { return "[" + ValueSequeue::DumpText() + "]"; }
  std::string ToString() const override { return "[" + ValueSequeue::ToString() + "]"; }
};
using ValueListPtr = std::shared_ptr<ValueList>;

inline ValuePtr MakeValue(const std::vector<ValuePtr> &v) { return std::make_shared<ValueTuple>(v); }
inline ValuePtr MakeValue(std::initializer_list<ValuePtr> v) { return std::make_shared<ValueTuple>(v); }

template <typename T>
struct is_vector : public std::false_type {};
template <typename T, typename A>
struct is_vector<std::vector<T, A>> : public std::true_type {};

template <typename T, typename U = typename std::enable_if<is_vector<T>::value, typename T::value_type>::type>
ValuePtr MakeValue(const T &vec) {
  std::vector<ValuePtr> list;
  (void)std::transform(vec.begin(), vec.end(), std::back_inserter(list), [](U ele) { return MakeValue(ele); });
  return std::make_shared<ValueTuple>(list);
}

class ValueSlice : public Value {
 public:
  ValueSlice(const ValuePtr &start, const ValuePtr &stop, const ValuePtr &step)
      : start_(start), stop_(stop), step_(step) {}
  ~ValueSlice() override = default;
  MS_DECLARE_PARENT(ValueSlice, Value)
  std::size_t hash() const override;
  bool operator==(const Value &other) const override;
  bool operator==(const ValueSlice &other) const;

  std::string ToString() const override;

  abstract::AbstractBasePtr ToAbstract() override;
  std::string DumpText() const override { return ToString(); }
  ValuePtr start() const { return start_; }
  ValuePtr stop() const { return stop_; }
  ValuePtr step() const { return step_; }

 private:
  ValuePtr start_;
  ValuePtr stop_;
  ValuePtr step_;
};
using ValueSlicePtr = std::shared_ptr<ValueSlice>;

class KeywordArg : public Value {
 public:
  KeywordArg(const std::string &key, const ValuePtr &value) : key_(key), value_(value) {}
  ~KeywordArg() override = default;
  MS_DECLARE_PARENT(KeywordArg, Value)
  std::size_t hash() const override;
  ValuePtr get_value() const { return value_; }
  bool operator==(const Value &other) const override;
  bool operator==(const KeywordArg &other) const;

  std::string ToString() const override;

  abstract::AbstractBasePtr ToAbstract() override;
  std::string DumpText() const override { return ToString(); }

 private:
  std::string key_;
  ValuePtr value_;
};
using KeywordArgPtr = std::shared_ptr<KeywordArg>;

class ValueDictionary : public Value {
 public:
  explicit ValueDictionary(const std::vector<std::pair<std::string, ValuePtr>> &key_values) : key_values_(key_values) {}
  ~ValueDictionary() override = default;
  MS_DECLARE_PARENT(ValueDictionary, Value)
  std::size_t hash() const override { return hash_combine(tid(), std::hash<std::size_t>{}(key_values_.size())); }
  std::size_t size() const { return key_values_.size(); }
  const ValuePtr operator[](const std::string &key) const;
  const std::vector<std::pair<std::string, ValuePtr>> &value() const { return key_values_; }
  bool operator==(const Value &other) const override;
  bool operator==(const ValueDictionary &other) const;

  std::string ToString() const override {
    std::ostringstream buffer;
    std::vector<std::string> keys;
    std::vector<ValuePtr> values;
    for (const auto &kv : key_values_) {
      keys.push_back(kv.first);
      values.push_back(kv.second);
    }
    buffer << "(Dict: "
           << " keys:(";
    for (const auto &key : keys) {
      buffer << key << ", ";
    }
    buffer << ") values:(";
    for (const auto &value : values) {
      MS_EXCEPTION_IF_NULL(value);
      buffer << value->ToString() << ", ";
    }
    buffer << ")";
    return buffer.str();
  }
  abstract::AbstractBasePtr ToAbstract() override;
  std::string DumpText() const override { return ToString(); }

 private:
  std::vector<std::pair<std::string, ValuePtr>> key_values_;
};
using ValueDictionaryPtr = std::shared_ptr<ValueDictionary>;

class StringImm : public Value {
 public:
  explicit StringImm(const std::string &str) : Value(kString), str_(str), hash_(std::hash<std::string>{}(str_)) {}

  ~StringImm() override = default;
  MS_DECLARE_PARENT(StringImm, Value)
  std::size_t hash() const override { return hash_; }
  const std::string &value() const { return str_; }
  bool operator==(const Value &other) const override;
  bool operator==(const StringImm &other) const;
  abstract::AbstractBasePtr ToAbstract() override;
  std::string ToString() const override { return str_; }

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

class RefKey : public Named {
 public:
  explicit RefKey(const std::string &tag) : Named(tag) {}

  ~RefKey() override = default;
  MS_DECLARE_PARENT(RefKey, Named)
  const std::string &tag() const { return name(); }
  abstract::AbstractBasePtr ToAbstract() override;
  std::string ToString() const override { return "RefKey[" + name() + "]"; }

  std::string DumpText() const override {
    std::ostringstream oss;
    oss << "RefKey[\"" << name() << "\"]";
    return oss.str();
  }
};
using RefKeyPtr = std::shared_ptr<RefKey>;

class AnyValue : public Value {
 public:
  AnyValue() = default;
  ~AnyValue() override = default;
  MS_DECLARE_PARENT(AnyValue, Value)
  std::size_t hash() const override { return tid(); }
  bool operator==(const Value &other) const override;
  abstract::AbstractBasePtr ToAbstract() override;
};
extern const ValuePtr kAnyValue;

class Monad : public Value {
 public:
  ~Monad() override = default;
  MS_DECLARE_PARENT(Monad, Value)
  abstract::AbstractBasePtr ToAbstract() override = 0;

 protected:
  explicit Monad(TypePtr type) : Value(type) {}
};

class UMonad : public Monad {
 public:
  UMonad() : Monad(kUMonadType) {}
  ~UMonad() override = default;
  MS_DECLARE_PARENT(UMonad, Monad)
  std::size_t hash() const override { return tid(); }
  bool operator==(const Value &other) const override;
  abstract::AbstractBasePtr ToAbstract() override;
  std::string ToString() const override { return "U"; }
};
using UMonadPtr = std::shared_ptr<UMonad>;
extern const ValuePtr kUMonad;

class IOMonad : public Monad {
 public:
  IOMonad() : Monad(kIOMonadType) {}
  ~IOMonad() override = default;
  MS_DECLARE_PARENT(IOMonad, Monad)
  std::size_t hash() const override { return tid(); }
  bool operator==(const Value &other) const override;
  abstract::AbstractBasePtr ToAbstract() override;
  std::string ToString() const override { return "IO"; }
};
using IOMonadPtr = std::shared_ptr<IOMonad>;
extern const ValuePtr kIOMonad;

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

  if (!value->isa<ValueSequeue>()) {
    MS_LOG(EXCEPTION) << "Error GetValue for value: " << value->ToString() << ", type: vector<" << typeid(U).name()
                      << ">";
  }
  std::vector<U> rets;
  const std::vector<ValuePtr> &vals = value->cast<ValueSequeuePtr>()->value();
  (void)std::transform(vals.begin(), vals.end(), std::back_inserter(rets),
                       [](const ValuePtr &v) { return GetValue<U>(v); });
  return rets;
}

inline ValueNodePtr NewValueNode(const ValuePtr &t) { return std::make_shared<ValueNode>(t); }

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
