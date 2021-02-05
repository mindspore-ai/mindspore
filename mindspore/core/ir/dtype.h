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

#ifndef MINDSPORE_CORE_IR_DTYPE_H_
#define MINDSPORE_CORE_IR_DTYPE_H_

#include <cstddef>
#include <iostream>
#include <initializer_list>
#include <memory>
#include <utility>
#include <sstream>
#include <string>
#include <vector>
#include <type_traits>
#include <unordered_map>
#include <algorithm>
#include "base/base.h"
#include "ir/named.h"

#include "ir/dtype/type.h"
#include "ir/dtype/number.h"
#include "ir/dtype/container.h"
#include "ir/dtype/empty.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/ref.h"
#include "ir/dtype/monad_type.h"

/* namespace to support intermediate representation definition */
namespace mindspore {
// Only few type supported now.
TypePtr TypeIdToType(TypeId id);

class String : public Object {
 public:
  String() : Object(kObjectTypeString, false) {}
  ~String() override = default;
  MS_DECLARE_PARENT(String, Object)

  TypeId generic_type_id() const override { return kObjectTypeString; }

  TypePtr DeepCopy() const override { return std::make_shared<String>(); }
  std::string ToString() const override { return std::string("String"); }
  std::string ToReprString() const override { return "string"; }
  std::string DumpText() const override { return "String"; }
};
using StringPtr = std::shared_ptr<String>;

class Keyword : public Object {
 public:
  Keyword() : Object(kObjectTypeKeyword, false), key_(""), value_(nullptr) {}
  Keyword(const std::string &key, const TypePtr &value) : Object(kObjectTypeKeyword, false), key_(key), value_(value) {}

  ~Keyword() override = default;
  MS_DECLARE_PARENT(Keyword, Object)

  TypeId generic_type_id() const override { return kObjectTypeKeyword; }
  TypePtr DeepCopy() const override;

  std::string ToString() const override;
  std::string DumpText() const override;
  bool operator==(const Type &other) const override;

  std::string GetKey() const { return key_; }
  TypePtr GetValue() const { return value_; }

 private:
  std::string key_;
  TypePtr value_;
};
using KeywordPtr = std::shared_ptr<Keyword>;

class Slice : public Object {
 public:
  Slice() : Object(kObjectTypeSlice), start_(nullptr), stop_(nullptr), step_(nullptr) {}
  Slice(const TypePtr &start, const TypePtr &stop, const TypePtr &step)
      : Object(kObjectTypeSlice, false), start_(start), stop_(stop), step_(step) {}

  ~Slice() override = default;
  MS_DECLARE_PARENT(Slice, Object)

  TypeId generic_type_id() const override { return kObjectTypeSlice; }
  TypePtr DeepCopy() const override;

  std::string ToString() const override;
  std::string DumpText() const override;
  bool operator==(const Type &other) const override;

  TypePtr get_start() const { return start_; }
  TypePtr get_stop() const { return stop_; }
  TypePtr get_step() const { return step_; }

 private:
  TypePtr start_;
  TypePtr stop_;
  TypePtr step_;
};
using SlicePtr = std::shared_ptr<Slice>;

class Function : public Object {
 public:
  Function();
  Function(const std::vector<TypePtr> &args, const TypePtr retval);
  ~Function() override = default;
  MS_DECLARE_PARENT(Function, Object)

  TypeId generic_type_id() const override { return kObjectTypeFunction; }

  // Add temporarily for return abstraction to avoid type checking.
  bool IsTransparent() const { return (args_.empty()) && (retval_ == nullptr); }
  const std::vector<TypePtr> &args() const { return args_; }
  const TypePtr &retval() const { return retval_; }

  TypePtr DeepCopy() const override;
  bool operator==(const Type &other) const override;
  std::string ToString() const override;
  std::string ToReprString() const override { return "function"; }

 private:
  std::vector<TypePtr> args_;
  TypePtr retval_;
};
using FunctionPtr = std::shared_ptr<Function>;

class JTagged : public Object {
 public:
  JTagged() : Object(kObjectTypeJTagged) {}
  explicit JTagged(const TypePtr &subtype) : Object(kObjectTypeJTagged, false), subtype_(subtype) {}
  ~JTagged() override = default;
  MS_DECLARE_PARENT(JTagged, Object)

  TypeId generic_type_id() const override { return kObjectTypeJTagged; }

  TypePtr DeepCopy() const override;
  std::string ToString() const override;
  std::string DumpText() const override;

 private:
  TypePtr subtype_;
};
using JTaggedPtr = std::shared_ptr<JTagged>;

class SymbolicKeyType : public Object {
 public:
  SymbolicKeyType() : Object(kObjectTypeSymbolicKeyType) {}
  ~SymbolicKeyType() override = default;
  MS_DECLARE_PARENT(SymbolicKeyType, Object)

  TypeId generic_type_id() const override { return kObjectTypeSymbolicKeyType; }
  TypePtr DeepCopy() const override { return std::make_shared<SymbolicKeyType>(); }
  std::string ToReprString() const override { return "symbolic_key"; }
  std::string DumpText() const override { return "SymType"; }
};

class EnvType : public Object {
 public:
  EnvType() : Object(kObjectTypeEnvType) {}
  ~EnvType() override = default;
  MS_DECLARE_PARENT(EnvType, Object)

  TypePtr DeepCopy() const override { return std::make_shared<EnvType>(); }
  std::string ToReprString() const override { return "env_type"; }
  std::string DumpText() const override { return "EnvType"; }
};
using EnvTypePtr = std::shared_ptr<EnvType>;

class TypeType : public Type {
 public:
  TypeType() : Type(kMetaTypeTypeType) {}
  ~TypeType() override = default;
  MS_DECLARE_PARENT(TypeType, Type)

  TypeId generic_type_id() const override { return kMetaTypeTypeType; }
  TypePtr DeepCopy() const override { return std::make_shared<TypeType>(); }
  std::string ToReprString() const override { return "type_type"; }
  std::string DumpText() const override { return "TypeType"; }
};
using TypeTypePtr = std::shared_ptr<TypeType>;

class Problem : public Type {
 public:
  Problem() : Type(kMetaTypeProblem), kind_(Named("unknown")) {}
  explicit Problem(const Named &kind) : Type(kMetaTypeProblem), kind_(kind) {}
  ~Problem() override = default;
  MS_DECLARE_PARENT(Problem, Type)

  TypeId generic_type_id() const override { return kMetaTypeProblem; }
  TypePtr DeepCopy() const override { return std::make_shared<Problem>(); }
  std::string ToString() const override { return kind_.name(); }
  std::string DumpText() const override { return "ProblemType"; }

  friend std::ostream &operator<<(std::ostream &os, const std::shared_ptr<Problem> problem);

 private:
  Named kind_;
};
using ProblemPtr = std::shared_ptr<Problem>;

class External : public Type {
 public:
  External() : Type(kMetaTypeExternal) {}
  ~External() override = default;
  MS_DECLARE_PARENT(External, Type)

  TypeId generic_type_id() const override { return kMetaTypeExternal; }
  TypePtr DeepCopy() const override { return std::make_shared<External>(); }
  std::string DumpText() const override { return "ExternalType"; }

 private:
  TypePtr kind;
};
using ExternalPtr = std::shared_ptr<External>;

// helper template
template <class T>
TypePtr Clone(const T &t) {
  return t.Clone();
}

TypePtr StringToType(const std::string &type_name);

// Judge whether x is predicate or is a subclass of predicate.
bool IsIdentidityOrSubclass(TypePtr const &x, TypePtr const &base_type);

bool IsParentOrChildrenType(TypePtr const &x, TypePtr const &base_type);

// Whether t1 is identity or a subclass of t2.
bool IsSubType(TypePtr const &t1, TypePtr const &t2 = nullptr);

struct TypeHasher {
  std::size_t operator()(TypePtr const &type) const;
};
struct TypeListHasher {
  std::size_t operator()(const TypePtrList &type_list) const;
};
struct TypeEqual {
  bool operator()(TypePtr const &t1, TypePtr const &t2) const;
};
struct TypeListEqual {
  bool operator()(TypePtrList const &lhs, TypePtrList const &rhs) const;
};

extern const TypePtr kTypeExternal;
extern const TypePtr kTypeEnv;
extern const TypePtr kTypeType;
extern const TypePtr kString;
extern const TypePtr kList;
extern const TypePtr kTuple;
extern const TypePtr kDict;
extern const TypePtr kSlice;
extern const TypePtr kKeyword;
extern const TypePtr kTensorType;
extern const TypePtr kTensorTypeFP16;
extern const TypePtr kTensorTypeFP32;
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_DTYPE_H_
