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

#ifndef MINDSPORE_CORE_IR_DTYPE_H_
#define MINDSPORE_CORE_IR_DTYPE_H_

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include "base/base.h"
#include "ir/named.h"

#include "ir/dtype/type.h"
#include "ir/dtype/number.h"
#include "ir/dtype/container.h"
#include "ir/dtype/empty.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/ref.h"
#include "ir/dtype/monad_type.h"
#include "utils/ms_utils.h"

/* namespace to support intermediate representation definition */
namespace mindspore {
/// \brief Get the shared_ptr of Type according to a TypeId.
///
/// \param[in] id Define a TypeId.
///
/// \return The shared_ptr of Type.
MS_CORE_API TypePtr TypeIdToType(TypeId id);

/// \brief Get the type string according to a TypeId.
///
/// \param[in] id Define a TypeId.
/// \param[in] to_lower Whether convert the string to lowercase.
///
/// \return The string of Type.
MS_CORE_API std::string TypeIdToString(TypeId id, bool to_lower = false);

/// \brief String defines a type of string.
class MS_CORE_API String final : public Object {
 public:
  /// \brief The constructor of String.
  ///
  /// \return The instance of String.
  String() : Object(kObjectTypeString, false) {}

  /// \brief The destructor of String.
  ~String() override = default;
  MS_DECLARE_PARENT(String, Object)

  TypeId generic_type_id() const override { return kObjectTypeString; }

  TypePtr DeepCopy() const override { return std::make_shared<String>(); }
  std::string ToString() const override { return std::string("String"); }
  std::string ToReprString() const override { return "string_"; }
  std::string DumpText() const override { return "String"; }
};
using StringPtr = std::shared_ptr<String>;

/// \brief Keyword defines a type of keyword.
class MS_CORE_API Keyword final : public Object {
 public:
  /// \brief The constructor of Keyword.
  ///
  /// \return The instance of Keyword.
  Keyword() : Object(kObjectTypeKeyword, false), key_(""), value_(nullptr) {}

  /// \brief The constructor of Keyword with some parameters.
  ///
  /// \param[in] key Define the key of Keyword.
  ///
  /// \param[in] value Define the value of Keyword.
  ///
  /// \return The instance of Keyword.
  Keyword(const std::string &key, const TypePtr &value) : Object(kObjectTypeKeyword, false), key_(key), value_(value) {}

  /// \brief The destructor of Keyword.
  ~Keyword() override = default;
  MS_DECLARE_PARENT(Keyword, Object)

  TypeId generic_type_id() const override { return kObjectTypeKeyword; }
  TypePtr DeepCopy() const override;

  std::string ToString() const override;
  std::string DumpText() const override;
  bool operator==(const Type &other) const override;

  /// \brief Get the key.
  ///
  /// \return The key.
  std::string GetKey() const { return key_; }

  /// \brief Get the value.
  ///
  /// \return The value.
  TypePtr GetValue() const { return value_; }

 private:
  std::string key_;
  TypePtr value_;
};
using KeywordPtr = std::shared_ptr<Keyword>;

/// \brief Slice defines a type of slice.
class MS_CORE_API Slice final : public Object {
 public:
  /// \brief The constructor of Slice.
  ///
  /// \return The instance of Slice.
  Slice() : Object(kObjectTypeSlice), start_(nullptr), stop_(nullptr), step_(nullptr) {}

  /// \brief The constructor of Slice with some parameters.
  ///
  /// \param[in] start Define the start type of Slice.
  ///
  /// \param[in] stop Define the stop type of Slice.
  ///
  /// \param[in] step Define the step type of Slice.
  ///
  /// \return The instance of Slice.
  Slice(const TypePtr &start, const TypePtr &stop, const TypePtr &step)
      : Object(kObjectTypeSlice, false), start_(start), stop_(stop), step_(step) {}

  /// \brief The destructor of Slice.
  ~Slice() override = default;
  MS_DECLARE_PARENT(Slice, Object)

  TypeId generic_type_id() const override { return kObjectTypeSlice; }
  TypePtr DeepCopy() const override;

  std::string ToString() const override;
  std::string DumpText() const override;
  bool operator==(const Type &other) const override;

  /// \brief Get the start type.
  ///
  /// \return The start type.
  TypePtr get_start() const { return start_; }

  /// \brief Get the stop type.
  ///
  /// \return The stop type.
  TypePtr get_stop() const { return stop_; }

  /// \brief Get the step type.
  ///
  /// \return The step type.
  TypePtr get_step() const { return step_; }

 private:
  TypePtr start_;
  TypePtr stop_;
  TypePtr step_;
};
using SlicePtr = std::shared_ptr<Slice>;

/// \brief Function defines a type of function.
class MS_CORE_API Function final : public Object {
 public:
  /// \brief The constructor of Function.
  ///
  /// \return The instance of Function.
  Function();

  /// \brief The constructor of Function with some parameters.
  ///
  /// \param[in] args Define the args type of the function.
  ///
  /// \param[in] retval Define the return value type of the function.
  ///
  /// \return The instance of Function.
  Function(const std::vector<TypePtr> &args, const TypePtr retval);

  /// \brief The destructor of Function.
  ~Function() override = default;
  MS_DECLARE_PARENT(Function, Object)

  TypeId generic_type_id() const override { return kObjectTypeFunction; }

  /// \brief Judge whether the function is transparent.
  ///
  /// \return The result of the judgment.
  bool IsTransparent() const { return (args_.empty()) && (retval_ == nullptr); }

  /// \brief Get the args type of the function.
  ///
  /// \return The args of the function.
  const std::vector<TypePtr> &args() const { return args_; }

  /// \brief Get the return value type of the function.
  ///
  /// \return The return value of the function.
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

/// \brief JTagged defines a type representing an object is tagged with J.
class MS_CORE_API JTagged final : public Object {
 public:
  /// \brief The constructor of JTagged.
  ///
  /// \return The instance of JTagged.
  JTagged() : Object(kObjectTypeJTagged) {}

  /// \brief The constructor of JTagged with a parameter.
  ///
  /// \param[in] subtype Define the sub type of JTagged.
  ///
  /// \return The instance of JTagged.
  explicit JTagged(const TypePtr &subtype) : Object(kObjectTypeJTagged, false), subtype_(subtype) {}

  /// \brief The destructor of JTagged.
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

/// \brief SymbolicKeyType defines a type of symbolic key.
class MS_CORE_API SymbolicKeyType final : public Object {
 public:
  /// \brief The constructor of SymbolicKeyType.
  ///
  /// \return The instance of SymbolicKeyType.
  SymbolicKeyType() : Object(kObjectTypeSymbolicKeyType) {}

  /// \brief The destructor of SymbolicKeyType.
  ~SymbolicKeyType() override = default;
  MS_DECLARE_PARENT(SymbolicKeyType, Object)

  TypeId generic_type_id() const override { return kObjectTypeSymbolicKeyType; }
  TypePtr DeepCopy() const override { return std::make_shared<SymbolicKeyType>(); }
  std::string ToReprString() const override { return "symbolic_key"; }
  std::string DumpText() const override { return "SymType"; }
};

/// \brief EnvType defines a type of environment variable.
class MS_CORE_API EnvType final : public Object {
 public:
  /// \brief The constructor of EnvType.
  ///
  /// \return The instance of EnvType.
  EnvType() : Object(kObjectTypeEnvType) {}

  /// \brief The destructor of EnvType.
  ~EnvType() override = default;
  MS_DECLARE_PARENT(EnvType, Object)

  TypePtr DeepCopy() const override { return std::make_shared<EnvType>(); }
  std::string ToReprString() const override { return "env_type"; }
  std::string DumpText() const override { return "EnvType"; }
};
using EnvTypePtr = std::shared_ptr<EnvType>;

/// \brief TypeType defines a type of type itself.
class MS_CORE_API TypeType final : public Type {
 public:
  /// \brief The constructor of TypeType.
  ///
  /// \return The instance of TypeType.
  TypeType() : Type(kMetaTypeTypeType) {}

  /// \brief The destructor of TypeType.
  ~TypeType() override = default;
  MS_DECLARE_PARENT(TypeType, Type)

  TypeId generic_type_id() const override { return kMetaTypeTypeType; }
  TypePtr DeepCopy() const override { return std::make_shared<TypeType>(); }
  std::string ToReprString() const override { return "type_type"; }
  std::string DumpText() const override { return "TypeType"; }
};
using TypeTypePtr = std::shared_ptr<TypeType>;

/// \brief Problem defines a type of problem.
class MS_CORE_API Problem final : public Type {
 public:
  /// \brief The constructor of Problem.
  ///
  /// \return The instance of Problem.
  Problem() : Type(kMetaTypeProblem), kind_(Named("unknown")) {}

  /// \brief The constructor of Problem with a parameter.
  ///
  /// \param[in] kind Define the kind of Problem.
  ///
  /// \return The instance of Problem.
  explicit Problem(const Named &kind) : Type(kMetaTypeProblem), kind_(kind) {}

  /// \brief The destructor of Problem.
  ~Problem() override = default;
  MS_DECLARE_PARENT(Problem, Type)

  TypeId generic_type_id() const override { return kMetaTypeProblem; }
  TypePtr DeepCopy() const override { return std::make_shared<Problem>(); }
  std::string ToString() const override { return kind_.name(); }
  std::string DumpText() const override { return "ProblemType"; }

  /// \brief The operator overloading for "<<".
  ///
  /// \param[in] os Define an output stream.
  ///
  /// \param[in] problem Define a shared_ptr of Problem.
  ///
  /// \return The output stream.
  friend std::ostream &operator<<(std::ostream &os, const std::shared_ptr<Problem> problem);

 private:
  Named kind_;
};
using ProblemPtr = std::shared_ptr<Problem>;

/// \brief MsClassType defines a type which is ms_class.
class MS_CORE_API MsClassType final : public Type {
 public:
  /// \brief The constructor of External.
  ///
  /// \return The instance of External.
  MsClassType() : Type(kObjectTypeClass) {}

  /// \brief The destructor of External.
  ~MsClassType() override = default;
  MS_DECLARE_PARENT(MsClassType, Type)

  TypeId generic_type_id() const override { return kObjectTypeClass; }
  TypePtr DeepCopy() const override { return std::make_shared<MsClassType>(); }
  std::string DumpText() const override { return "MsClassType"; }
};
using MsClassTypePtr = std::shared_ptr<MsClassType>;

/// \brief External defines a type which is external.
class MS_CORE_API External final : public Type {
 public:
  /// \brief The constructor of External.
  ///
  /// \return The instance of External.
  External() : Type(kMetaTypeExternal) {}

  /// \brief The destructor of External.
  ~External() override = default;
  MS_DECLARE_PARENT(External, Type)

  TypeId generic_type_id() const override { return kMetaTypeExternal; }
  TypePtr DeepCopy() const override { return std::make_shared<External>(); }
  std::string DumpText() const override { return "ExternalType"; }
};
using ExternalPtr = std::shared_ptr<External>;

// helper template
template <class T>
TypePtr Clone(const T &t) {
  return t.Clone();
}

/// \brief Get the shared_ptr of Type according to a string of type name.
///
/// \param[in] type_name Define a string of type name.
///
/// \return The shared_ptr of type.
MS_CORE_API TypePtr StringToType(const std::string &type_name);

/// \brief Get the TypeId of Type according to a string of type name.
///
/// \param[in] type_name Define a string of type name.
///
/// \return The TypeId of type.
MS_CORE_API TypeId StringToTypeId(const std::string &type_name);

/// \brief Given a type x and a base type, judge whether x is the base type or is a subclass of the base type.
///
/// \param[in] x Define the type to be judged.
///
/// \param[in] base_type Define the base type.
///
/// \return The result of the judgment.
MS_CORE_API bool IsIdentidityOrSubclass(TypePtr const &x, TypePtr const &base_type);

/// \brief Given a type t1 and another type t2, judge whether t1 is the subclass of the t2.
///
/// \param[in] t1 Define the type to be judged.
///
/// \param[in] t2 Define the base type.
///
/// \return The result of the judgment.
MS_CORE_API bool IsSubType(TypePtr const &t1, TypePtr const &t2 = nullptr);

GVAR_DEF(TypePtr, kTypeExternal, std::make_shared<External>());
GVAR_DEF(TypePtr, kTypeEnv, std::make_shared<EnvType>());
GVAR_DEF(TypePtr, kTypeType, std::make_shared<TypeType>());
GVAR_DEF(TypePtr, kString, std::make_shared<String>());
GVAR_DEF(TypePtr, kList, std::make_shared<List>());
GVAR_DEF(TypePtr, kTuple, std::make_shared<Tuple>());
GVAR_DEF(TypePtr, kDict, std::make_shared<Dictionary>());
GVAR_DEF(TypePtr, kSlice, std::make_shared<Slice>());
GVAR_DEF(TypePtr, kKeyword, std::make_shared<Keyword>());
GVAR_DEF(TypePtr, kTensorType, std::make_shared<TensorType>());
GVAR_DEF(TypePtr, kTensorTypeFP16, std::make_shared<TensorType>(std::make_shared<Float>(16)));
GVAR_DEF(TypePtr, kTensorTypeFP32, std::make_shared<TensorType>(std::make_shared<Float>(32)));
GVAR_DEF(TypePtr, kTensorTypeFP64, std::make_shared<TensorType>(std::make_shared<Float>(64)));
GVAR_DEF(TypePtr, kCSRTensorType, std::make_shared<CSRTensorType>());
GVAR_DEF(TypePtr, kCOOTensorType, std::make_shared<COOTensorType>());
GVAR_DEF(TypePtr, kRowTensorType, std::make_shared<RowTensorType>());
GVAR_DEF(TypePtr, kMapTensorType, std::make_shared<MapTensorType>());
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_DTYPE_H_
