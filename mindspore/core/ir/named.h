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

#ifndef MINDSPORE_CORE_IR_NAMED_H_
#define MINDSPORE_CORE_IR_NAMED_H_

#include <string>
#include <memory>
#include <functional>

#include "ir/anf.h"

namespace mindspore {
/// \brief Named defines an abstract class that records the name and hash_id.
class MS_CORE_API Named : public Value {
 public:
  /// \brief The constructor for Named.
  ///
  /// \param[in] name The name of object.
  explicit Named(const std::string &name) : name_(name), hash_id_(std::hash<std::string>{}(name)) {}
  /// \brief The constructor for Named, create a Named for another Named.
  ///
  /// \param[in]  other The input tensor.
  Named(const Named &other) : Value(other) {
    this->name_ = other.name_;
    hash_id_ = std::hash<std::string>{}(other.name_);
  }
  /// \brief The destructor of None.
  ~Named() override = default;
  MS_DECLARE_PARENT(Named, Value);
  /// \brief Getting name of object.
  ///
  /// \return The name of object.
  const std::string &name() const { return name_; }
  /// \brief Setting name of object.
  ///
  /// \param[in] name The name set for the object.
  /// \no return.
  void set_name(const std::string &name) { name_ = name; }
  /// \brief Check whether two Named objects are the same.
  ///
  /// \param[in] other The other Named to be compared with.
  /// \return Return true if the same,otherwise return false.
  virtual bool operator==(const Named &other) const { return name_ == other.name(); }
  bool operator==(const Value &other) const override;
  /// \brief Overloads operator '=' for Named.
  ///
  /// \param[in] other An existing Named object.
  /// \return A Named object set with the same type, name and hash_id as other.
  virtual Named &operator=(const Named &other) {
    if (&other != this) {
      this->type_ = other.type_;
      this->name_ = other.name_;
      hash_id_ = std::hash<std::string>{}(name_);
    }
    return *this;
  }
  /// \brief Get hash id for named.
  ///
  /// \return The restored hash id of Named.
  std::size_t Hash() const { return std::hash<std::string>{}(name_); }
  std::size_t hash() const override { return std::hash<std::string>{}(name_); }
  /// \brief Overloads operator << for Named.
  ///
  /// \param os The output stream.
  /// \param nmd Named to be displayed.
  /// \return Output stream that contains the name of Named object.
  friend std::ostream &operator<<(std::ostream &os, const Named &nmd) {
    os << nmd.name();
    return os;
  }
  /// \brief Get name for Named.
  ///
  /// \return The restored name of Named.
  std::string ToString() const override { return name(); }

 private:
  std::string name_;
  std::size_t hash_id_;
};
using NamedPtr = std::shared_ptr<Named>;
/// \brief Implementation of hash operation.
struct MS_CORE_API NamedHasher {
  /// \brief Implementation of hash operation.
  ///
  /// \param name The Name object need to be hashed.
  /// \return The hash result.
  std::size_t operator()(NamedPtr const &name) const {
    std::size_t hash = name->Hash();
    return hash;
  }
};
/// \brief Equal operator for Name.
struct MS_CORE_API NamedEqual {
  /// \brief Implementation of Equal operation.
  ///
  /// \param t1 The left Named to compare.
  /// \param t2 The right Named to compare.
  /// \return The comparison result, Return true if t1 and t2 is the same,else return false.
  bool operator()(NamedPtr const &t1, NamedPtr const &t2) const {
    MS_EXCEPTION_IF_NULL(t1);
    MS_EXCEPTION_IF_NULL(t2);
    return *t1 == *t2;
  }
};
/// \brief None defines interface for none data.
class MS_CORE_API None final : public Named {
 public:
  /// \brief The default constructor for None.
  None() : Named("None") {}
  /// \brief The destructor of None.
  ~None() override = default;
  MS_DECLARE_PARENT(None, Named);
  abstract::AbstractBasePtr ToAbstract() override;
};
GVAR_DEF(NamedPtr, kNone, std::make_shared<None>());

/// \brief Null defines interface for null data.
class MS_CORE_API Null final : public Named {
 public:
  /// \brief The default constructor for Null.
  Null() : Named("Null") {}
  /// \brief The destructor of Null.
  ~Null() override = default;
  MS_DECLARE_PARENT(Null, Named);
  abstract::AbstractBasePtr ToAbstract() override;
};
GVAR_DEF(NamedPtr, kNull, std::make_shared<Null>());

/// \brief Ellipsis defines interface for ... data.
class MS_CORE_API Ellipsis final : public Named {
 public:
  /// \brief The default constructor for Ellipsis.
  Ellipsis() : Named("Ellipsis") {}
  /// \brief The destructor of Ellipsis.
  ~Ellipsis() override = default;
  MS_DECLARE_PARENT(Ellipsis, Named);
  abstract::AbstractBasePtr ToAbstract() override;
};
GVAR_DEF(NamedPtr, kEllipsis, std::make_shared<Ellipsis>());

class MS_CORE_API MindIRClassType final : public Named {
 public:
  explicit MindIRClassType(const std::string &class_type) : Named(class_type) {}
  ~MindIRClassType() override = default;
  MS_DECLARE_PARENT(MindIRClassType, Named);
  abstract::AbstractBasePtr ToAbstract() override;
};
using MindIRClassTypePtr = std::shared_ptr<MindIRClassType>;

class MindIRMetaFuncGraph final : public Named {
 public:
  explicit MindIRMetaFuncGraph(const std::string &name) : Named(name) {}
  ~MindIRMetaFuncGraph() override = default;
  MS_DECLARE_PARENT(MindIRMetaFuncGraph, Named);
};
using MindIRMetaFuncGraphPtr = std::shared_ptr<MindIRMetaFuncGraph>;

class MS_CORE_API MindIRNameSpace final : public Named {
 public:
  explicit MindIRNameSpace(const std::string &name_space) : Named(name_space), name_space_(name_space) {}
  ~MindIRNameSpace() override = default;
  MS_DECLARE_PARENT(MindIRNameSpace, Named);
  const std::string &name_space() const { return name_space_; }
  abstract::AbstractBasePtr ToAbstract() override;

 private:
  std::string name_space_;
};
using MindIRNameSpacePtr = std::shared_ptr<MindIRNameSpace>;

class MS_CORE_API MindIRSymbol final : public Named {
 public:
  explicit MindIRSymbol(const std::string &symbol) : Named(symbol), symbol_(symbol) {}
  ~MindIRSymbol() override = default;
  MS_DECLARE_PARENT(MindIRSymbol, Named);
  const std::string &symbol() const { return symbol_; }
  abstract::AbstractBasePtr ToAbstract() override;

 private:
  std::string symbol_;
};
using MindIRSymbolPtr = std::shared_ptr<MindIRSymbol>;
}  // namespace mindspore
#endif  // MINDSPORE_CORE_IR_NAMED_H_
