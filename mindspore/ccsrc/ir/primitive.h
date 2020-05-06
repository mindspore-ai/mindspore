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

#ifndef MINDSPORE_CCSRC_IR_PRIMITIVE_H_
#define MINDSPORE_CCSRC_IR_PRIMITIVE_H_

#include <unordered_map>
#include <vector>
#include <memory>
#include <string>
#include <tuple>
#include "pybind11/pybind11.h"

#include "pipeline/static_analysis/abstract_value.h"
#include "utils/misc.h"
#include "utils/log_adapter.h"
#include "ir/signature.h"

#include "parallel/ops_info/operator_info.h"

namespace py = pybind11;

namespace mindspore {
using abstract::AbstractBasePtr;
using abstract::AbstractBasePtrList;
// Supported meta type
enum PrimType {
  kPrimTypeUnknown = 0,
  kPrimTypeBegin = kTypeUnknown,
  kPrimTypeBuiltIn,        // Built-in primitive operator
  kPrimTypePyInferShape,   // Primitive operator defined by custom
  kPrimTypePyInferTensor,  // Primitive operator defined by custom
  kPrimTypeUserCustom
};

class Primitive : public Named {
 public:
  explicit Primitive(const std::string &name, const PrimType prim_type = kPrimTypeBuiltIn)
      : Named(name), signatures_(), prim_type_(prim_type) {}

  Primitive(const Primitive &prim)
      : Named(prim),
        attrs_(prim.attrs_),
        signatures_(prim.signatures_),
        instance_name_(prim.instance_name_),
        prim_type_(prim.prim_type_) {}

  MS_DECLARE_PARENT(Primitive, Named);

  abstract::AbstractBasePtr ToPrimAbstract(const AnfNodePtr &anf_node);
  std::string ToString() const override { return name(); }
  virtual py::function GetBpropFunction();
  virtual py::function GetComputeFunction();
  Primitive &AddAttr(const std::string &name, const ValuePtr &attr) {
    attrs_[name] = attr;
    return *this;
  }

  Primitive &SetAttrs(const std::unordered_map<std::string, ValuePtr> &attrs) {
    for (auto &attr : attrs) {
      attrs_[attr.first] = attr.second;
    }
    return *this;
  }

  void set_signatures(
    std::vector<std::tuple<std::string, SignatureEnumRW, SignatureEnumKind, py::object, SignatureEnumDType>>
      signatures);

  const std::vector<Signature> &signatures() const { return signatures_; }

  void set_attr(const std::string &attrName, const ValuePtr &attr) { attrs_[attrName] = attr; }
  void EraseAttr(const std::string &attrName) { (void)attrs_.erase(attrName); }

  ValuePtr GetAttr(const std::string &attrName) const {
    auto iter = attrs_.find(attrName);
    return iter == attrs_.cend() ? nullptr : iter->second;
  }

  const std::unordered_map<std::string, ValuePtr> &attrs() const { return attrs_; }

  // if Primitive has any attribute, for Primitives like scalar_add, return, etc, don't have any attribute.
  bool HasAttr() const { return !attrs_.empty(); }
  bool HasAttr(const std::string &attrName) const {
    auto iter = attrs_.find(attrName);
    return !(iter == attrs_.cend());
  }
  void set_prim_type(const PrimType t) { prim_type_ = t; }
  void set_instance_name(const std::string s) { instance_name_ = s; }
  bool HasPyEvaluator() const { return prim_type_ == kPrimTypePyInferShape || prim_type_ == kPrimTypeUserCustom; }
  bool HasPyInferTensor() const { return prim_type_ == kPrimTypePyInferTensor; }
  bool IsCustomPrim() const { return prim_type_ == kPrimTypeUserCustom; }

  PrimType prim_type() const { return prim_type_; }
  std::string instance_name() const { return instance_name_; }
  std::string GetAttrsText() const;
  bool operator==(const Value &other) const override;
  bool operator==(const Primitive &other) const;
  ~Primitive() override = default;

 protected:
  std::unordered_map<std::string, ValuePtr> attrs_;

 private:
  std::vector<Signature> signatures_;
  std::string instance_name_;
  PrimType prim_type_;
};

class PrimitivePy : public Primitive {
 public:
  PrimitivePy(const py::str &name, const py::object &python_obj) : Primitive(name), python_obj_(python_obj) {}
  ~PrimitivePy() override = default;
  MS_DECLARE_PARENT(PrimitivePy, Primitive);
  py::function GetBpropFunction() override;
  py::function GetComputeFunction() override;

  void AddPyAttr(const py::str &name, const py::object &obj);

  py::dict GetAttrDict();

  const bool parse_info_ = true;
  const py::object &GetPyObj() const { return python_obj_; }
  bool is_tuple_input_ = false;

 private:
  py::object python_obj_;
};

using PrimitivePyPtr = std::shared_ptr<PrimitivePy>;

inline std::ostream &operator<<(std::ostream &os, const PrimitivePtr &p) {
  os << *p;
  return os;
}

struct PrimitiveEqual {
  bool operator()(PrimitivePtr const &t1, PrimitivePtr const &t2) const {
    MS_EXCEPTION_IF_NULL(t1);
    MS_EXCEPTION_IF_NULL(t2);
    return t1->name() == t2->name();
  }
};

struct PrimitiveHasher {
  std::size_t operator()(PrimitivePtr const &prim) const { return prim->Hash(); }
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_IR_PRIMITIVE_H_
