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

#include "ir/primitive.h"

#include <utility>
#include "abstract/abstract_function.h"
#include "utils/ms_utils.h"

namespace mindspore {
Primitive::Primitive(const std::string &name, const bool is_base, const PrimType prim_type)
    : Named(name),
      is_base_(is_base),
      has_signature_(false),
      prim_type_(prim_type),
      record_evaluate_add_attr_(false),
      is_const_prim_(false) {}

Primitive::Primitive(const std::string &name, const mindspore::HashMap<std::string, ValuePtr> &attrs)
    : Named(name),
      attrs_(attrs),
      is_base_(true),
      has_signature_(false),
      prim_type_(kPrimTypeBuiltIn),
      record_evaluate_add_attr_(false),
      is_const_prim_(false) {}

Primitive::Primitive(const Primitive &prim)
    : Named(prim),
      attrs_(prim.attrs_),
      evaluate_added_attrs_(prim.evaluate_added_attrs_),
      instance_name_(prim.instance_name_),
      is_base_(prim.is_base_),
      has_signature_(prim.has_signature_),
      prim_type_(prim.prim_type_),
      record_evaluate_add_attr_(false),
      is_const_prim_(false),
      const_input_indexes_(prim.const_input_indexes_) {}

Primitive &Primitive::operator=(const Primitive &other) {
  if (this == &other) {
    return *this;
  }
  Named::operator=(other);
  attrs_ = other.attrs_;
  evaluate_added_attrs_ = other.evaluate_added_attrs_;
  instance_name_ = other.instance_name_;
  is_base_ = other.is_base_;
  has_signature_ = other.has_signature_;
  prim_type_ = other.prim_type_;
  record_evaluate_add_attr_ = false;
  is_const_prim_ = false;
  const_input_indexes_ = other.const_input_indexes_;
  return *this;
}

abstract::AbstractBasePtr Primitive::ToAbstract() {
  return std::make_shared<abstract::PrimitiveAbstractClosure>(shared_from_base<Primitive>(), nullptr);
}

bool Primitive::operator==(const Value &other) const {
  if (other.isa<Primitive>()) {
    auto other_prim = static_cast<const Primitive &>(other);
    return *this == other_prim;
  } else {
    return false;
  }
}

bool Primitive::operator==(const Primitive &other) const {
  if (name() != other.name()) {
    return false;
  }
  return common::IsAttrsEqual(attrs_, other.attrs_);
}

std::string Primitive::GetAttrsText() const {
  if (attrs_.empty()) {
    return "";
  }

  std::ostringstream oss;
  oss << "[";
  bool is_first = true;
  for (auto &attr : attrs_) {
    if (is_first) {
      is_first = false;
    } else {
      oss << ", ";
    }
    const std::string value = attr.second == nullptr ? "" : attr.second->DumpText();
    oss << attr.first << "=" << value;
  }
  oss << "]";

  return oss.str();
}
}  // namespace mindspore
