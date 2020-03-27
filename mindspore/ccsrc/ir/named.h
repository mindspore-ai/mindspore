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

#ifndef MINDSPORE_CCSRC_IR_NAMED_H_
#define MINDSPORE_CCSRC_IR_NAMED_H_

#include <string>
#include <memory>
#include <functional>

#include "ir/base.h"
#include "ir/anf.h"

namespace mindspore {
class Named : public Value {
 public:
  explicit Named(const std::string& name) : name_(name) { hash_id_ = std::hash<std::string>{}(name); }
  Named(const Named& other) : Value(other) {
    this->name_ = other.name_;
    hash_id_ = std::hash<std::string>{}(other.name_);
  }
  ~Named() override = default;
  MS_DECLARE_PARENT(Named, Value);

  const std::string& name() const { return name_; }
  virtual bool operator==(const Named& other) const { return name_ == other.name(); }
  bool operator==(const Value& other) const override;
  Named& operator=(const Named& other) {
    if (&other != this) {
      this->type_ = other.type_;
      this->name_ = other.name_;
      hash_id_ = std::hash<std::string>{}(name_);
    }
    return *this;
  }

  std::size_t Hash() const { return hash_id_; }
  std::size_t hash() const override { return hash_id_; }

  friend std::ostream& operator<<(std::ostream& os, const Named& nmd) {
    os << nmd.name();
    return os;
  }

  std::string ToString() const override { return name(); }

 private:
  std::string name_;
  std::size_t hash_id_;
};

using NamedPtr = std::shared_ptr<Named>;

class None : public Named {
 public:
  None() : Named("None") {}
  ~None() override = default;
  MS_DECLARE_PARENT(None, Named);
  abstract::AbstractBasePtr ToAbstract() override;
};

extern const NamedPtr kNone;

class NullObj : public Named {
 public:
  NullObj() : Named("Null") {}
  ~NullObj() override = default;
  MS_DECLARE_PARENT(NullObj, Named);
  abstract::AbstractBasePtr ToAbstract() override;
};

extern const NamedPtr kNullObj;
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_IR_NAMED_H_
