/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_IR_CELL_H_
#define MINDSPORE_CCSRC_IR_CELL_H_

#include <unordered_map>
#include <vector>
#include <memory>
#include <string>
#include <tuple>

#include "abstract/abstract_value.h"
#include "utils/misc.h"

namespace mindspore {
using abstract::AbstractBasePtr;
using abstract::AbstractBasePtrList;
// value for Cell

class MS_CORE_API Cell : public Named {
 public:
  explicit Cell(const std::string &name) : Named(name) {}
  MS_DECLARE_PARENT(Cell, Named);
  abstract::AbstractBasePtr ToAbstract() override;
  std::string ToString() const override;
  std::string GetAttrString() const;

  const std::unordered_map<std::string, ValuePtr> &attrs() const { return attrs_; }
  void set_attrs(const std::unordered_map<std::string, ValuePtr> &attrs_input) { attrs_ = attrs_input; }

  void AddAttr(const std::string &name, const ValuePtr &attr) { attrs_[name] = attr; }
  void DelAttr(const std::string &name);
  ValuePtr GetAttr(const std::string &attr_name) const {
    auto iter = attrs_.find(attr_name);
    return iter == attrs_.cend() ? nullptr : iter->second;
  }

  bool HasAttr(const std::string &attr_name) const {
    auto iter = attrs_.find(attr_name);
    return !(iter == attrs_.cend());
  }

  bool operator==(const Value &other) const override;
  bool operator==(const Cell &other) const;
  ~Cell() override = default;

 private:
  std::unordered_map<std::string, ValuePtr> attrs_;
};

using CellPtr = std::shared_ptr<Cell>;
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_IR_CELL_H_
