/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_IR_VARIABLE_H_
#define MINDSPORE_CORE_IR_VARIABLE_H_

#include <memory>
#include <string>
#include "ir/anf.h"

namespace mindspore {
class MS_CORE_API Variable : public Value {
 public:
  explicit Variable(const ValuePtr &real_value) : real_value_(real_value) {}
  ~Variable() override = default;
  MS_DECLARE_PARENT(Variable, Value)

  abstract::AbstractBasePtr ToAbstract() override;

  const ValuePtr &real_value() const { return real_value_; }

  bool operator==(const Variable &other) const;

  bool operator==(const Value &other) const override {
    if (other.isa<Variable>()) {
      auto other_variable = static_cast<const Variable &>(other);
      return *this == other_variable;
    }
    return false;
  }

  std::string ToString() const override;

  std::string DumpText() const override;

 private:
  ValuePtr real_value_{nullptr};
};
using VariablePtr = std::shared_ptr<Variable>;
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_VARIABLE_H_
