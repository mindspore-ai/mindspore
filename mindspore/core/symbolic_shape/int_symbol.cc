/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "mindspore/core/symbolic_shape/int_symbol.h"
#include "mindspore/core/symbolic_shape/utils.h"

namespace mindspore {
namespace symshape {
std::string IntSymbol::ToRawString() const { return has_data_ ? std::to_string(value_) : sid(); }

ValuePtr IntSymbol::ToValue() const { return has_data_ ? MakeValue<int64_t>(value_) : kValueAny; }

std::shared_ptr<IntSymbol> IntSymbol::Make(int64_t val, const OpPtr &op) {
  auto s = std::make_shared<IntSymbol>(true, true, op);
  s->value_ = val;
  s->SetRange(val, val);
  return s;
}

std::string IntSymbol::ToString() const {
  std::ostringstream oss;
  oss << ScalarSymbol::ToString();
  if (has_data_) {
    return oss.str();
  }
  math_info_.Dump(oss);
  return oss.str();
}

void IntSymbol::UpdateImpl(const SymbolPtr &s) {
  if (is_const_) {
    MS_LOG(EXCEPTION) << "Const symbol '" << ToString() << "' cannot be updated, other: " << s->ToString();
  }
  if (!s->HasData()) {
    MS_LOG(EXCEPTION) << "Symbol " << s->ToString() << " has no data.";
  }
  value_ = s->as<IntSymbol>()->value_;
  has_data_ = true;
}

bool IntSymbol::operator==(const Symbol &s) const {
  if (this == &s) {
    return true;
  }
  auto other = s.as_noexcept<IntSymbol>();
  if (other == nullptr) {
    return false;
  }
  if (has_data_ && other->has_data_) {
    return value_ == other->value_;
  }
  return math_info_.MathEquals(other->math_info_);
}

bool IntSymbol::operator<(const IntSymbol &s) const {
  if (this == &s) {
    return false;
  }
  if (has_data_ && s.has_data_) {
    return value_ < s.value_;
  }
  return math_info_.MathLess(s.math_info_);
}

bool IntSymbol::operator<=(const IntSymbol &s) const {
  if (this == &s) {
    return true;
  }
  if (has_data_ && s.has_data_) {
    return value_ <= s.value_;
  }
  return math_info_.MathLessEqual(s.math_info_);
}

bool IntSymbol::is_divisible_by(int64_t d) const {
  if (has_data_) {
    return value_ % d == 0;
  }
  return (divisor() % d == 0) && (remainder() % d == 0);
}

bool IntSymbol::is_divisible_by(const IntSymbolPtr &d) const {
  return (d->HasData() && is_divisible_by(d->value())) || (this->HasData() && value() == 0) || this->EqualsTo(d);
}
}  // namespace symshape
}  // namespace mindspore
