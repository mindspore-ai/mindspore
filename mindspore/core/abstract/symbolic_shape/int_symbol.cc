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

#include "abstract/symbolic_shape/int_symbol.h"
#include "abstract/symbolic_shape/utils.h"
#include "ir/dtype/tensor_type.h"

namespace mindspore {
namespace symshape {
std::string IntSymbol::ToRawString() const { return has_data_ ? std::to_string(value_) : sid(); }

ValuePtr IntSymbol::ToValue() const { return has_data_ ? MakeValue<int64_t>(value_) : kValueAny; }
ValuePtr IntSymbol::ToValueOf(const TypePtr &type) const {
  if (!has_data_) {
    return kValueAny;
  }
  TypeId type_id = type->type_id();
  if (type_id == kObjectTypeTensorType) {
    auto tensor_type = type->cast_ptr<TensorType>();
    MS_EXCEPTION_IF_NULL(tensor_type);
    type_id = tensor_type->element()->type_id();
  }
  switch (type_id) {
    case kNumberTypeInt64:
      return MakeValue<int64_t>(value_);
    case kNumberTypeInt32:
      return MakeValue<int32_t>(static_cast<int32_t>(value_));
    case kNumberTypeInt16:
      return MakeValue<int16_t>(static_cast<int16_t>(value_));
    case kNumberTypeInt8:
      return MakeValue<int8_t>(static_cast<int8_t>(value_));
    case kNumberTypeUInt64:
      return MakeValue<uint64_t>(static_cast<uint64_t>(value_));
    case kNumberTypeUInt32:
      return MakeValue<uint32_t>(static_cast<uint32_t>(value_));
    case kNumberTypeUInt16:
      return MakeValue<uint16_t>(static_cast<uint16_t>(value_));
    case kNumberTypeUInt8:
      return MakeValue<uint8_t>(static_cast<uint8_t>(value_));
    default:
      MS_LOG(INTERNAL_EXCEPTION) << "Cannot convert the IntSymbol to type " << type->ToString();
  }
  return ToValue();
}

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

void IntSymbol::SetPositive() {
  if (has_data_) {
    // in some case, there is item "0" in shape.
    if (value_ < 0) {
      MS_LOG(INTERNAL_EXCEPTION) << "The symbol " << ToString() << " can not be set to positive.";
    }
    return;
  }
  if (is_less_equal(0)) {
    MS_LOG(INTERNAL_EXCEPTION) << "The symbol " << ToString() << " can not be set to positive.";
  }
  if (!is_positive()) {
    SetRangeMin(1);
  }
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

bool IntSymbol::is_subset_of(const IntSymbol *other, bool strict) const {
  if (*this == *other) {
    return true;
  }
  if (this->is_const()) {
    auto v = this->value();
    if (other->is_const()) {
      return v == other->value();
    }
    if ((other->range_min() <= v) && (v <= other->range_max())) {
      if (strict && other->math_info_.relation_expr_.s != nullptr) {
        // other = a * s + b, check v in {other}
        auto s = other->math_info_.relation_expr_.s;
        auto a = other->math_info_.relation_expr_.a;
        auto b = other->math_info_.relation_expr_.b;
        auto tmp = (Frac(v) - b) / a;
        return tmp.is_int() && IntSymbol::Make(tmp.x())->is_subset_of(s.get(), strict);
      }
      // v in {other.d * N + other.r}
      return (v - other->remainder()) % other->divisor() == 0;
    }
    return false;
  }
  if (other->is_const()) {
    return false;
  }
  if (strict) {
    if (this->math_info_.relation_expr_.s != other->math_info_.relation_expr_.s &&
        other->math_info_.relation_expr_.s != nullptr) {
      return false;
    }
  }
  if (this->range_min() < other->range_min() || this->range_max() > other->range_max()) {
    return false;
  }
  auto d1 = this->divisor();
  auto r1 = this->remainder();
  auto d2 = other->divisor();
  auto r2 = other->remainder();
  if (r1 == r2) {
    return d1 % d2 == 0;
  }
  return false;
}
}  // namespace symshape
}  // namespace mindspore
