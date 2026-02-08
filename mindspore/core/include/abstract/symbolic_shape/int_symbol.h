/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CORE_ABSTRACT_SYMBOLIC_SHAPE_INT_SYMBOL_H_
#define MINDSPORE_CORE_ABSTRACT_SYMBOLIC_SHAPE_INT_SYMBOL_H_

#include <memory>
#include <string>
#include "abstract/symbolic_shape/symbol.h"
#include "abstract/symbolic_shape/math_info.h"

namespace mindspore {
namespace symshape {
class MS_CORE_API IntSymbol final : public ScalarSymbol {
 public:
  using elem_type = int64_t;
  /// \brief make a constant IntSymbol
  /// \param val the symbol value
  /// \param op the Operation that created this symbol.
  /// \return IntSymbolPtr
  static IntSymbolPtr Make(int64_t val, const OpPtr &op = nullptr);

  /// \brief make a variable IntSymbol
  /// \param op the Operation that created this symbol.
  /// \return IntSymbolPtr
  inline static IntSymbolPtr Make(const OpPtr &op = nullptr) { return std::make_shared<IntSymbol>(false, false, op); }

  IntSymbol(bool is_const, bool has_data, const OpPtr &op) : ScalarSymbol(is_const, has_data, op), math_info_(this) {}
  ~IntSymbol() override = default;
  MS_DECLARE_PARENT(IntSymbol, ScalarSymbol)
  friend class MathInfo;

  inline void SetValue(int64_t v) {
    MS_EXCEPTION_IF_CHECK_FAIL(!is_const_, ToString() + " is const symbol and cannot be updated.");
    has_data_ = true;
    value_ = v;
  }
  inline int64_t value() const {
    MS_EXCEPTION_IF_CHECK_FAIL(has_data_, ToString() + "has no value.");
    return value_;
  }

  std::string ToRawString() const override;
  std::string ToString() const override;
  ValuePtr ToValue() const override;
  ValuePtr ToValueOf(const TypePtr &type) const override;

  bool operator==(const Symbol &s) const override;
  bool operator<(const IntSymbol &s) const;
  bool operator<=(const IntSymbol &s) const;
  bool operator>(const IntSymbol &s) const { return s < *this; }
  bool operator>=(const IntSymbol &s) const { return s <= *this; }

  /// \brief Set the minimum value of symbol.
  void SetRangeMin(int64_t m) { math_info_.SetRangeMin(m); }
  /// \brief Set the maximum value of symbol.
  void SetRangeMax(int64_t m) { math_info_.SetRangeMax(m); }
  /// \brief Set the minimum and maximum values of symbol.
  void SetRange(int64_t minv, int64_t maxv) {
    SetRangeMin(minv);
    SetRangeMax(maxv);
  }
  /// \brief Set the symbol to positive, exception will be throw if the symbol is always negative.
  void SetPositive();
  /// \brief Set the divisor and remainder value of symbol, the symbol is "d * N + r", for N is positive integer.
  void SetDivisorRemainder(int64_t d, int64_t r) { math_info_.SetDivisorRemainder(d, r); }

  /// \brief Get the minimum value of symbol.
  int64_t range_min() const { return math_info_.range_min(); }
  /// \brief Get the maximum value of symbol.
  int64_t range_max() const { return math_info_.range_max(); }
  /// \brief Get the divisor value of symbol.
  int64_t divisor() const { return math_info_.divisor(); }
  /// \brief Get the remainder value of symbol.
  int64_t remainder() const { return math_info_.remainder(); }

  /// \brief Check the symbol is divisible by 'd'
  bool is_divisible_by(int64_t d) const;
  bool is_divisible_by(const IntSymbolPtr &d) const;
  /// \brief Check the symbol is ALWAYS greater than x
  bool is_greater_than(int64_t x) const { return has_data_ ? value_ > x : range_min() > x; }
  /// \brief Check the symbol is ALWAYS greater than or equal to x
  bool is_greater_equal(int64_t x) const { return has_data_ ? value_ >= x : range_min() >= x; }
  /// \brief Check the symbol is ALWAYS less than x
  bool is_less_than(int64_t x) const { return has_data_ ? value_ < x : range_max() < x; }
  /// \brief Check the symbol is ALWAYS less than or equal to x
  bool is_less_equal(int64_t x) const { return has_data_ ? value_ <= x : range_max() <= x; }
  /// \brief Check the symbol is ALWAYS positive
  bool is_positive() const { return is_greater_than(0); }
  /// \brief Check the symbol is ALWAYS negative
  bool is_negative() const { return is_less_than(0); }
  /// \brief Check the symbol's available value is subset of symbol `s`
  bool is_subset_of(const IntSymbol *s, bool strict = false) const;

  /// \brief Set the two symbols are equal.
  void SetEqual(const IntSymbolPtr &other) {
    MS_EXCEPTION_IF_NULL(other);
    // only variables can set equal
    if (!is_const_ && !other->is_const_) {
      math_info_.SetEqual(other);
    }
  }
  /// \brief Set the expression of symbol to "this = s * a + b"
  void SetMathExpr(const IntSymbolPtr &s, const Frac &a, int64_t b) { math_info_.SetMathExpr(s, a, b); }

 protected:
  void UpdateImpl(const SymbolPtr &s) override;
  MathInfo math_info_;
  int64_t value_{0};
};
}  // namespace symshape
GVAR_DEF(IntSymbolPtr, kSym0, IntSymbol::Make(0));
GVAR_DEF(IntSymbolPtr, kSym1, IntSymbol::Make(1));
GVAR_DEF(IntSymbolPtr, kSym2, IntSymbol::Make(2));
GVAR_DEF(IntSymbolPtr, kSymNeg1, IntSymbol::Make(-1));
}  // namespace mindspore
#endif  // MINDSPORE_CORE_ABSTRACT_SYMBOLIC_SHAPE_INT_SYMBOL_H_
