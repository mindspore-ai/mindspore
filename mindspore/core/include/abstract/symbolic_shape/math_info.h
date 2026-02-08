/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CORE_ABSTRACT_SYMBOLIC_SHAPE_MATH_INFO_H_
#define MINDSPORE_CORE_ABSTRACT_SYMBOLIC_SHAPE_MATH_INFO_H_

#include <ostream>
#include <numeric>
#include <utility>
#include <functional>
#include <memory>

#include "utils/log_adapter.h"

namespace mindspore {
namespace symshape {
constexpr int64_t kINF = 1000000000LL;  // 1e9

// calculations of the Range value
inline int64_t GetRangeInf(int64_t x) { return x < 0 ? -kINF : kINF; }
inline int64_t RangeAdd(int64_t a, int64_t b) { return std::abs(a) == kINF ? a : std::abs(b) == kINF ? b : a + b; }
inline int64_t RangeSub(int64_t a, int64_t b) { return RangeAdd(a, -b); }
inline int64_t RangeMul(int64_t a, int64_t b) {
  return (std::abs(a) == kINF || std::abs(b) == kINF) ? GetRangeInf(a * b) : a * b;
}
inline int64_t RangeDiv(int64_t a, int64_t b) {
  if (b == 0) {
    return GetRangeInf(a);
  }
  if (std::abs(b) == kINF) {
    return 0;
  }
  return std::abs(a) == kINF ? GetRangeInf(a * b) : a / b;
}

/// \brief Fraction. The `Frac(x, y)` means `x/y`.
class Frac {
 public:
  explicit Frac(int64_t numerator, int64_t denominator = 1) : x_(numerator), y_(denominator) { Normalize(); }
  ~Frac() = default;

  int64_t x() const { return x_; }
  int64_t y() const { return y_; }

  Frac operator+(const Frac &b) const { return Frac(x_ * b.y_ + y_ * b.x_, y_ * b.y_); }
  Frac operator+(int64_t b) const { return Frac(x_ + b * y_, y_); }
  Frac operator-(const Frac &b) const { return Frac(x_ * b.y_ - y_ * b.x_, y_ * b.y_); }
  Frac operator-(int64_t b) const { return Frac(x_ - b * y_, y_); }
  Frac operator-() const { return Frac(-x_, y_); }
  Frac operator*(const Frac &b) const { return Frac(x_ * b.x_, y_ * b.y_); }
  Frac operator*(int64_t b) const { return Frac(x_ * b, y_); }
  Frac operator/(const Frac &b) const { return Frac(x_ * b.y_, b.x_ * y_); }
  Frac operator/(int64_t b) const { return Frac(x_, y_ * b); }
  bool operator==(const Frac &b) const { return x_ == b.x_ && y_ == b.y_; }
  bool operator==(int64_t b) const { return x_ == b && y_ == 1; }
  bool operator<(const Frac &b) const {
    // check "this" - b < 0
    auto tx = x_ * b.y_ - y_ * b.x_;
    auto ty = y_ * b.y_;
    return tx * ty < 0;
  }
  bool operator<=(const Frac &b) const {
    // check "this" - b <= 0
    auto tx = x_ * b.y_ - y_ * b.x_;
    auto ty = y_ * b.y_;
    return tx * ty <= 0;
  }
  bool operator!=(const Frac &b) const { return !(*this == b); }
  bool operator!=(int64_t b) const { return !(*this == b); }
  bool operator>(const Frac &b) const { return b < (*this); }
  Frac rec() const { return Frac(y_, x_); }

  bool is_int() const { return y_ == 1; }

  friend std::ostream &operator<<(std::ostream &os, const Frac &a) {
    if (a.y_ == 1) return os << a.x_;
    return os << a.x_ << "/" << a.y_;
  }

 private:
  void Normalize() {
    if (y_ == 0) {
      MS_LOG(INTERNAL_EXCEPTION) << "Invalid fraction : " << x_ << "/" << y_;
    }
    if (x_ == 0) {
      y_ = 1;
      return;
    }
    auto g = std::gcd(x_, y_);
    MS_EXCEPTION_IF_CHECK_FAIL(g != 0, "The result of gcd can not be zero");
    x_ /= g;
    y_ /= g;
    // make denominator as positive.
    if (y_ < 0) {
      x_ = -x_;
      y_ = -y_;
    }
  }
  int64_t x_;
  int64_t y_;
};

inline static const Frac kFrac0(0);
inline static const Frac kFrac1(1);

class IntSymbol;
using IntSymbolPtr = std::shared_ptr<IntSymbol>;
// The expression represents 'a * s + b'
struct RelationExpr {
  IntSymbolPtr s{nullptr};
  Frac a{1, 1};
  Frac b{0, 1};
  bool operator==(const RelationExpr &other) const { return a == other.a && b == other.b; }
  bool operator<(const RelationExpr &other) const;
  // note: "a <= b" is false does not means "b < a" is true.
  bool operator<=(const RelationExpr &other) const;
};

class MS_CORE_API MathInfo {
 public:
  explicit MathInfo(IntSymbol *sym) : int_symbol_(sym), math_info_id_(++inc_math_info_id) {}
  ~MathInfo() = default;
  friend class IntSymbol;

  int64_t range_min() const { return range_.first; }
  int64_t range_max() const { return range_.second; }
  int64_t divisor() const;
  int64_t remainder() const;

  void SetRangeMin(int64_t m) { range_.first = (m < -kINF ? -kINF : (m > kINF ? kINF : m)); }
  void SetRangeMax(int64_t m) { range_.second = (m < -kINF ? -kINF : (m > kINF ? kINF : m)); }
  void SetDivisorRemainder(int64_t d, int64_t r);

  bool is_greater_than(int64_t x) const { return range_min() > x; }
  bool is_less_than(int64_t x) const { return range_max() < x; }
  bool is_positive() const { return is_greater_than(0); }
  bool is_negative() const { return is_less_than(0); }

  // bind two symbols
  void SetEqual(const IntSymbolPtr &other);
  // set "this = s * a + b"
  void SetMathExpr(const IntSymbolPtr &s, const Frac &a, int64_t b) {
    if (relation_expr_.s == nullptr && static_cast<void *>(s.get()) != static_cast<void *>(this)) {
      relation_expr_ = {s, a, Frac(b)};
    }
  }

 protected:
  void Dump(std::ostringstream &oss) const;
  bool MathEquals(const MathInfo &other) const {
    return (this->root() == other.root()) && (this->relation_expr_ == other.relation_expr_);
  }
  bool MathLess(const MathInfo &other) const;
  bool MathLessEqual(const MathInfo &other) const;
  void UpdateExprRoot() const;
  IntSymbolPtr root() const {
    if (relation_expr_.s == nullptr) {
      return ToIntSymbol();
    }
    UpdateExprRoot();
    return relation_expr_.s;
  }
  IntSymbolPtr ToIntSymbol() const;

  std::pair<int64_t, int64_t> range_{-kINF, kINF};  // close range, represent "[first, second]" in IntSymbol.
  mutable RelationExpr relation_expr_;              // if the current symbol is 's2', it represents 's2 = a * s + b'.
  std::pair<int64_t, int64_t> div_rem_{1, 0};       // divisor and remainder. it represents "d * N + r".
  IntSymbol *int_symbol_;
  const size_t math_info_id_;

 private:
  using RelationCmpFunc = std::function<bool(const RelationExpr &, const RelationExpr &)>;
  inline static size_t inc_math_info_id = 0;
};
}  // namespace symshape
}  // namespace mindspore
#endif  // MINDSPORE_CORE_ABSTRACT_SYMBOLIC_SHAPE_MATH_INFO_H_
