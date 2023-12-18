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
#ifndef MINDSPORE_CORE_SYMBOLIC_SHAPE_SYMBOL_H_
#define MINDSPORE_CORE_SYMBOLIC_SHAPE_SYMBOL_H_
#include <memory>
#include <vector>
#include <algorithm>
#include <ostream>
#include <string>
#include <utility>
#include "base/base.h"
#include "mindspore/core/symbolic_shape/math_info.h"

namespace mindspore {
namespace symshape {
class Symbol;
using SymbolPtr = std::shared_ptr<Symbol>;
using SymbolPtrList = std::vector<SymbolPtr>;

class Operation;
using OpPtr = std::shared_ptr<Operation>;
using OpPtrList = std::vector<OpPtr>;
using OpWeakPtr = std::weak_ptr<Operation>;

/// \brief The base class of symbol objects in symbolic shape.
///
/// The symbol can represent a shape, items of shape, or values for inferring shape, etc.
///
/// NOTE: The 'cast' and 'isa' function of Base is hid, 'cast_ptr' can be used to convert the original symbol.
/// Use 'as' and 'is' to cast and check the type of symbol, to make the `DynamicSymbol` transparent in most situation.
class MS_CORE_API Symbol : public Base {
 public:
  /// \brief Constructor of Symbol
  ///
  /// \param[in] op The operation that built this symbol (if exists)
  explicit Symbol(const OpPtr &op = nullptr) : operation_(op) {}
  ~Symbol() override = default;
  MS_DECLARE_PARENT(Symbol, Base)

  /// \brief Update the symbol data in runtime. Only variable symbol can be updated.
  inline void Update(const SymbolPtr &s) {
    if (s != nullptr && s.get() != this) {
      UpdateImpl(s);
    }
  }

  /// \brief Whether the symbol has data.
  ///
  /// Variable symbol has no data in compiling, it has data after updating in runtime.
  /// Constant symbol always has data.
  virtual bool HasData() const { return true; }

  /// @brief Whether the symbol can be updated in runtime, only variable symbol can be updated.
  virtual bool CanUpdate() const { return true; }

  /// \brief Whether two symbols are equal in mathematic.
  virtual bool operator==(const Symbol &s) const { return this == &s; }

  /// \brief Whether two symbols are equal in mathematic.
  inline bool EqualsTo(const SymbolPtr &other) const { return (other != nullptr) && ((*this) == (*other)); }

  /// \brief Get the raw data of symbol.
  virtual std::string ToRawString() const { return ToString(); }

  /// \brief Get the operation that built this symbol.
  inline OpPtr operation() const { return operation_.lock(); }

  template <typename T>
  inline bool is() const {
    auto *s = const_cast<Symbol *>(this)->real_symbol();
    return s != nullptr && s->isa<T>();
  }
  template <typename T>
  inline T *as() {
    auto s = real_symbol();
    return s == nullptr ? nullptr : s->cast_ptr<T>();
  }
  template <typename T>
  inline const T *as() const {
    auto *s = const_cast<Symbol *>(this)->real_symbol();
    return s == nullptr ? nullptr : s->cast_ptr<T>();
  }
  template <typename T>
  inline std::shared_ptr<T> as_sptr() {
    auto s = real_symbol();
    return s == nullptr ? nullptr : s->cast<std::shared_ptr<T>>();
  }

 protected:
  using Base::cast;
  using Base::isa;
  virtual void UpdateImpl(const SymbolPtr &s) {
    MS_EXCEPTION(NotImplementedError) << "The 'Update' of " << type_name() << " is not implemented.";
  }
  virtual Symbol *real_symbol() { return this; }
  inline std::string sid() const { return "s" + std::to_string(id()); }
  OpWeakPtr operation_;

 private:
  size_t id() const;
  mutable size_t id_{0};
};

/// \brief DynamicSymbol represents the symbol type is dynamic, such as "symbol of scalar or list".
class MS_CORE_API DynamicSymbol : public Symbol {
 public:
  using Symbol::Symbol;
  ~DynamicSymbol() override = default;
  MS_DECLARE_PARENT(DynamicSymbol, Symbol)
  inline static std::shared_ptr<DynamicSymbol> Make(const OpPtr &op = nullptr) {
    return std::make_shared<DynamicSymbol>(op);
  }
  bool operator==(const Symbol &s) const override { return (this == &s) || ((symbol_ != nullptr) && (*symbol_ == s)); }
  bool HasData() const override { return symbol_ != nullptr; }
  std::string ToString() const override { return symbol_ == nullptr ? "DYN-" + sid() : symbol_->ToString(); }
  std::string ToRawString() const override { return symbol_ == nullptr ? sid() : symbol_->ToRawString(); }
  const SymbolPtr &symbol() const { return symbol_; }

 protected:
  void UpdateImpl(const SymbolPtr &s) override;
  Symbol *real_symbol() override { return symbol_.get(); }
  SymbolPtr symbol_{nullptr};
};
using DynamicSymbolPtr = std::shared_ptr<DynamicSymbol>;

/// \brief The base class of scalar objects.
class MS_CORE_API ScalarSymbol : public Symbol {
 public:
  ScalarSymbol(bool is_const, bool has_data, const OpPtr &op) : Symbol(op), is_const_(is_const), has_data_(has_data) {}
  ~ScalarSymbol() override = default;
  MS_DECLARE_PARENT(ScalarSymbol, Symbol)
  bool HasData() const override { return has_data_; }
  bool CanUpdate() const override { return !is_const_; }
  bool operator==(const Symbol &s) const override;
  std::string ToString() const override { return ToRawString(); }
  bool is_const() const { return is_const_; }

 protected:
  void UpdateImpl(const SymbolPtr &s) override;
  /// \brief set value, called by `UpdateImpl`
  virtual void SetValueByScalar(const Symbol *s) {
    MS_EXCEPTION(NotImplementedError) << "The 'SetValueByScalar' of " << type_name() << " is not implemented.";
  }
  /// \brief check value equal, called by `operator==`
  virtual bool CheckEqualValue(const Symbol *s) const {
    MS_EXCEPTION(NotImplementedError) << "The 'CheckEqualValue' of " << type_name() << " is not implemented.";
  }

  bool is_const_;
  bool has_data_;
};
using ScalarSymbolPtr = std::shared_ptr<ScalarSymbol>;

class MS_CORE_API BoolSymbol : public ScalarSymbol {
 public:
  using ScalarSymbol::ScalarSymbol;
  ~BoolSymbol() override = default;
  MS_DECLARE_PARENT(BoolSymbol, ScalarSymbol)
  static inline std::shared_ptr<BoolSymbol> Make(bool val, const OpPtr &op = nullptr) {
    auto s = std::make_shared<BoolSymbol>(true, true, op);
    s->value_ = val;
    return s;
  }
  static inline std::shared_ptr<BoolSymbol> Make(const OpPtr &op = nullptr) {
    return std::make_shared<BoolSymbol>(false, false, op);
  }
  inline void SetValue(bool v) {
    MS_EXCEPTION_IF_CHECK_FAIL(!is_const_, ToString() + " is const symbol and cannot be updated.");
    has_data_ = true;
    value_ = v;
  }
  inline bool value() const {
    MS_EXCEPTION_IF_CHECK_FAIL(has_data_, ToString() + "has no value.");
    return value_;
  }
  std::string ToRawString() const override;

 protected:
  void SetValueByScalar(const Symbol *s) override { value_ = static_cast<const BoolSymbol *>(s)->value_; }
  bool CheckEqualValue(const Symbol *s) const override { return value_ == static_cast<const BoolSymbol *>(s)->value_; }

  bool value_{false};
};
using BoolSymbolPtr = std::shared_ptr<BoolSymbol>;

class MS_CORE_API FloatSymbol : public ScalarSymbol {
 public:
  using ScalarSymbol::ScalarSymbol;
  ~FloatSymbol() override = default;
  MS_DECLARE_PARENT(FloatSymbol, ScalarSymbol)
  static inline std::shared_ptr<FloatSymbol> Make(double val, const OpPtr &op = nullptr) {
    auto s = std::make_shared<FloatSymbol>(true, true, op);
    s->value_ = val;
    return s;
  }
  static inline std::shared_ptr<FloatSymbol> Make(const OpPtr &op = nullptr) {
    return std::make_shared<FloatSymbol>(false, false, op);
  }
  inline void SetValue(double v) {
    MS_EXCEPTION_IF_CHECK_FAIL(!is_const_, ToString() + " is const symbol and cannot be updated.");
    has_data_ = true;
    value_ = v;
  }
  inline double value() const {
    MS_EXCEPTION_IF_CHECK_FAIL(has_data_, ToString() + "has no value.");
    return value_;
  }
  std::string ToRawString() const override;

 protected:
  void SetValueByScalar(const Symbol *s) override { value_ = static_cast<const FloatSymbol *>(s)->value_; }
  bool CheckEqualValue(const Symbol *s) const override { return value_ == static_cast<const FloatSymbol *>(s)->value_; }

  double value_{0};
};
using FloatSymbolPtr = std::shared_ptr<FloatSymbol>;

class MS_CORE_API StrSymbol : public ScalarSymbol {
 public:
  using ScalarSymbol::ScalarSymbol;
  ~StrSymbol() override = default;
  MS_DECLARE_PARENT(StrSymbol, ScalarSymbol)
  static inline std::shared_ptr<StrSymbol> Make(const std::string &val, const OpPtr &op = nullptr) {
    auto s = std::make_shared<StrSymbol>(true, true, op);
    s->value_ = val;
    return s;
  }
  static inline std::shared_ptr<StrSymbol> Make(const OpPtr &op = nullptr) {
    return std::make_shared<StrSymbol>(false, false, op);
  }
  inline void SetValue(const std::string &v) {
    MS_EXCEPTION_IF_CHECK_FAIL(!is_const_, ToString() + " is const symbol and cannot be updated.");
    has_data_ = true;
    value_ = v;
  }
  inline const std::string &value() const {
    MS_EXCEPTION_IF_CHECK_FAIL(has_data_, ToString() + "has no value.");
    return value_;
  }
  std::string ToRawString() const override;

 protected:
  void SetValueByScalar(const Symbol *s) override { value_ = static_cast<const StrSymbol *>(s)->value_; }
  bool CheckEqualValue(const Symbol *s) const override { return value_ == static_cast<const StrSymbol *>(s)->value_; }

  std::string value_;
};
using StrSymbolPtr = std::shared_ptr<StrSymbol>;

class MS_CORE_API IntSymbol : public ScalarSymbol {
 public:
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
  /// \brief Get the minimum value of symbol.
  int64_t range_min() const { return math_info_.range_min(); }
  /// \brief Get the maximum value of symbol.
  int64_t range_max() const { return math_info_.range_max(); }
  /// \brief Check the symbol is ALWAYS greater than x
  bool is_greater_than(int64_t x) const { return range_min() > x; }
  /// \brief Check the symbol is ALWAYS less than x
  bool is_less_than(int64_t x) const { return range_max() < x; }
  /// \brief Check the symbol is ALWAYS positive
  bool is_positive() const { return is_greater_than(0); }
  /// \brief Check the symbol is ALWAYS negative
  bool is_negative() const { return is_less_than(0); }

  /// \brief Set the two symbols are equal.
  void SetEqual(const IntSymbolPtr &other) { math_info_.SetEqual(other); }
  /// \brief Set the expression of symbol to "this = s * a + b"
  void SetMathExpr(const IntSymbolPtr &s, const Frac &a, int64_t b) { math_info_.SetMathExpr(s, a, b); }

 protected:
  void UpdateImpl(const SymbolPtr &s) override;
  MathInfo math_info_;
  int64_t value_{0};
};

class MS_CORE_API ListSymbol : public Symbol {
 public:
  using SPtr = std::shared_ptr<ListSymbol>;
  ListSymbol(const SymbolPtrList &slist, const OpPtr &op) : Symbol(op), symbols_(slist) {}
  ListSymbol(SymbolPtrList &&slist, const OpPtr &op) : Symbol(op), symbols_(slist) {}
  ListSymbol(const std::initializer_list<SymbolPtr> &slist, const OpPtr &op) : Symbol(op), symbols_(slist) {}
  explicit ListSymbol(const OpPtr &op) : Symbol(op), is_dyn_len_(true), has_data_(false) {}
  ~ListSymbol() override = default;
  MS_DECLARE_PARENT(ListSymbol, Symbol)

  static inline SPtr Make(const SymbolPtrList &slist, const OpPtr &op = nullptr) {
    return std::make_shared<ListSymbol>(slist, op);
  }
  static inline SPtr Make(SymbolPtrList &&slist, const OpPtr &op = nullptr) {
    return std::make_shared<ListSymbol>(slist, op);
  }
  static inline SPtr Make(const std::initializer_list<SymbolPtr> &slist, const OpPtr &op = nullptr) {
    return std::make_shared<ListSymbol>(slist, op);
  }
  static inline SPtr Make(const OpPtr &op = nullptr) { return std::make_shared<ListSymbol>(op); }

  bool operator==(const Symbol &s) const override;
  std::string ToString() const override;
  std::string ToRawString() const override;

  bool HasData() const override { return has_data_; }
  bool AllHaveData() const {
    return has_data_ && std::all_of(symbols_.cbegin(), symbols_.cend(), [](const SymbolPtr &s) {
             return s->is<ListSymbol>() ? s->as<ListSymbol>()->AllHaveData() : s->HasData();
           });
  }
  bool CanUpdate() const override {
    return is_dyn_len_ || std::any_of(symbols_.cbegin(), symbols_.cend(), [](auto &s) { return s->CanUpdate(); });
  }
  void UpdateList(const SymbolPtrList &slist);
  inline void UpdateList(SymbolPtrList &&slist) {
    if (is_dyn_len_) {
      has_data_ = true;
      symbols_ = slist;
    } else {
      UpdateList(static_cast<const SymbolPtrList &>(slist));
    }
  }
  const SymbolPtr &item(size_t i) const;
  template <typename T>
  const T *item_as(size_t i) const {
    auto ret = item(i)->as<T>();
    if (ret == nullptr) {
      MS_LOG(INTERNAL_EXCEPTION) << "Convert failed for item " << i << " of " << ToString();
    }
    return ret;
  }
  template <typename T>
  std::shared_ptr<T> item_as_sptr(size_t i) const {
    auto ret = item(i)->as_sptr<T>();
    if (ret == nullptr) {
      MS_LOG(INTERNAL_EXCEPTION) << "Convert failed for item " << i << " of " << ToString();
    }
    return ret;
  }
  const SymbolPtrList &symbols() const { return symbols_; }
  size_t size() const { return symbols_.size(); }
  bool is_dyn_len() const { return is_dyn_len_; }

 protected:
  void UpdateImpl(const SymbolPtr &s) override;
  SymbolPtrList symbols_;
  bool is_dyn_len_{false};
  bool has_data_{true};
};
using ListSymbolPtr = std::shared_ptr<ListSymbol>;
}  // namespace symshape

using symshape::BoolSymbol;
using symshape::BoolSymbolPtr;
using symshape::DynamicSymbol;
using symshape::DynamicSymbolPtr;
using symshape::FloatSymbol;
using symshape::FloatSymbolPtr;
using symshape::IntSymbol;
using symshape::IntSymbolPtr;
using symshape::ListSymbol;
using symshape::ListSymbolPtr;
using symshape::ScalarSymbol;
using symshape::ScalarSymbolPtr;
using symshape::Symbol;
using symshape::SymbolPtr;
using symshape::SymbolPtrList;
}  // namespace mindspore
#endif  // MINDSPORE_CORE_SYMBOLIC_SHAPE_SYMBOL_H_
