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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_SYMBOL_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_SYMBOL_H_
#include <memory>
#include <vector>
#include <algorithm>
#include <ostream>
#include <string>
#include "base/base.h"
#include "abstract/abstract_value.h"
#include "utils/shape_utils.h"

namespace mindspore::graphkernel::symbol {
class Symbol;
using SymbolPtr = std::shared_ptr<Symbol>;
using SymbolPtrList = std::vector<SymbolPtr>;

enum class SymbolType { kBool, kInt, kList, kParam, kDynamic };

namespace ops {
class Operation;
}  // namespace ops
using OpPtr = std::shared_ptr<ops::Operation>;
using OpPtrList = std::vector<OpPtr>;
using OpWeakPtr = std::weak_ptr<ops::Operation>;

class Symbol : public Base {
 public:
  explicit Symbol(const OpPtr &op = nullptr) : operation_(op) {}
  ~Symbol() override = default;
  MS_DECLARE_PARENT(Symbol, Base)

  /// \brief Update the symbol data.
  inline void Update(const SymbolPtr &s) {
    if (s != nullptr && s.get() != this) {
      UpdateImpl(s);
    }
  }
  virtual bool HasData() const { return true; }
  virtual bool CanUpdate() const { return true; }

  virtual bool operator==(const Symbol &s) const { return this == &s; }
  inline bool EqualsTo(const SymbolPtr &other) const { return (*this) == (*other); }
  virtual std::string ToExpr() const { return ToString(); }

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
  inline OpPtr operation() const { return operation_.lock(); }

 protected:
  // hide the 'cast' and 'isa' function from Base. the 'cast_ptr' can be used to convert the original symbol.
  using Base::cast;
  using Base::isa;
  virtual void UpdateImpl(const SymbolPtr &s) {
    MS_EXCEPTION(NotImplementedError) << "The 'Update' of " << type_name() << " is not implemented.";
  }
  virtual Symbol *real_symbol() { return this; }
  inline size_t id() const {
    static size_t cur_id = 0;
    if (id_ == 0) {
      id_ = ++cur_id;
    }
    return id_;
  }
  inline std::string sid() const { return "s" + std::to_string(id()); }
  OpWeakPtr operation_;

 private:
  mutable size_t id_{0};
};

class DynamicSymbol : public Symbol {
 public:
  using Symbol::Symbol;
  ~DynamicSymbol() override = default;
  MS_DECLARE_PARENT(DynamicSymbol, Symbol)
  bool operator==(const Symbol &s) const override { return (this == &s) || ((symbol_ != nullptr) && (*symbol_ == s)); }
  bool HasData() const override { return symbol_ != nullptr; }
  std::string ToString() const override { return symbol_ == nullptr ? sid() : "D" + symbol_->ToString(); }
  std::string ToExpr() const override { return symbol_ == nullptr ? sid() : symbol_->ToExpr(); }
  const SymbolPtr &symbol() const { return symbol_; }

 protected:
  void UpdateImpl(const SymbolPtr &s) override {
    if (auto d = s->cast_ptr<DynamicSymbol>(); d != nullptr) {
      symbol_ = d->symbol_;
    } else {
      symbol_ = s;
    }
  }
  Symbol *real_symbol() override { return symbol_.get(); }
  SymbolPtr symbol_{nullptr};
};

class InputSymbol : public Symbol {
 public:
  explicit InputSymbol(const AbstractBasePtr &abs) : abstract_(abs) {}
  ~InputSymbol() override = default;
  MS_DECLARE_PARENT(InputSymbol, Symbol)
  static inline std::shared_ptr<InputSymbol> Make(const AbstractBasePtr &abs) {
    return std::make_shared<InputSymbol>(abs);
  }

  // set abstract directly, unnecessary to create a new symbol.
  inline void SetAbstract(const AbstractBasePtr &abs) { abstract_ = abs; }
  inline const AbstractBasePtr &abstract() const { return abstract_; }

  std::string ToString() const override { return "Input: " + abstract_->ToString(); }

 protected:
  AbstractBasePtr abstract_;
};
using InputSymbolPtr = std::shared_ptr<InputSymbol>;
using InputSymbolPtrList = std::vector<InputSymbolPtr>;

class ScalarSymbol : public Symbol {
 public:
  explicit ScalarSymbol(bool is_const, bool has_data, const OpPtr &op)
      : Symbol(op), is_const_(is_const), has_data_(has_data) {}
  ~ScalarSymbol() override = default;
  MS_DECLARE_PARENT(ScalarSymbol, Symbol)
  bool HasData() const override { return has_data_; }
  bool CanUpdate() const override { return !is_const_; }
  std::string ToString() const override { return is_const_ ? ("C" + ToExpr()) : ToExpr(); }
  std::string ToExpr() const override {
    if (!has_data_) {
      return sid();
    }
    std::ostringstream oss;
    GetValueStr(oss);
    return oss.str();
  }
  bool is_const() const { return is_const_; }

 protected:
  void UpdateImpl(const SymbolPtr &s) override;
  virtual void SetValueByScalar(const ScalarSymbol *s) = 0;
  virtual void GetValueStr(std::ostringstream &oss) const = 0;
  bool is_const_;
  bool has_data_;
};

#define DECLARE_SCALAR_CLASS(cls, vtype)                                                                    \
  class cls : public ScalarSymbol {                                                                         \
   public:                                                                                                  \
    using ScalarSymbol::ScalarSymbol;                                                                       \
    ~cls() override = default;                                                                              \
    MS_DECLARE_PARENT(cls, ScalarSymbol)                                                                    \
    static inline std::shared_ptr<cls> Make(vtype val, const OpPtr &op = nullptr) {                         \
      auto s = std::make_shared<cls>(true, true, op);                                                       \
      s->value_ = val;                                                                                      \
      return s;                                                                                             \
    }                                                                                                       \
    static inline std::shared_ptr<cls> Make(const OpPtr &op = nullptr) {                                    \
      return std::make_shared<cls>(false, false, op);                                                       \
    }                                                                                                       \
    bool operator==(const Symbol &s) const override {                                                       \
      if (this == &s) {                                                                                     \
        return true;                                                                                        \
      }                                                                                                     \
      if (!has_data_ || !s.HasData()) {                                                                     \
        return false;                                                                                       \
      }                                                                                                     \
      auto p = s.as<cls>();                                                                                 \
      return p != nullptr && p->value_ == value_;                                                           \
    }                                                                                                       \
    void SetValue(vtype v) {                                                                                \
      has_data_ = true;                                                                                     \
      value_ = v;                                                                                           \
    }                                                                                                       \
    vtype value() const { return value_; }                                                                  \
                                                                                                            \
   protected:                                                                                               \
    void SetValueByScalar(const ScalarSymbol *s) override { value_ = static_cast<const cls *>(s)->value_; } \
    void GetValueStr(std::ostringstream &oss) const override { oss << std::boolalpha << value_; }           \
    vtype value_{};                                                                                         \
  }

DECLARE_SCALAR_CLASS(IntSymbol, int64_t);
DECLARE_SCALAR_CLASS(BoolSymbol, bool);
DECLARE_SCALAR_CLASS(FloatSymbol, double);
#undef DECLARE_SCALAR_CLASS

std::string SymbolListToStr(const SymbolPtrList &slist, const std::string &pre, const std::string &post,
                            bool expr = false);

class ListSymbol : public Symbol {
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
  std::string ToString() const override {
    return SymbolListToStr(symbols_, (is_dyn_len_ ? "L[" : "L("), (is_dyn_len_ ? "]" : ")"));
  }
  std::string ToExpr() const override { return SymbolListToStr(symbols_, "[", "]", true); }

  bool HasData() const override { return has_data_; }
  bool AllHaveData() const {
    return has_data_ && std::all_of(symbols_.cbegin(), symbols_.cend(), [](auto &s) { return s->HasData(); });
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
  const SymbolPtrList &symbols() const { return symbols_; }
  size_t size() const { return symbols_.size(); }
  bool is_dyn_len() const { return is_dyn_len_; }

 protected:
  void UpdateImpl(const SymbolPtr &s) override;
  SymbolPtrList symbols_;
  bool is_dyn_len_{false};
  bool has_data_{true};
};

// IntList symbol
class IListSymbol : public ListSymbol {
 public:
  using SPtr = std::shared_ptr<IListSymbol>;
  using ListSymbol::ListSymbol;
  ~IListSymbol() override = default;
  MS_DECLARE_PARENT(IListSymbol, ListSymbol)

  static inline SPtr Make(const SymbolPtrList &slist, const OpPtr &op = nullptr) {
    return std::make_shared<IListSymbol>(slist, op);
  }
  static inline SPtr Make(SymbolPtrList &&slist, const OpPtr &op = nullptr) {
    return std::make_shared<IListSymbol>(slist, op);
  }
  static inline SPtr Make(const std::initializer_list<SymbolPtr> &slist, const OpPtr &op = nullptr) {
    return std::make_shared<IListSymbol>(slist, op);
  }
  static inline SPtr Make(const OpPtr &op = nullptr) { return std::make_shared<IListSymbol>(op); }
  static SPtr FromShape(const ShapeVector &shape, bool real_value = false, const OpPtr &op = nullptr);
  int64_t item(size_t i) const {
    auto shape_item = symbols_[i]->as<IntSymbol>();
    MS_EXCEPTION_IF_NULL(shape_item);
    return shape_item->value();
  }

  std::string ToString() const override {
    return SymbolListToStr(symbols_, (is_dyn_len_ ? "[" : "("), (is_dyn_len_ ? "]" : ")"));
  }
};
}  // namespace mindspore::graphkernel::symbol
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_SYMBOL_H_
