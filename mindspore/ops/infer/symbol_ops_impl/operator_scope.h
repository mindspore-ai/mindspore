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
#ifndef MINDSPORE_CORE_OPS_SYMBOL_OPS_IMPL_OPERATOR_SCOPE_H_
#define MINDSPORE_CORE_OPS_SYMBOL_OPS_IMPL_OPERATOR_SCOPE_H_

#include <memory>
#include "mindspore/ops/infer/symbol_ops_impl/scalar_add.h"
#include "mindspore/ops/infer/symbol_ops_impl/scalar_sub.h"
#include "mindspore/ops/infer/symbol_ops_impl/scalar_mul.h"
#include "mindspore/ops/infer/symbol_ops_impl/scalar_div.h"

namespace mindspore {
namespace symshape {
namespace ops {
/// \brief This class is to bind an OperationEmitter to support operator overloading of Symbol
///
/// The operator overloading supports "+" for ScalarAdd, "-" for ScalarSub, "*" for ScalarMul and "/".
/// when using "/", the division type should be specific (ScalarDiv, ScalarFloorDiv or ScalarCeilDiv), defaults exact
/// division for ScalarDiv.
class OPS_API OperatorScope {
 public:
  enum class DivType { EXACT_DIV, FLOOR_DIV, CEIL_DIV };
  explicit OperatorScope(const OperationEmitter &e, DivType div_type = DivType::EXACT_DIV)
      : emitter_(e), div_type_(div_type) {}
  ~OperatorScope() = default;

  class OPS_API SymbolHolder {
   public:
    friend class OperatorScope;
    ~SymbolHolder() = default;
    // implicitly convert SymbolHolder to SymbolPtr
    operator SymbolPtr() const { return sym_; }
    template <typename T>
    std::shared_ptr<T> as() {
      return sym_->as_sptr<T>();
    }

    SymbolHolder operator+(const SymbolHolder &s2) const { return (*this) + s2.sym_; }
    SymbolHolder operator-(const SymbolHolder &s2) const { return (*this) - s2.sym_; }
    SymbolHolder operator*(const SymbolHolder &s2) const { return (*this) * s2.sym_; }
    SymbolHolder operator/(const SymbolHolder &s2) const { return (*this) / s2.sym_; }
    SymbolHolder operator+(const SymbolPtr &s2) const {
      return SymbolHolder(e_.Emit(std::make_shared<ScalarAdd>(sym_, s2)), div_, e_);
    }
    SymbolHolder operator-(const SymbolPtr &s2) const {
      return SymbolHolder(e_.Emit(std::make_shared<ScalarSub>(sym_, s2)), div_, e_);
    }
    SymbolHolder operator*(const SymbolPtr &s2) const {
      return SymbolHolder(e_.Emit(std::make_shared<ScalarMul>(sym_, s2)), div_, e_);
    }
    SymbolHolder operator/(const SymbolPtr &s2) const {
      if (div_ == OperatorScope::DivType::EXACT_DIV) {
        return SymbolHolder(e_.Emit(std::make_shared<ScalarDiv>(sym_, s2)), div_, e_);
      }
      if (div_ == OperatorScope::DivType::CEIL_DIV) {
        return SymbolHolder(e_.Emit(std::make_shared<ScalarCeilDiv>(sym_, s2)), div_, e_);
      }
      // OperatorScope::DivType::FLOOR_DIV
      return SymbolHolder(e_.Emit(std::make_shared<ScalarFloorDiv>(sym_, s2)), div_, e_);
    }

    friend SymbolHolder operator+(const SymbolPtr &s1, const SymbolHolder &s2);
    friend SymbolHolder operator-(const SymbolPtr &s1, const SymbolHolder &s2);
    friend SymbolHolder operator*(const SymbolPtr &s1, const SymbolHolder &s2);
    friend SymbolHolder operator/(const SymbolPtr &s1, const SymbolHolder &s2);

   private:
    SymbolHolder(const SymbolPtr &s, DivType d, const OperationEmitter &e) : sym_(s), div_(d), e_(e) {}
    SymbolPtr sym_;
    DivType div_;
    const OperationEmitter &e_;
  };

  // this function create a symbol holder for operator + - * /
  SymbolHolder operator()(const SymbolPtr &s) { return SymbolHolder(s, div_type_, emitter_); }
  const OperationEmitter &emitter_;
  DivType div_type_;
};

inline OperatorScope::SymbolHolder operator+(const SymbolPtr &s1, const OperatorScope::SymbolHolder &s2) {
  return OperatorScope::SymbolHolder(s2.e_.Emit(std::make_shared<ScalarAdd>(s1, s2.sym_)), s2.div_, s2.e_);
}
inline OperatorScope::SymbolHolder operator-(const SymbolPtr &s1, const OperatorScope::SymbolHolder &s2) {
  return OperatorScope::SymbolHolder(s2.e_.Emit(std::make_shared<ScalarSub>(s1, s2.sym_)), s2.div_, s2.e_);
}
inline OperatorScope::SymbolHolder operator*(const SymbolPtr &s1, const OperatorScope::SymbolHolder &s2) {
  return OperatorScope::SymbolHolder(s2.e_.Emit(std::make_shared<ScalarMul>(s1, s2.sym_)), s2.div_, s2.e_);
}
inline OperatorScope::SymbolHolder operator/(const SymbolPtr &s1, const OperatorScope::SymbolHolder &s2) {
  if (s2.div_ == OperatorScope::DivType::EXACT_DIV) {
    return OperatorScope::SymbolHolder(s2.e_.Emit(std::make_shared<ScalarDiv>(s1, s2.sym_)), s2.div_, s2.e_);
  }
  if (s2.div_ == OperatorScope::DivType::CEIL_DIV) {
    return OperatorScope::SymbolHolder(s2.e_.Emit(std::make_shared<ScalarCeilDiv>(s1, s2.sym_)), s2.div_, s2.e_);
  }
  // OperatorScope::DivType::FLOOR_DIV
  return OperatorScope::SymbolHolder(s2.e_.Emit(std::make_shared<ScalarFloorDiv>(s1, s2.sym_)), s2.div_, s2.e_);
}
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_SYMBOL_OPS_IMPL_OPERATOR_SCOPE_H_
