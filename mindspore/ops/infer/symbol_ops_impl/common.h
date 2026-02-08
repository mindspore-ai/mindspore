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
#ifndef MINDSPORE_CORE_OPS_SYMBOL_OPS_IMPL_COMMON_H_
#define MINDSPORE_CORE_OPS_SYMBOL_OPS_IMPL_COMMON_H_
#include <algorithm>
#include <memory>
#include <vector>
#include <string>
#include <utility>

#include "abstract/symbolic_shape/utils.h"
#include "abstract/symbolic_shape/operation.h"
#include "abstract/symbolic_shape/operation_builder.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace symshape {
class Symbol;
using SymbolPtr = std::shared_ptr<Symbol>;
using SymbolPtrList = std::vector<SymbolPtr>;

namespace ops {
class OPS_API InferShapeOp : public Operation {
 public:
  using Operation::Operation;
  ~InferShapeOp() override = default;
  MS_DECLARE_PARENT(InferShapeOp, Operation)
  static void SetPositive(const ListSymbol *list);

 protected:
  void UpdateMathInfo() override { SetPositive(output_as<ListSymbol>()); }
};

class OPS_API InferValueOp : public Operation {
 public:
  using Operation::Operation;
  ~InferValueOp() override = default;
  MS_DECLARE_PARENT(InferValueOp, Operation)
};

class OPS_API ScalarIntOp : public InferValueOp {
 public:
  using InferValueOp::InferValueOp;
  MS_DECLARE_PARENT(ScalarIntOp, InferValueOp)
  bool SupportCse() const override { return output() != nullptr && output()->is<IntSymbol>(); }
};

/// \brief Set input value to output shape symbol.
///
/// \note This function will set the input value symbol to positive.
SymbolPtr TransValueToShape(OperationBuilder *b);

/// \brief accumulate int symbols, only support ScalarAdd or ScalarMul.
/// This interface accumulates variable and constant symbols separately, and then accumulates them together.

/// \brief Accumulate int symbols
/// \tparam OP operation, only support ScalarAdd or ScalarMul
/// \param symbols the symbols to be accumulated
/// \param e the OperationEmitter
/// \param[out] out_var accumulated variable symbols
/// \param[out] out_const accumulated const symbols
template <typename OP>
void Accumulate(const SymbolPtrList &symbols, const OperationEmitter &e, SymbolPtr *out_var, int64_t *out_const);

/// \brief Accumulate int symbols.
///        This interface accumulates variable and constant symbols separately, and then accumulates them together.
/// \tparam OP operation, only support ScalarAdd or ScalarMul
/// \param symbols the symbols to be accumulated
/// \param e the OperationEmitter
/// \return result to accumulate all symbols
template <typename OP>
SymbolPtr Accumulate(const SymbolPtrList &symbols, const OperationEmitter &e) {
  SymbolPtr vars;
  int64_t constv;
  Accumulate<OP>(symbols, e, &vars, &constv);
  if (vars == nullptr) {
    return IntSymbol::Make(constv);
  }
  return e.Emit(std::make_shared<OP>(vars, IntSymbol::Make(constv)));
}
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_SYMBOL_OPS_IMPL_COMMON_H_
