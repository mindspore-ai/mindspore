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

#include "mindspore/core/symbolic_shape/symbol.h"
#include "mindspore/core/symbolic_shape/utils.h"
#include "mindspore/core/symbolic_shape/operation.h"
#include "mindspore/core/symbolic_shape/operation_builder.h"

namespace mindspore {
namespace symshape {
namespace ops {
class MS_CORE_API InferShapeOp : public Operation {
 public:
  using Operation::Operation;
  ~InferShapeOp() override = default;
  MS_DECLARE_PARENT(InferShapeOp, Operation)
  static void SetPositive(const ListSymbol *list);

 protected:
  void UpdateMathInfo() override { SetPositive(output_as<ListSymbol>()); }
};

class MS_CORE_API InferValueOp : public Operation {
 public:
  using Operation::Operation;
  ~InferValueOp() override = default;
  MS_DECLARE_PARENT(InferValueOp, Operation)
};

class MS_CORE_API ScalarIntOp : public InferValueOp {
 public:
  using InferValueOp::InferValueOp;
  MS_DECLARE_PARENT(ScalarIntOp, InferValueOp)
};

/// \brief Set input value to output shape symbol.
///
/// \note This function will set the input value symbol to positive.
SymbolPtr TransValueToShape(OperationBuilder *b);

/// \brief accumulate int symbols, only support ScalarAdd or ScalarMul
template <typename OP>
SymbolPtr Accumulate(const SymbolPtrList &symbols, const OperationEmitter &e);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_SYMBOL_OPS_IMPL_COMMON_H_
