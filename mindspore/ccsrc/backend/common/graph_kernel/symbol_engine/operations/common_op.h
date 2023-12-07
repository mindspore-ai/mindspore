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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_OPERATIONS_COMMON_OP_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_OPERATIONS_COMMON_OP_H_
#include <algorithm>
#include <memory>
#include <vector>
#include <string>

#include "backend/common/graph_kernel/symbol_engine/symbol.h"
#include "backend/common/graph_kernel/symbol_engine/utils.h"
#include "backend/common/graph_kernel/symbol_engine/operations/operation.h"

namespace mindspore::graphkernel::symbol {
namespace ops {
class ScalarOp : public Operation {
 public:
  using Operation::Operation;
  MS_DECLARE_PARENT(ScalarOp, Operation)
};

class ScalarAdd : public ScalarOp {
 public:
  ScalarAdd(const SymbolPtr &lhs, const SymbolPtr &rhs) : ScalarOp({lhs, rhs}) { support_commutative_law_ = true; }
  MS_DECLARE_PARENT(ScalarAdd, ScalarOp)

 protected:
  SymbolPtr Eval() override;
  void EvalOnRun() override { output_as<IntSymbol>()->SetValue(AsInt(input(0)) + AsInt(input(1))); }
  void UpdateMathInfo() override;
};

class ScalarSub : public ScalarOp {
 public:
  ScalarSub(const SymbolPtr &lhs, const SymbolPtr &rhs) : ScalarOp({lhs, rhs}) {}
  MS_DECLARE_PARENT(ScalarSub, ScalarOp)

 protected:
  SymbolPtr Eval() override;
  void EvalOnRun() override { output_as<IntSymbol>()->SetValue(AsInt(input(0)) - AsInt(input(1))); }
  void UpdateMathInfo() override;
};

class ScalarMul : public ScalarOp {
 public:
  ScalarMul(const SymbolPtr &lhs, const SymbolPtr &rhs) : ScalarOp({lhs, rhs}) { support_commutative_law_ = true; }
  MS_DECLARE_PARENT(ScalarMul, ScalarOp)

 protected:
  SymbolPtr Eval() override;
  void EvalOnRun() override { output_as<IntSymbol>()->SetValue(AsInt(input(0)) * AsInt(input(1))); }
  void UpdateMathInfo() override;
};

class ScalarDiv : public ScalarOp {
 public:
  ScalarDiv(const SymbolPtr &lhs, const SymbolPtr &rhs) : ScalarOp({lhs, rhs}) {}
  MS_DECLARE_PARENT(ScalarDiv, ScalarOp)

 protected:
  SymbolPtr Eval() override;
  void EvalOnRun() override { output_as<IntSymbol>()->SetValue(AsInt(input(0)) / AsInt(input(1))); }
  void UpdateMathInfo() override;
};

class ScalarMax : public ScalarOp {
 public:
  ScalarMax(const SymbolPtr &lhs, const SymbolPtr &rhs) : ScalarOp({lhs, rhs}) { support_commutative_law_ = true; }
  MS_DECLARE_PARENT(ScalarMax, ScalarOp)

 protected:
  SymbolPtr Eval() override;
  void EvalOnRun() override { output_as<IntSymbol>()->SetValue(std::max(AsInt(input(0)), AsInt(input(1)))); }
  void UpdateMathInfo() override;
};

class ScalarMin : public ScalarOp {
 public:
  ScalarMin(const SymbolPtr &lhs, const SymbolPtr &rhs) : ScalarOp({lhs, rhs}) { support_commutative_law_ = true; }
  MS_DECLARE_PARENT(ScalarMin, ScalarOp)

 protected:
  SymbolPtr Eval() override;
  void EvalOnRun() override { output_as<IntSymbol>()->SetValue(std::min(AsInt(input(0)), AsInt(input(1)))); }
  void UpdateMathInfo() override;
};

class ScalarEQ : public Operation {
 public:
  ScalarEQ(const SymbolPtr &a, const SymbolPtr &b) : Operation({a, b}) {}
  ~ScalarEQ() override = default;
  MS_DECLARE_PARENT(ScalarEQ, Operation)
 protected:
  SymbolPtr Eval() override;
  void EvalOnRun() override { output_as<BoolSymbol>()->SetValue(AsInt(input(0)) == AsInt(input(1))); }
};
}  // namespace ops
}  // namespace mindspore::graphkernel::symbol
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_OPERATIONS_COMMON_OP_H_
