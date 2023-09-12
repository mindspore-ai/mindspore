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
class ScalarAdd : public Operation {
 public:
  ScalarAdd(const SymbolPtr &lhs, const SymbolPtr &rhs) : Operation({lhs, rhs}) {}
  std::string type_name() const override { return "ScalarAdd"; }

 protected:
  SymbolPtr Eval() override;
  void EvalOnRun() override { output_as<IntSymbol>()->SetValue(AsInt(input(0)) + AsInt(input(1))); };
};

class ScalarSub : public Operation {
 public:
  ScalarSub(const SymbolPtr &lhs, const SymbolPtr &rhs) : Operation({lhs, rhs}) {}
  std::string type_name() const override { return "ScalarSub"; }

 protected:
  SymbolPtr Eval() override;
  void EvalOnRun() override { output_as<IntSymbol>()->SetValue(AsInt(input(0)) - AsInt(input(1))); };
};

class ScalarMul : public Operation {
 public:
  ScalarMul(const SymbolPtr &lhs, const SymbolPtr &rhs) : Operation({lhs, rhs}) {}
  std::string type_name() const override { return "ScalarMul"; }

 protected:
  SymbolPtr Eval() override;
  void EvalOnRun() override { output_as<IntSymbol>()->SetValue(AsInt(input(0)) * AsInt(input(1))); };
};

class ScalarDiv : public Operation {
 public:
  ScalarDiv(const SymbolPtr &lhs, const SymbolPtr &rhs) : Operation({lhs, rhs}) {}
  std::string type_name() const override { return "ScalarDiv"; }

 protected:
  SymbolPtr Eval() override;
  void EvalOnRun() override { output_as<IntSymbol>()->SetValue(AsInt(input(0)) / AsInt(input(1))); };
};

class ScalarMax : public Operation {
 public:
  ScalarMax(const SymbolPtr &lhs, const SymbolPtr &rhs) : Operation({lhs, rhs}) {}
  std::string type_name() const override { return "ScalarMax"; }

 protected:
  SymbolPtr Eval() override;
  void EvalOnRun() override { output_as<IntSymbol>()->SetValue(std::max(AsInt(input(0)), AsInt(input(1)))); };
};

class ScalarMin : public Operation {
 public:
  ScalarMin(const SymbolPtr &lhs, const SymbolPtr &rhs) : Operation({lhs, rhs}) {}
  std::string type_name() const override { return "ScalarMin"; }

 protected:
  SymbolPtr Eval() override;
  void EvalOnRun() override { output_as<IntSymbol>()->SetValue(std::min(AsInt(input(0)), AsInt(input(1)))); };
};

class Product : public Operation {
 public:
  explicit Product(const SymbolPtr &input) : Operation({input}) {}
  ~Product() = default;
  std::string type_name() const override { return "Product"; }

 protected:
  SymbolPtr Eval() override;
};

class Find : public Operation {
 public:
  Find(const SymbolPtr &input, const SymbolPtr &value) : Operation({input, value}) {}
  ~Find() override = default;
  std::string type_name() const override { return "Find"; }

 protected:
  SymbolPtr Eval() override;
};

class SetValue : public Operation {
 public:
  SetValue(const SymbolPtr &input, const SymbolPtr &index, const SymbolPtr &value) : Operation({input, index, value}) {}
  ~SetValue() override = default;
  std::string type_name() const override { return "SetValue"; }

 protected:
  SymbolPtr Eval() override;
};

class ListAppend : public Operation {
 public:
  ListAppend(const SymbolPtr &a, const SymbolPtr &b) : Operation({a, b}) {}
  ~ListAppend() override = default;
  std::string type_name() const override { return "ListAppend"; }

 protected:
  SymbolPtr Eval() override;
};
}  // namespace ops
}  // namespace mindspore::graphkernel::symbol
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_OPERATIONS_COMMON_OP_H_
