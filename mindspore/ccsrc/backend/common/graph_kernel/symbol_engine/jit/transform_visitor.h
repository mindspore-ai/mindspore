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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_JIT_TRANSFORM_VISITOR_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_JIT_TRANSFORM_VISITOR_H_

#include <exception>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "backend/common/graph_kernel/symbol_engine/symbol_visitor.h"
#include "backend/common/graph_kernel/symbol_engine/jit/syntax.h"
#include "backend/common/graph_kernel/symbol_engine/symbol_engine_impl.h"

namespace mindspore::graphkernel::symbol {

struct TransformError : public std::runtime_error {
  explicit TransformError(const char *msg) : std::runtime_error(msg) {}
};

/// \brief transform Symbol to ast::Symbol
class TransformVisitor : public SymbolVisitor {
 public:
  TransformVisitor() = default;
  ~TransformVisitor() = default;

  /// \brief Init Transformer.
  /// Get abstract index information
  void Init(const std::shared_ptr<SymbolEngineImpl> &symbol_engine);

  using SymbolVisitor::Visit;
  /// \brief Transform a Symbol to ast::shape
  bool Transform(Symbol *symbol);
  void Visit(DynamicSymbol *symbol) override { MS_LOG(EXCEPTION) << "Unexpected Symbol"; }
  void Visit(InputSymbol *symbol) override;
  void Visit(ScalarSymbol *symbol) override;
  void Visit(IntSymbol *symbol) override;
  void Visit(BoolSymbol *symbol) override;
  void Visit(FloatSymbol *symbol) override;
  void Visit(ListSymbol *symbol) override;
  void Visit(IListSymbol *symbol) override;

  void Visit(ops::Operation *op) override;

  const ast::SymbolTable &GetSymbolTable() { return symbols_table_; }
  const std::vector<ast::ShapePtr> &GetShapes() const { return shapes_; }

  std::string SymbolExprPrint();

 protected:
  // A helper to emit BinOp
  void EmitBinOp(ast::BinOpType type, const ops::Operation *operation);
  ast::VarPtr NewVal(ast::TermPtr term, const std::string &name);

 protected:
  std::vector<ast::ShapePtr> shapes_;
  std::vector<ast::TermPtr> symbols_;                      // a stack used in transforming Symbol to ast::Symbol
  std::unordered_map<std::string, ast::VarPtr> temp_map_;  // helper map used to record Symbol
  ast::SymbolTable symbols_table_;                         // a map: id -> symbol
};

inline void TransformVisitor::EmitBinOp(ast::BinOpType optype, const ops::Operation *operation) {
  ast::BinOp op;
  op.optype_ = optype;
  SymbolVisitor::Visit(operation->input(1).get());
  op.b_ = symbols_.back();
  symbols_.pop_back();
  SymbolVisitor::Visit(operation->input(0).get());
  op.a_ = symbols_.back();
  symbols_.pop_back();
  ast::SingleTermPtr smbl_p = std::make_shared<ast::BinOp>(std::move(op));
  symbols_.push_back(smbl_p);
}
}  // namespace mindspore::graphkernel::symbol
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_JIT_TRANSFORM_VISITOR_H_
