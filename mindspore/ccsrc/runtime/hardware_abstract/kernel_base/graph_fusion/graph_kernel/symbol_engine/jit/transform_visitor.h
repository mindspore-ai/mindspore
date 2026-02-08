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

#include "abstract/symbolic_shape/symbol_visitor.h"
#include "runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/symbol_engine/jit/syntax.h"

namespace mindspore::graphkernel::symshape {
using mindspore::symshape::Operation;
using mindspore::symshape::SymbolVisitor;

/// \brief transform Symbol to ast::Symbol
class TransformVisitor : public SymbolVisitor {
 public:
  TransformVisitor() = default;
  ~TransformVisitor() = default;

  /// \brief Init Transformer.
  /// Get abstract index information
  void Init(const FuncGraphPtr &func_graph);

  /// \brief Transform the output symbol to ast::shape
  bool Transform(const FuncGraphPtr &func_graph);
  void VisitImpl(DynamicSymbol *symbol) override { MS_LOG(EXCEPTION) << "Unexpected DynamicSymbol"; }
  void VisitImpl(ScalarSymbol *symbol) override { MS_LOG(EXCEPTION) << "Unexpected ScalarSymbol"; }
  void VisitImpl(IntSymbol *symbol) override;
  void VisitImpl(BoolSymbol *symbol) override { MS_LOG(EXCEPTION) << "Unsupported BoolSymbol"; }
  void VisitImpl(FloatSymbol *symbol) override { MS_LOG(EXCEPTION) << "Unsupported FloatSymbol"; }
  void VisitImpl(ListSymbol *symbol) override;

  void VisitImpl(Operation *op) override;

  const ast::SymbolTable &GetSymbolTable() { return symbols_table_; }
  const std::vector<ast::ShapePtr> &GetShapes() const { return shapes_; }

  std::string SymbolExprPrint();

 protected:
  // A helper to emit BinOp
  void EmitBinOp(ast::BinOpType type, const Operation *operation);
  ast::VarPtr NewVal(ast::TermPtr term, const std::string &name);

 protected:
  std::vector<ast::ShapePtr> shapes_;
  std::vector<ast::TermPtr> symbols_;                      // a stack used in transforming Symbol to ast::Symbol
  std::unordered_map<std::string, ast::VarPtr> temp_map_;  // helper map used to record Symbol
  ast::SymbolTable symbols_table_;                         // a map: id -> symbol
};

inline void TransformVisitor::EmitBinOp(ast::BinOpType optype, const Operation *operation) {
  ast::BinOp op;
  op.optype_ = optype;
  Visit(operation->input(1).get());
  op.b_ = symbols_.back();
  symbols_.pop_back();
  Visit(operation->input(0).get());
  op.a_ = symbols_.back();
  symbols_.pop_back();
  ast::SingleTermPtr smbl_p = std::make_shared<ast::BinOp>(std::move(op));
  symbols_.push_back(smbl_p);
}
}  // namespace mindspore::graphkernel::symshape
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_JIT_TRANSFORM_VISITOR_H_
