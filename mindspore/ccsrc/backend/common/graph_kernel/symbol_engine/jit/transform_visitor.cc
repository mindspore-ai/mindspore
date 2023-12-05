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
#include "backend/common/graph_kernel/symbol_engine/jit/transform_visitor.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

namespace mindspore::graphkernel::symbol {

void TransformVisitor::Init(const std::shared_ptr<SymbolEngineImpl> &symbol_engine) {
  MS_LOG(DEBUG) << "TransformVisitor Initing...";
  auto &input_index = symbol_engine->cache().GetInputIndex();
  for (const auto &[node, idx] : input_index) {
    auto shape = symbol_engine->QuerySymbolicShape(node);
    MS_EXCEPTION_IF_NULL(shape);
    // Assume that input are all tensors
    auto shape_list = shape->as<ListSymbol>();
    for (size_t j = 0; j < shape_list->symbols().size(); ++j) {
      auto sym = shape_list->symbols()[j]->as<ScalarSymbol>();
      MS_EXCEPTION_IF_NULL(sym);
      if (!sym->HasData()) {
        auto input_sym = std::make_shared<ast::Input>(idx, j);
        auto val = NewVal(input_sym, sym->ToExpr());
        MS_LOG(DEBUG) << "Init a input symbol: " << idx << ", " << j << " -> " << val->ToString();
        temp_map_[sym->ToExpr()] = val;
      }
    }
  }
}

bool TransformVisitor::Transform(Symbol *symbol) {
  try {
    MS_LOG_TRY_CATCH_SCOPE;
    // Assume that symbol is either tuple of shapes or just one shape
    if (symbol->tid() == ListSymbol::kTypeId) {
      for (auto &ilist_symbol : symbol->as<ListSymbol>()->symbols()) {
        SymbolVisitor::Visit(ilist_symbol.get());
      }
    } else {
      SymbolVisitor::Visit(symbol);
    }
  } catch (const std::exception &ex) {
    symbols_.clear();
    return false;
  }
  for (auto term : symbols_) {
    auto shape = std::dynamic_pointer_cast<ast::Shape>(term);
    shapes_.push_back(std::move(shape));
  }
  symbols_.clear();
  return true;
}

void TransformVisitor::Visit(InputSymbol *symbol) { MS_LOG(EXCEPTION) << "Visit Input symbol: " << symbol->ToString(); }

void TransformVisitor::Visit(ScalarSymbol *symbol) { MS_LOG(EXCEPTION) << "Unexpected ScalarSymbol"; }

void TransformVisitor::Visit(IntSymbol *symbol) {
  if (symbol->HasData()) {
    MS_LOG(DEBUG) << ">>> Visit IntSymbol: " << symbol->ToString();
    symbols_.push_back(std::make_shared<ast::IntImm>(symbol->value()));
    MS_LOG(DEBUG) << "<<< Visit IntSymbol: Push back a Int " << symbol->value();
  } else {
    auto ite = temp_map_.find(symbol->ToExpr());
    if (ite != temp_map_.end()) {
      // Already a var
      symbols_.push_back(ite->second);
      return;
    } else {
      MS_LOG(DEBUG) << ">>> Visit IntSymbol, a thunk" << symbol->ToExpr() << ":" << symbol->operation()->type_name();
      Visit(symbol->operation().get());
      auto smbl_p = symbols_.back();
      auto val_p = NewVal(smbl_p, symbol->ToExpr());
      symbols_.pop_back();
      MS_LOG_DEBUG << "<<< val symbol " << val_p->ToString() << " point to " << symbol->ToString()
                   << "---: " << smbl_p->ToString();
      temp_map_[symbol->ToExpr()] = val_p;
      symbols_.push_back(val_p);
    }
  }
}

void TransformVisitor::Visit(BoolSymbol *symbol) { MS_LOG(EXCEPTION) << "Unsupported BoolSymbol"; }

void TransformVisitor::Visit(FloatSymbol *symbol) { MS_LOG(EXCEPTION) << "Unsupported FloatSymbol"; }

void TransformVisitor::Visit(ListSymbol *symbol) {
  MS_LOG(DEBUG) << "Visit ListSymbol: " << symbol->ToString();
  auto shape = std::make_shared<ast::Shape>();
  auto sym_list = symbol->symbols();
  for (auto sym_p : sym_list) {
    SymbolVisitor::Visit(sym_p.get());
  }
  if (sym_list.size() > symbols_.size()) {
    MS_LOG(EXCEPTION) << "Unexpected error: shape size:" << sym_list.size() << " > symbols_ size: " << symbols_.size();
  }
  for (size_t i = symbols_.size() - sym_list.size(); i < symbols_.size(); ++i) {
    shape->smbls_.push_back(std::dynamic_pointer_cast<ast::SingleTerm>(symbols_[i]));
  }
  MS_LOG(DEBUG) << "<<< Visit IListSymbol: Push back a shape: " << shape->ToString();
  symbols_.resize(symbols_.size() - sym_list.size());
  symbols_.push_back(shape);
}

void TransformVisitor::Visit(IListSymbol *symbol) {
  MS_LOG(DEBUG) << ">>> Visit IListSymbol: " << symbol->ToString();
  auto shape = std::make_shared<ast::Shape>();
  auto sym_list = symbol->symbols();
  for (auto sym_p : sym_list) {
    SymbolVisitor::Visit(sym_p.get());
  }
  if (sym_list.size() > symbols_.size()) {
    MS_LOG(EXCEPTION) << "Unexpected error: shape size:" << sym_list.size() << " > symbols_ size: " << symbols_.size();
  }
  for (size_t i = symbols_.size() - sym_list.size(); i < symbols_.size(); ++i) {
    shape->smbls_.push_back(std::dynamic_pointer_cast<ast::SingleTerm>(symbols_[i]));
  }
  MS_LOG(DEBUG) << "<<< Visit IListSymbol: Push back a shape: " << shape->ToString();
  symbols_.resize(symbols_.size() - sym_list.size());
  symbols_.push_back(shape);
}

void TransformVisitor::Visit(ops::Operation *op) {
  switch (op->tid()) {
    case ops::ScalarAdd::kTypeId:
      EmitBinOp(ast::BinOpType::ScalarAdd, op);
      break;
    case ops::ScalarSub::kTypeId:
      EmitBinOp(ast::BinOpType::ScalarSub, op);
      break;
    case ops::ScalarMul::kTypeId:
      EmitBinOp(ast::BinOpType::ScalarMul, op);
      break;
    case ops::ScalarMin::kTypeId:
      EmitBinOp(ast::BinOpType::ScalarMin, op);
      break;
    case ops::ScalarMax::kTypeId:
      EmitBinOp(ast::BinOpType::ScalarMax, op);
      break;
    case ops::ScalarDiv::kTypeId:
      EmitBinOp(ast::BinOpType::ScalarDiv, op);
      break;
    default:
      MS_LOG(EXCEPTION) << "Unsupported operation: " << op->name();
      break;
  }
}

std::string TransformVisitor::SymbolExprPrint() {
  std::stringstream ss;
  ss << "\n";
  for (auto &[name, smbl] : temp_map_) {
    if (std::dynamic_pointer_cast<ast::Input>(symbols_table_[smbl->id_]) != nullptr) {
      ss << name << ": " << name << "\n";
    } else {
      ss << name << ": " << symbols_table_[smbl->id_]->ToString() << "\n";
    }
  }
  for (auto &sym : symbols_) {
    ss << sym->ToString() << "\n";
  }
  return ss.str();
}

// Basically, this function return a new VarPtr, with id = cunrrent size of symbols_table_,
// so it should be used just before pushing back a TermPtr into symbols_table_
ast::VarPtr TransformVisitor::NewVal(ast::TermPtr term, const std::string &name) {
  ast::VarPtr val_p = std::make_shared<ast::Var>(symbols_table_.size(), name);
  symbols_table_.push_back(term);
  return val_p;
}

}  // namespace mindspore::graphkernel::symbol
