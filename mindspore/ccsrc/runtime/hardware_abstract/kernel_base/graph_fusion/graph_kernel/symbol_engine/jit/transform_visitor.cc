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
#include "runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/symbol_engine/jit/transform_visitor.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include "abstract/dshape.h"
#include "ir/func_graph.h"
#include "mindspore/ops/infer/symbol_ops_impl/scalar_add.h"
#include "mindspore/ops/infer/symbol_ops_impl/scalar_sub.h"
#include "mindspore/ops/infer/symbol_ops_impl/scalar_mul.h"
#include "mindspore/ops/infer/symbol_ops_impl/scalar_div.h"
#include "mindspore/ops/infer/symbol_ops_impl/scalar_max.h"
#include "mindspore/ops/infer/symbol_ops_impl/scalar_min.h"

namespace mindspore::graphkernel::symshape {
void TransformVisitor::Init(const FuncGraphPtr &func_graph) {
  MS_LOG(DEBUG) << "TransformVisitor init with graph " << func_graph->ToString();
  auto record_symbol = [this](const SymbolPtr &symbol, size_t i, size_t j) {
    auto sym = symbol->as<ScalarSymbol>();
    if (!sym->HasData()) {
      auto input_sym = std::make_shared<ast::Input>(i, j);
      auto val = NewVal(input_sym, sym->ToRawString());
      MS_LOG(DEBUG) << "Init a input symbol: " << i << ", " << j << " -> " << val->ToString();
      temp_map_[sym->ToRawString()] = val;
    }
  };
  for (size_t idx = 0; idx < func_graph->parameters().size(); idx++) {
    SymbolPtr sym_shape = func_graph->parameters()[idx]->abstract()->GetSymbolicShape();
    if (sym_shape == nullptr) {
      sym_shape = func_graph->parameters()[idx]->abstract()->GetSymbolicValue();
    }
    MS_EXCEPTION_IF_NULL(sym_shape);
    auto shape_list = sym_shape->as_noexcept<ListSymbol>();
    if (shape_list == nullptr) {
      record_symbol(sym_shape, idx, 0);
    }
    for (size_t j = 0; j < shape_list->symbols().size(); ++j) {
      record_symbol(shape_list->symbols()[j], idx, j);
    }
  }
}

bool TransformVisitor::Transform(const FuncGraphPtr &func_graph) {
  try {
    MS_LOG_TRY_CATCH_SCOPE;
    auto out_symbol = func_graph->output()->abstract()->GetSymbolicShape();
    MS_EXCEPTION_IF_NULL(out_symbol);
    // Assume that symbol is either tuple of shapes or just one shape
    if (func_graph->output()->abstract()->GetShape()->isa<abstract::SequenceShape>()) {
      for (auto &tensor_symbol : out_symbol->symbols()) {
        Visit(tensor_symbol.get());
      }
    } else {
      Visit(out_symbol.get());
    }
  } catch (const std::exception &ex) {
    symbols_.clear();
    return false;
  }
  for (auto term : symbols_) {
    auto shape = std::dynamic_pointer_cast<ast::Shape>(term);
    if (shape == nullptr) {
      MS_LOG(DEBUG) << "null shape exists.";
      return false;
    }
    shapes_.push_back(std::move(shape));
  }
  symbols_.clear();
  return true;
}

void TransformVisitor::VisitImpl(IntSymbol *symbol) {
  if (symbol->HasData()) {
    MS_LOG(DEBUG) << ">>> Visit IntSymbol: " << symbol->ToString();
    symbols_.push_back(std::make_shared<ast::IntImm>(symbol->value()));
    MS_LOG(DEBUG) << "<<< Visit IntSymbol: Push back a Int " << symbol->value();
  } else {
    auto ite = temp_map_.find(symbol->ToRawString());
    if (ite != temp_map_.end()) {
      // Already a var
      symbols_.push_back(ite->second);
      return;
    } else {
      MS_LOG(DEBUG) << ">>> Visit IntSymbol, a thunk" << symbol->ToRawString() << ":"
                    << symbol->operation()->type_name();
      Visit(symbol->operation().get());
      auto smbl_p = symbols_.back();
      auto val_p = NewVal(smbl_p, symbol->ToRawString());
      symbols_.pop_back();
      MS_LOG(DEBUG) << "<<< val symbol " << val_p->ToString() << " point to " << symbol->ToString()
                    << "---: " << smbl_p->ToString();
      temp_map_[symbol->ToRawString()] = val_p;
      symbols_.push_back(val_p);
    }
  }
}

void TransformVisitor::VisitImpl(ListSymbol *symbol) {
  MS_LOG(DEBUG) << "Visit ListSymbol: " << symbol->ToString();
  auto shape = std::make_shared<ast::Shape>();
  auto sym_list = symbol->symbols();
  for (auto sym_p : sym_list) {
    Visit(sym_p.get());
  }
  if (sym_list.size() > symbols_.size()) {
    MS_LOG(EXCEPTION) << "Unexpected error: shape size:" << sym_list.size() << " > symbols_ size: " << symbols_.size();
  }
  for (size_t i = symbols_.size() - sym_list.size(); i < symbols_.size(); ++i) {
    shape->smbls_.push_back(std::dynamic_pointer_cast<ast::SingleTerm>(symbols_[i]));
  }
  MS_LOG(DEBUG) << "<<< Visit ListSymbol: Push back a shape: " << shape->ToString();
  symbols_.resize(symbols_.size() - sym_list.size());
  symbols_.push_back(shape);
}

void TransformVisitor::VisitImpl(Operation *op) {
  switch (op->tid()) {
    case mindspore::symshape::ops::ScalarAdd::kTypeId:
      EmitBinOp(ast::BinOpType::ScalarAdd, op);
      break;
    case mindspore::symshape::ops::ScalarSub::kTypeId:
      EmitBinOp(ast::BinOpType::ScalarSub, op);
      break;
    case mindspore::symshape::ops::ScalarMul::kTypeId:
      EmitBinOp(ast::BinOpType::ScalarMul, op);
      break;
    case mindspore::symshape::ops::ScalarMin::kTypeId:
      EmitBinOp(ast::BinOpType::ScalarMin, op);
      break;
    case mindspore::symshape::ops::ScalarMax::kTypeId:
      EmitBinOp(ast::BinOpType::ScalarMax, op);
      break;
    case mindspore::symshape::ops::ScalarDiv::kTypeId:
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
}  // namespace mindspore::graphkernel::symshape
