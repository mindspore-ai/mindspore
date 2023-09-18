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
#include "backend/common/graph_kernel/symbol_engine/symbol_engine_impl.h"
#include <algorithm>
#include <ostream>
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/graph_utils.h"
#include "ops/array_ops.h"
#include "ops/framework_ops.h"
#include "utils/check_convert_utils.h"
#include "utils/anf_utils.h"
#include "backend/common/graph_kernel/symbol_engine/utils.h"

namespace mindspore::graphkernel::symbol {
void SymbolEngineImpl::Build(const FuncGraphPtr &func_graph) {
  MS_LOG(DEBUG) << "Build " << ToString() << " with graph " << func_graph->ToString();
  cnodes_ = TopoSort(func_graph->output(), SuccIncoming,
                     [](const AnfNodePtr &node) { return node->isa<CNode>() ? FOLLOW : EXCLUDE; });
  cache_.InitInputs(func_graph->parameters());
  emitter_ = std::make_unique<OperationEmitter>(&ops_);
  for (auto &node : cnodes_) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    BuildCNodeSmbl(cnode);
  }
  Dump();
}

bool SymbolEngineImpl::Infer(const AbstractBasePtrList &inputs) {
  MS_LOG(DEBUG) << "Infer " << ToString() << " with inputs: " << inputs;
  if (!cache_.UpdateInputs(inputs)) {
    return false;
  }
  for (auto &op : ops_) {
    op->Run();
  }
  Dump();
  return true;
}

ShapeArray SymbolEngineImpl::QueryShape(const AnfNodePtr &node) {
  auto output = cache_.GetShape(node);
  if (output == nullptr) {
    auto value = cache_.GetValue(node);
    if (value != nullptr) {
      auto value_list = value->as<ListSymbol>();
      if (value_list != nullptr) {
        return value->HasData() ? ShapeArray{{SizeToLong(value_list->size())}} : ShapeArray{{-1}};
      }
      return {{}};
    }
    output = ops::builders::OperationBuilder(emitter_.get(), &cache_, {}).RealShape(node);
    MS_EXCEPTION_IF_NULL(output);
  }
  if (output->is<IListSymbol>()) {
    return ShapeArray{ToShape(output)};
  }
  MS_LOG(DEBUG) << "Query node: " << node->fullname_with_scope() << " output: " << output->ToString();
  auto output_arr = output->as<ListSymbol>();
  MS_EXCEPTION_IF_NULL(output_arr);
  ShapeArray ret;
  ret.reserve(output_arr->size());
  (void)std::transform(output_arr->symbols().begin(), output_arr->symbols().end(), std::back_inserter(ret),
                       [](auto &s) { return ToShape(s); });
  return ret;
}

ShapeArray SymbolEngineImpl::QueryValue(const AnfNodePtr &node) {
  // todo
  return ShapeArray();
}

std::vector<std::string> SymbolEngineImpl::QuerySymbolicShape(const AnfNodePtr &node) {
  auto output = cache_.GetShape(node);
  if (output == nullptr) {
    output = ops::builders::OperationBuilder(emitter_.get(), &cache_, {}).RealShape(node);
    MS_EXCEPTION_IF_NULL(output);
  }
  auto shape_list = output->as<ListSymbol>();
  MS_EXCEPTION_IF_NULL(shape_list);
  std::vector<std::string> res;
  res.reserve(shape_list->size());
  (void)std::transform(shape_list->symbols().cbegin(), shape_list->symbols().cend(), std::back_inserter(res),
                       [](const SymbolPtr &s) { return s->ToExpr(); });
  return res;
}

std::string QuerySymbolExprHelper(const SymbolPtr &s,
                                  const std::unordered_map<std::string, std::string> &symbol_expr_map) {
  if (s->is<ListSymbol>() || s->HasData()) {
    return s->ToExpr();
  }
  if (s->operation()->name() == "RealShape" || s->operation()->name() == "RealValue") {
    return s->ToExpr();
  }
  if (symbol_expr_map.find(s->ToExpr()) != symbol_expr_map.end()) {
    return s->ToExpr();
  }
  auto operation = s->operation();
  MS_EXCEPTION_IF_NULL(operation);
  std::ostringstream oss;
  oss << operation->name() << "(";
  bool first = true;
  for (auto &input : operation->inputs()) {
    if (first == true) {
      first = false;
    } else {
      oss << ", ";
    }
    oss << QuerySymbolExprHelper(input, symbol_expr_map);
  }
  oss << ")";
  return oss.str();
}

void SymbolEngineImpl::QuerySymbolExpr(const AnfNodePtr &node,
                                       std::unordered_map<std::string, std::string> *symbol_expr_map) {
  auto shape = cache_.GetShape(node);
  if (shape == nullptr) {
    return;
  }
  auto shape_list = shape->as<ListSymbol>();
  MS_EXCEPTION_IF_NULL(shape_list);
  for (const auto &symbol : shape_list->symbols()) {
    auto name = symbol->ToExpr();
    if (name[0] == 's' && symbol_expr_map->find(name) == symbol_expr_map->end()) {
      auto expr = QuerySymbolExprHelper(symbol, *symbol_expr_map);
      (*symbol_expr_map)[name] = expr;
    }
  }
}

void SymbolEngineImpl::BuildFuncSmbl(const CNodePtr &cnode, bool infer_value) {
  auto sub_fg_v = GetCNodePrimitive(cnode)->GetAttr(kAttrFuncGraph);
  MS_EXCEPTION_IF_NULL(sub_fg_v);
  auto sub_fg = sub_fg_v->cast<FuncGraphPtr>();
  MS_EXCEPTION_IF_NULL(sub_fg);
  for (size_t i = 0; i < sub_fg->parameters().size(); i++) {
    cache_.BindNode(sub_fg->parameters()[i], cnode->input(i + 1));
  }
  auto todo = TopoSort(sub_fg->output(), SuccIncoming,
                       [](const AnfNodePtr &node) { return node->isa<CNode>() ? FOLLOW : EXCLUDE; });
  for (auto &node : todo) {
    BuildCNodeSmbl(node->cast<CNodePtr>(), infer_value);
  }
  cache_.BindNode(cnode, sub_fg->output());
}

void SymbolEngineImpl::BuildCNodeSmbl(const CNodePtr &cnode, bool infer_value) {
  auto name = AnfUtils::GetCNodeName(cnode);
  if (name == prim::kPrimShapeCalc->name()) {
    return BuildFuncSmbl(cnode, true);
  }
  auto builder = OperationBuilderRegistry::GetBuilder(name, emitter_.get(), &cache_);
  if (builder == nullptr) {
    MS_LOG(EXCEPTION) << "Node " << cnode->DebugString() << " is not supported now.";
  }
  if (infer_value) {
    MS_LOG(DEBUG) << "Build value for node " << cnode->DebugString();
    auto v = builder->BuildValue(cnode);
    if (v == nullptr) {
      MS_LOG(EXCEPTION) << "Node " << cnode->DebugString() << " does not support BuildValue.";
    }
    cache_.SetValue(cnode, v);
  } else {
    MS_LOG(DEBUG) << "Build shape for node " << cnode->DebugString();
    auto s = builder->BuildShape(cnode);
    if (s == nullptr) {
      MS_LOG(EXCEPTION) << "Node " << cnode->DebugString() << " does not support BuildShape.";
    }
    cache_.SetShape(cnode, s);
  }
}

void SymbolEngineImpl::Dump() {
  static const bool dump_symbol_engine = (common::GetEnv("MS_DEV_DUMP_SYMBOL") == "on");
  if (!dump_symbol_engine) {
    return;
  }
  MS_LOG(INFO) << "======= Dump Graph =========================";
  for (auto op : ops_) {
    MS_LOG(INFO) << op->output()->ToString() << " = " << op->ToString();
  }
  MS_LOG(INFO) << "======= Dump Shapes ========================";
  auto dump_symbol = [this](const AnfNodePtr &node) -> std::string {
    auto output = cache_.GetShape(node);
    if (output == nullptr) {
      auto value = cache_.GetValue(node);
      if (value == nullptr) {
        return "none";
      }
      return value->ToString();
    }
    return output->ToString();
  };
  for (auto &node : cnodes_) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    MS_LOG(INFO) << "Node " << cnode->fullname_with_scope();
    MS_LOG(INFO) << "  inputs shape:";
    for (size_t i = 1; i < cnode->size(); i++) {
      MS_LOG(INFO) << "    " << i << ": " << QueryShape(cnode->input(i)) << ". symbol:" << dump_symbol(cnode->input(i));
    }
    MS_LOG(INFO) << "  output shape: " << QueryShape(cnode) << ". symbol:" << dump_symbol(cnode);
  }
  MS_LOG(INFO) << "============================================";
}
}  // namespace mindspore::graphkernel::symbol
