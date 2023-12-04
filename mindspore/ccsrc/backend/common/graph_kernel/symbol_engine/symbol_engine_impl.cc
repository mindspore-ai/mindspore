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
#include "backend/common/graph_kernel/symbol_engine/operations/infershape_op.h"

namespace mindspore::graphkernel::symbol {
using ops::builders::DependOn;
namespace {
std::pair<FuncGraphPtr, size_t> GetFuncGraphFromCNode(const CNodePtr &cnode) {
  auto sub_fg = GetCNodeFuncGraph(cnode);
  size_t index = kIndex1;
  if (sub_fg == nullptr && IsPrimitiveCNode(cnode, prim::kPrimPartial)) {
    auto vnode = cnode->input(kIndex1)->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(vnode);
    sub_fg = vnode->value()->cast<FuncGraphPtr>();
    MS_EXCEPTION_IF_NULL(sub_fg);
    index = kIndex2;
  }
  return std::make_pair(sub_fg, index);
}
}  // namespace

void SymbolEngineImpl::BuildNodesSymbol(const AnfNodePtrList &nodes) {
  for (auto &node : nodes) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (auto fg_with_index = GetFuncGraphFromCNode(cnode); fg_with_index.first != nullptr) {
      // "call" or "Partial" node
      BuildSubgraph(cnode, fg_with_index.first, fg_with_index.second);
    } else {
      // theoretically, it's possible that both shape and value are required for a same node.
      auto depend_on = depend_status_map_[node];
      if (depend_on.shape) {
        BuildCNodeSymbol(cnode, false);
      }
      if (depend_on.value) {
        BuildCNodeSymbol(cnode, true);
      }
    }
  }
}

void SymbolEngineImpl::Build() {
  auto func_graph = func_graph_.lock();
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(DEBUG) << "Build " << ToString() << " with graph " << func_graph->ToString();
  visited_.clear();  // the visited is useless
  cache_.InitInputs(func_graph->parameters());
  emitter_ = std::make_unique<OperationEmitter>(&ops_);
  cnodes_ = TopoSort(func_graph->get_return(), SuccIncoming,
                     [](const AnfNodePtr &node) { return node->isa<CNode>() ? FOLLOW : EXCLUDE; });
  BuildNodesSymbol(cnodes_);
  Dump();
  emitter_->Clean();
}

void SymbolEngineImpl::BuildWithOuterInfo(const CNodePtr &cnode, const SymbolEngineImpl &main_engine) {
  MS_LOG(DEBUG) << "Build subgraph " << ToString() << " of " << cnode->fullname_with_scope();
  visited_.clear();  // the visited is useless
  auto func_graph = func_graph_.lock();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto param_num = func_graph->parameters().size();
  MS_EXCEPTION_IF_CHECK_FAIL(param_num + 1 == cnode->size(), "cnode and parameter size mismatch");
  cache_.InitInputs(func_graph->parameters());
  emitter_ = std::make_unique<OperationEmitter>(&ops_);
  ops::infershape::RealShape::ShapeHint shape_hint;
  shape_hint.cnode_inputs.resize(param_num);
  shape_hint.param_inputs.resize(param_num);
  for (size_t i = 0; i < param_num; i++) {
    auto &param = func_graph->parameters()[i];
    auto depend_on = depend_status_map_[param];
    auto inp_symbol = cache_.GetInput(param);
    if (depend_on.shape) {
      // refer to maingraph, if some inputs have same shape symbols, use same symbols in subgraph.
      shape_hint.input_index = i;
      shape_hint.cnode_inputs[i] = main_engine.cache_.GetShape(cnode->input(i + 1));
      auto smbl = emitter_->Emit(std::make_shared<ops::infershape::RealShape>(inp_symbol, &shape_hint));
      cache_.SetShape(param, smbl);
      shape_hint.param_inputs[i] = smbl;
    }
    if (depend_on.value) {
      // todo, optimize the same value symbol if need.
      cache_.SetValue(param, emitter_->RealValue(inp_symbol));
    }
  }
  cnodes_ = TopoSort(func_graph->get_return(), SuccIncoming,
                     [](const AnfNodePtr &node) { return node->isa<CNode>() ? FOLLOW : EXCLUDE; });
  BuildNodesSymbol(cnodes_);
  Dump();
}

void SymbolEngineImpl::PreBuild(bool depend_on_value) {
  auto func_graph = func_graph_.lock();
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(DEBUG) << "Prebuild " << ToString() << ". depend_on_value=" << std::boolalpha << depend_on_value;
  (void)DfsQueryDependStatus(func_graph->get_return(), depend_on_value);
}

void SymbolEngineImpl::DfsQueryDependStatus(const AnfNodePtr &node, bool depend_on_value) {
  // node can be visited twice for query value and shape.
  if (!visited_.insert({node, depend_on_value}).second) {
    return;
  }
  if (depend_on_value) {
    depend_status_map_[node].value = true;
    MS_LOG(DEBUG) << "The value of " << node->DebugString() << "(" << node->fullname_with_scope() << ") is required";
  } else {
    depend_status_map_[node].shape = true;
    MS_LOG(DEBUG) << "The shape of " << node->DebugString() << "(" << node->fullname_with_scope() << ") is required";
  }
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    return;
  }
  if (cnode->input(0)->isa<CNode>()) {
    // the Call node of control-flow.
    DfsQueryDependStatus(cnode->input(0), depend_on_value);
    return;
  }
  auto subfg_with_index = GetFuncGraphFromCNode(cnode);
  if (subfg_with_index.first != nullptr) {
    DfsSubgraphQueryDependStatus(cnode, depend_on_value, subfg_with_index.first, subfg_with_index.second);
    return;
  }
  auto *info = OperationBuilderRegistry::GetBuildInfo(AnfUtils::GetCNodeName(cnode));
  auto default_depend = depend_on_value ? DependOn::kValue : DependOn::kShape;
  if (info == nullptr) {
    for (size_t i = 1; i < cnode->size(); i++) {
      DfsQueryDependStatus(cnode->input(i), default_depend == DependOn::kValue);
    }
    return;
  }
  auto depends = info->GetDepends(cnode, depend_on_value);
  for (size_t i = 1; i < cnode->size(); i++) {
    auto current_depend = (i - 1 < depends.size() ? depends[i - 1] : default_depend);
    DfsQueryDependStatus(cnode->input(i), current_depend == DependOn::kValue);
  }
}

void SymbolEngineImpl::DfsSubgraphQueryDependStatus(const CNodePtr &cnode, bool depend_on_value,
                                                    const FuncGraphPtr &sub_fg, size_t begin_input_index) {
  SymbolEngineImpl *sub_engine = nullptr;
  if (multi_engine_) {
    // when the node is visited in ahead, use the exist
    if (visited_.count({cnode, !depend_on_value}) > 0) {
      auto symbol_engine_attr = sub_fg->get_attr(kAttrSymbolEngine);
      MS_EXCEPTION_IF_NULL(symbol_engine_attr);
      sub_engine = symbol_engine_attr->cast_ptr<SymbolEngineImpl>();
      MS_EXCEPTION_IF_NULL(sub_engine);
    } else {
      auto sub_engine_sptr = std::make_shared<SymbolEngineImpl>(sub_fg, true);
      sub_fg->set_attr(kAttrSymbolEngine, sub_engine_sptr);
      sub_engine = sub_engine_sptr.get();
    }
    sub_engine->PreBuild(depend_on_value);
  } else {
    // use the unique symbolengine.
    sub_engine = this;
    sub_fg->set_attr(kAttrSymbolEngine, this->cast<SymbolEnginePtr>());
    DfsQueryDependStatus(sub_fg->get_return(), depend_on_value);
  }

  // visit nodes in main graph according to subgraph's parameters
  for (auto &param : sub_fg->parameters()) {
    auto depend_on = sub_engine->depend_status_map_[param];
    if (depend_on.shape) {
      DfsQueryDependStatus(cnode->input(begin_input_index), false);
    }
    if (depend_on.value) {
      DfsQueryDependStatus(cnode->input(begin_input_index), true);
    }
    begin_input_index++;
  }
}

bool SymbolEngineImpl::Infer(const AbstractBasePtrList &inputs) {
  if (!support_infer_) {
    MS_LOG(WARNING) << "The " << ToString() << " does not support infer";
    return false;
  }
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

ListSymbolPtr SymbolEngineImpl::QuerySymbolicShape(const AnfNodePtr &node) const {
  auto s = cache_.GetShape(node);
  return s == nullptr ? nullptr : s->as_sptr<ListSymbol>();
}
SymbolPtr SymbolEngineImpl::QuerySymbolicValue(const AnfNodePtr &node) const { return cache_.GetValue(node); }

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
  auto output_arr = output->as<ListSymbol>();
  MS_EXCEPTION_IF_NULL(output_arr);
  ShapeArray ret;
  ret.reserve(output_arr->size());
  (void)std::transform(output_arr->symbols().begin(), output_arr->symbols().end(), std::back_inserter(ret),
                       [&output](const SymbolPtr &s) -> ShapeVector {
                         if (s->is<IListSymbol>()) {
                           return ToShape(s);
                         }
                         MS_LOG(DEBUG) << "QueryShape only support ShapeVector or ShapeArray, but got "
                                       << output->ToString();
                         return {};
                       });
  return ret;
}

ShapeArray SymbolEngineImpl::QueryValue(const AnfNodePtr &node) {
  // todo
  return ShapeArray();
}

std::vector<std::string> SymbolEngineImpl::QuerySymbolicShapeStr(const AnfNodePtr &node) {
  auto output = cache_.GetShape(node);
  if (output == nullptr) {
    output = ops::builders::OperationBuilder(emitter_.get(), &cache_, {}).RealShape(node);
    MS_EXCEPTION_IF_NULL(output);
  }
  auto shape_list = output->as<ListSymbol>();
  MS_EXCEPTION_IF_NULL(shape_list);
  if (shape_list->size() == 0) {
    return {"1"};
  }
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

void SymbolEngineImpl::BuildSubgraph(const CNodePtr &cnode, const FuncGraphPtr &sub_fg, size_t begin_input_index) {
  MS_EXCEPTION_IF_NULL(sub_fg);
  if (has_built_.count(sub_fg.get()) != 0) {
    // in while-block, the funcgraph is called recursively, only build symbolengine once.
    return;
  }
  (void)has_built_.insert(sub_fg.get());
  MS_LOG(DEBUG) << "Build subgraph " << sub_fg->ToString() << " of node " << cnode->fullname_with_scope();
  if (multi_engine_) {
    auto sub_engine_attr = sub_fg->get_attr(kAttrSymbolEngine);
    MS_EXCEPTION_IF_NULL(sub_engine_attr);
    auto sub_engine = sub_engine_attr->cast_ptr<SymbolEngineImpl>();
    MS_EXCEPTION_IF_NULL(sub_engine);
    sub_engine->BuildWithOuterInfo(cnode, *this);
    if (depend_status_map_[cnode].shape) {
      cache_.SetShape(cnode, sub_engine->cache_.GetShape(sub_fg->get_return()));
    }
    if (depend_status_map_[cnode].value) {
      cache_.SetValue(cnode, sub_engine->cache_.GetValue(sub_fg->get_return()));
    }
  } else {
    auto param_num = sub_fg->parameters().size();
    MS_EXCEPTION_IF_CHECK_FAIL(param_num + begin_input_index == cnode->size(), "cnode and parameter size mismatch");
    for (size_t i = 0; i < param_num; i++) {
      if (auto s = cache_.GetShape(cnode->input(i + begin_input_index)); s != nullptr) {
        cache_.SetShape(sub_fg->parameters()[i], s);
      }
      if (auto v = cache_.GetValue(cnode->input(i + begin_input_index)); v != nullptr) {
        cache_.SetValue(sub_fg->parameters()[i], v);
      }
    }
    auto inner_nodes = TopoSort(sub_fg->get_return(), SuccIncoming,
                                [](const AnfNodePtr &node) { return node->isa<CNode>() ? FOLLOW : EXCLUDE; });
    BuildNodesSymbol(inner_nodes);
    cache_.BindNode(cnode, sub_fg->get_return());
  }
}

void SymbolEngineImpl::BuildCNodeSymbol(const CNodePtr &cnode, bool infer_value) {
  auto name = cnode->input(0)->isa<CNode>() ? kControlFlowOperation : AnfUtils::GetCNodeName(cnode);
  auto builder = OperationBuilderRegistry::GetBuilder(name, emitter_.get(), &cache_);
  MS_EXCEPTION_IF_NULL(builder);
  auto abstract = (name == kReturnOpName ? cnode->input(1)->abstract() : cnode->abstract());
  MS_EXCEPTION_IF_NULL(abstract);
  if (infer_value) {
    MS_LOG(DEBUG) << "Build value for node " << cnode->fullname_with_scope() << ".   " << cnode->DebugString();
    SymbolPtr v;
    try {
      MS_LOG_TRY_CATCH_SCOPE;
      v = builder->BuildValue(cnode);
    } catch (std::exception &) {
      v = nullptr;
    }
    if (v == nullptr) {
      MS_LOG(DEBUG) << "Node " << cnode->fullname_with_scope() << " does not support BuildValue.";
      support_infer_ = false;
      v = emitter_->RealValue(InputSymbol::Make(abstract));
      MS_EXCEPTION_IF_NULL(v);
    }
    MS_LOG(DEBUG) << "Set value for node: " << cnode->fullname_with_scope() << ". symbol: " << v->ToString();
    cache_.SetValue(cnode, v);
  } else {
    MS_LOG(DEBUG) << "Build shape for node " << cnode->fullname_with_scope() << ".   " << cnode->DebugString();
    if (!abstract->GetShape()->IsDynamic()) {
      auto static_shape = emitter_->RealShape(InputSymbol::Make(abstract));
      MS_LOG(DEBUG) << "Node " << cnode->fullname_with_scope() << " is static shape: " << static_shape->ToString();
      cache_.SetShape(cnode, static_shape);
      return;
    }
    SymbolPtr s;
    try {
      MS_LOG_TRY_CATCH_SCOPE;
      s = builder->BuildShape(cnode);
    } catch (std::exception &) {
      s = nullptr;
    }
    if (s == nullptr) {
      support_infer_ = false;
      MS_LOG(DEBUG) << "Node " << cnode->fullname_with_scope() << " does not support BuildShape.";
      s = emitter_->RealShape(InputSymbol::Make(abstract));
      MS_EXCEPTION_IF_NULL(s);
    }
    MS_LOG(DEBUG) << "Set shape for node: " << cnode->fullname_with_scope() << ". symbol: " << s->ToString();
    cache_.SetShape(cnode, s);
  }
}

void SymbolEngineImpl::Dump() const {
  static const bool dump_symbol_engine = (common::GetEnv("MS_DEV_DUMP_SYMBOL") == "on");
  if (!dump_symbol_engine) {
    return;
  }
  MS_LOG(INFO) << "======= Begin dump graph of " << name_ << " =================";
  for (auto op : ops_) {
    MS_LOG(INFO) << op->output()->ToString() << " = " << op->ToString();
  }
  MS_LOG(INFO) << "======= Begin dump shapes of " << name_ << " ================";
  for (size_t i = 0; i < cnodes_.size(); i++) {
    DumpCNode(cnodes_[i], std::to_string(i));
  }
  MS_LOG(INFO) << "======= Finish dumping " << name_ << " ======================";
}

void SymbolEngineImpl::DumpCNode(const AnfNodePtr &node, const std::string &id) const {
  auto dump_symbol = [this](const AnfNodePtr &node) -> std::string {
    auto shape = cache_.GetShape(node);
    auto value = cache_.GetValue(node);
    if (shape == nullptr && value == nullptr) {
      return " symbol: none";
    }
    std::string out;
    if (shape != nullptr) {
      out = " shape-symbol: " + shape->ToString();
    }
    if (value != nullptr) {
      out += " value-symbol: " + value->ToString();
    }
    return out;
  };
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(INFO) << "Node " << id << ": " << cnode->fullname_with_scope();
  for (size_t j = 1; j < cnode->size(); j++) {
    MS_LOG(INFO) << "  input[" << j << "]:" << dump_symbol(cnode->input(j));
  }
  MS_LOG(INFO) << "  output:" << dump_symbol(cnode);
  if (multi_engine_) {
    return;
  }
  // dump subgraph
  auto sub_fg = GetCNodeFuncGraph(node);
  if (sub_fg == nullptr) {
    return;
  }
  auto inner_nodes = TopoSort(sub_fg->get_return(), SuccIncoming,
                              [](const AnfNodePtr &n) { return n->isa<CNode>() ? FOLLOW : EXCLUDE; });
  for (size_t j = 0; j < inner_nodes.size(); j++) {
    DumpCNode(inner_nodes[j], id + "_" + std::to_string(j));
  }
}
}  // namespace mindspore::graphkernel::symbol
