/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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
#include "include/common/symbol_engine/symbol_engine_impl.h"
#include <algorithm>
#include <ostream>
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/graph_utils.h"
#include "ops/array_ops.h"
#include "ops/framework_ops.h"
#include "mindspore/core/ops/symbol_ops_impl/switch.h"
#include "utils/check_convert_utils.h"
#include "utils/anf_utils.h"
#include "mindspore/core/symbolic_shape/utils.h"
#include "mindspore/core/symbolic_shape/operation_builder.h"

namespace mindspore {
namespace symshape {
AnfNodePtrList GetCNodesOfFuncGraph(const FuncGraphPtr &fg) {
  return TopoSort(fg->output(), SuccIncoming,
                  [](const AnfNodePtr &node) { return node->isa<CNode>() ? FOLLOW : EXCLUDE; });
}

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

SymbolEngineImplPtr SymbolEngineImpl::Build(const FuncGraphPtr &func_graph) {
  auto engine = std::make_shared<SymbolEngineImpl>(func_graph);
  func_graph->set_symbol_engine(engine);
  engine->PreBuild();
  engine->BuildImpl();
  return engine;
}

void SymbolEngineImpl::BuildNodesSymbol(const FuncGraphPtr &fg, const AnfNodePtrList &cnodes) {
  for (auto &node : cnodes) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (auto fg_with_index = GetFuncGraphFromCNode(cnode); fg_with_index.first != nullptr) {
      // "call" or "Partial" node
      BuildSubgraphImpl(cnode, fg_with_index.first, fg_with_index.second);
    } else {
      BuildCNodeSymbol(cnode);
    }
  }
  // the funcgraph can be empty or only return a ValueNode.
  if (!cnodes.empty()) {
    return;
  }
  auto node = fg->output();
  if (node->isa<ValueNode>()) {
    auto depend_status = depend_status_map_[node];
    CloneAbstractIfSymbolExists(node);
    auto node_abs = node->abstract();
    MS_EXCEPTION_IF_NULL(node_abs);
    if (depend_status.shape) {
      auto sym_shape = node_abs->GetShape()->BuildSymbolicShape();
      MS_LOG(DEBUG) << "Set shape for node: " << node->DebugString() << ". symbol: " << sym_shape->ToString();
      node_abs->SetSymbolicShape(sym_shape);
    }
    if (depend_status.value) {
      auto sym_value = BuildSymbolicValue(node_abs);
      MS_LOG(DEBUG) << "Set value for node: " << node->DebugString() << ". symbol: " << sym_value->ToString();
      node_abs->SetSymbolicValue(sym_value);
    }
  }
}

void SymbolEngineImpl::PreBuild() {
  auto func_graph = func_graph_.lock();
  MS_EXCEPTION_IF_NULL(func_graph);
  cnodes_ = GetCNodesOfFuncGraph(func_graph);
  (void)visited_graph_.insert(func_graph.get());
  PreBuildQueryDependStatus(cnodes_);
  visited_graph_.clear();
}

void SymbolEngineImpl::BuildImpl() {
  auto func_graph = func_graph_.lock();
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(DEBUG) << "Build " << ToString() << " with graph " << func_graph->ToString();
  emitter_ = std::make_unique<OperationEmitter>(&ops_);
  (void)visited_graph_.insert(func_graph.get());
  BuildNodesSymbol(func_graph, cnodes_);
  emitter_->Clean();
  visited_graph_.clear();
}

void SymbolEngineImpl::PreBuildQueryDependStatus(const AnfNodePtrList &cnodes) {
  for (auto iter = cnodes.rbegin(); iter != cnodes.rend(); ++iter) {
    auto cnode = (*iter)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto &depend_status = depend_status_map_[cnode];
    if (!depend_status.value && !depend_status.shape) {
      // build symbolic shape for the node even though it's not depended by any nodes.
      depend_status.shape = true;
    }
    MS_LOG(DEBUG) << "The depend status of " << cnode->DebugString() << "(" << cnode->fullname_with_scope()
                  << "): shape-depend=" << depend_status.shape << ", value-depend=" << depend_status.value;
    // the control-flow node.
    if (cnode->input(0)->isa<CNode>()) {
      depend_status_map_[cnode->input(0)] = depend_status;
      continue;
    }
    // the "call" node or Partial node.
    auto subfg_with_index = GetFuncGraphFromCNode(cnode);
    if (subfg_with_index.first != nullptr) {
      PreBuildQuerySubgraphDependStatus(cnode, subfg_with_index.first, subfg_with_index.second);
      continue;
    }
    // the normal CNode, get the depend status from operation builder info.
    auto *info = OperationBuilderInfoRegistry::GetBuildInfo(AnfUtils::GetCNodeName(cnode));
    if (info == nullptr) {
      continue;
    }
    auto prim = GetCNodePrimitive(cnode);
    auto set_prev_node_func = [this, &cnode, info](const PrimitivePtr &prim, bool depend_value) {
      auto depends = info->GetDepends(prim, depend_value);
      for (size_t i = 0; i + 1 < cnode->size(); i++) {
        DependOn input_depend;
        if (depends.empty()) {
          // if the depend status is not set in build_info, set the output depend status to inputs.
          input_depend = depend_value ? DependOn::kValue : DependOn::kShape;
        } else {
          // if the depend status is set in build_info, use the config status.
          // and if the size of config is less than input size, skip post inputs (depend nothing).
          // e.g. like BiasAdd, set "{kShape, kNone}" is equivalent to "{kShape}".
          if (i >= depends.size()) {
            break;
          }
          input_depend = depends[i];
        }
        if (input_depend == DependOn::kValue) {
          depend_status_map_[cnode->input(i + 1)].value = true;
        } else if (input_depend == DependOn::kShape) {
          depend_status_map_[cnode->input(i + 1)].shape = true;
        }
      }
    };
    if (depend_status.shape) {
      set_prev_node_func(prim, false);
    }
    if (depend_status.value) {
      set_prev_node_func(prim, true);
    }
  }
}

void SymbolEngineImpl::PreBuildQuerySubgraphDependStatus(const CNodePtr &cnode, const FuncGraphPtr &sub_fg,
                                                         size_t begin_input_index) {
  if (!visited_graph_.insert(sub_fg.get()).second) {
    return;
  }
  sub_fg->set_symbol_engine(shared_from_base<SymbolEngine>());
  depend_status_map_[sub_fg->output()] = depend_status_map_[cnode];
  PreBuildQueryDependStatus(GetCNodesOfFuncGraph(sub_fg));
  for (auto &param : sub_fg->parameters()) {
    auto &cnode_input_depend_status = depend_status_map_[cnode->input(begin_input_index++)];
    auto depend_status = depend_status_map_[param];
    if (depend_status.shape) {
      cnode_input_depend_status.shape = true;
    }
    if (depend_status.value) {
      cnode_input_depend_status.value = true;
    }
  }
}

bool SymbolEngineImpl::Infer(const AbstractBasePtrList &inputs) {
  if (!support_infer_) {
    MS_LOG(WARNING) << "The " << ToString() << " does not support infer";
    return false;
  }
  MS_LOG(DEBUG) << "Infer " << ToString() << " with inputs: " << inputs;
  auto fg = func_graph_.lock();
  MS_EXCEPTION_IF_NULL(fg);
  auto &params = fg->parameters();
  // There may be params like UpdateStates, which won't contribute to infer
  if (params.size() < inputs.size()) {
    MS_LOG(EXCEPTION) << "The parameter size should be equal to or larger than inputs size, but got " << params.size()
                      << " vs " << inputs.size();
  }
  for (size_t i = 0; i < inputs.size(); i++) {
    if (auto shape = params[i]->abstract()->GetSymbolicShape(); shape != nullptr) {
      auto cur_shape = inputs[i]->GetShape()->BuildSymbolicShape();
      MS_EXCEPTION_IF_NULL(cur_shape);
      MS_LOG(DEBUG) << "Update shape for input[" << i << "]: " << cur_shape->ToRawString();
      shape->Update(cur_shape);
    }
    if (auto value = params[i]->abstract()->GetSymbolicValue(); value != nullptr) {
      auto cur_value = BuildSymbolicValue(inputs[i]);
      MS_EXCEPTION_IF_NULL(cur_value);
      MS_LOG(DEBUG) << "Update value for input[" << i << "]: " << cur_value->ToRawString();
      value->Update(cur_value);
    }
  }
  for (auto &op : ops_) {
    op->Run();
  }
  return true;
}

BaseShapePtr SymbolEngineImpl::QueryShape(const AnfNodePtr &node) {
  auto abs = node->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  auto symbolic_shape = abs->GetSymbolicShape();
  if (symbolic_shape == nullptr) {
    return nullptr;
  }
  auto digital_shape = abs->GetShape();
  MS_EXCEPTION_IF_NULL(digital_shape);
  if (!symbolic_shape->HasData()) {
    return digital_shape;
  }
  if (digital_shape->isa<abstract::NoShape>()) {
    return digital_shape;
  }
  if (digital_shape->isa<abstract::TensorShape>()) {
    return std::make_shared<abstract::TensorShape>(ToShape(symbolic_shape));
  }
  abstract::BaseShapePtrList shape_arr;
  shape_arr.reserve(symbolic_shape->size());
  (void)std::transform(symbolic_shape->symbols().begin(), symbolic_shape->symbols().end(),
                       std::back_inserter(shape_arr), [](const SymbolPtr &s) {
                         auto shape = s->as<ListSymbol>();
                         MS_EXCEPTION_IF_NULL(shape);
                         return std::make_shared<abstract::TensorShape>(ToShape(shape));
                       });
  return std::make_shared<abstract::TupleShape>(std::move(shape_arr));
}

ValuePtr SymbolEngineImpl::QueryValue(const AnfNodePtr &node) {
  auto abs = node->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  auto symbolic_value = abs->GetSymbolicValue();
  auto digital_value = abs->GetValue();
  MS_EXCEPTION_IF_NULL(digital_value);
  if (symbolic_value == nullptr) {
    return digital_value;
  }
  if (!symbolic_value->HasData()) {
    MS_LOG(WARNING) << "symbolic value of node has no data: " << node->fullname_with_scope();
    return digital_value;
  }
  return SymbolToValue(symbolic_value.get());
}

bool SymbolEngineImpl::IsDependValue(const AnfNodePtr &node) {
  if (depend_status_map_.find(node) != depend_status_map_.end()) {
    return depend_status_map_[node].value;
  }
  return false;
}

bool SymbolEngineImpl::IsDependShape(const AnfNodePtr &node) {
  if (depend_status_map_.find(node) != depend_status_map_.end()) {
    return depend_status_map_[node].shape;
  }
  return false;
}
std::string SymbolEngineImpl::QuerySymbolExprHelper(
  const SymbolPtr &s, const std::unordered_map<std::string, std::string> &symbol_expr_map) {
  auto raw_string = s->ToRawString();
  if (s->is<ListSymbol>() || s->HasData()) {
    return raw_string;
  }
  if (symbol_expr_map.find(raw_string) != symbol_expr_map.end()) {
    return raw_string;
  }
  auto operation = s->operation();
  if (operation == nullptr) {
    return raw_string;
  }
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
  // todo, use SymbolVisitor to export symbol expr.
  auto symbolic_shape = node->abstract()->GetSymbolicShape();
  if (symbolic_shape == nullptr) {
    return;
  }
  for (const auto &symbol : symbolic_shape->symbols()) {
    auto name = symbol->ToRawString();
    if (name[0] == 's' && symbol_expr_map->find(name) == symbol_expr_map->end()) {
      auto expr = QuerySymbolExprHelper(symbol, *symbol_expr_map);
      (*symbol_expr_map)[name] = expr;
    }
  }
}

void SymbolEngineImpl::BuildSubgraphImpl(const CNodePtr &cnode, const FuncGraphPtr &sub_fg, size_t begin_input_index) {
  MS_EXCEPTION_IF_NULL(sub_fg);
  if (!visited_graph_.insert(sub_fg.get()).second) {
    // in while-block, the funcgraph is called recursively, only build symbolengine once.
    return;
  }
  MS_LOG(DEBUG) << "Build subgraph " << sub_fg->ToString() << " of node " << cnode->fullname_with_scope();
  auto param_num = sub_fg->parameters().size();
  MS_EXCEPTION_IF_CHECK_FAIL(param_num + begin_input_index == cnode->size(), "cnode and parameter size mismatch");
  for (size_t i = 0; i < param_num; i++) {
    CloneAbstractIfSymbolExists(sub_fg->parameters()[i]);
    auto param_abs = sub_fg->parameters()[i]->abstract();
    MS_EXCEPTION_IF_NULL(param_abs);
    auto input_abs = cnode->input(i + begin_input_index)->abstract();
    MS_EXCEPTION_IF_NULL(input_abs);
    param_abs->SetSymbolicShape(input_abs->GetSymbolicShape());
    param_abs->SetSymbolicValue(input_abs->GetSymbolicValue());
  }
  auto nodes = GetCNodesOfFuncGraph(sub_fg);
  BuildNodesSymbol(sub_fg, nodes);
  auto out_abs = sub_fg->output()->abstract();
  MS_EXCEPTION_IF_NULL(out_abs);
  CloneAbstractIfSymbolExists(cnode);
  cnode->abstract()->SetSymbolicShape(out_abs->GetSymbolicShape());
  cnode->abstract()->SetSymbolicValue(out_abs->GetSymbolicValue());
}

SymbolPtr SymbolEngineImpl::BuildCNodeSymbolicShape(OperationBuilder *builder, const PrimitivePtr &prim,
                                                    const AbstractBasePtrList &inputs, const AbstractBasePtr &abstract,
                                                    const CNodePtr &cnode) {
  auto digital_shape = abstract->GetShape();
  MS_EXCEPTION_IF_NULL(digital_shape);
  if (!digital_shape->IsDynamic()) {
    auto static_shape = digital_shape->BuildSymbolicShape();
    MS_LOG(DEBUG) << "Node " << cnode->fullname_with_scope() << " is static shape: " << digital_shape->ToString();
    return static_shape;
  }
  if (builder == nullptr) {
    support_infer_ = false;
    MS_LOG(DEBUG) << "Node " << cnode->fullname_with_scope() << " does not support BuildShape, builder not found.";
    return digital_shape->BuildSymbolicShape();
  }
  SymbolPtr s = nullptr;
  try {
    MS_LOG_TRY_CATCH_SCOPE;
    s = builder->BuildShape(prim, inputs, abstract);
  } catch (std::exception &e) {
    MS_LOG(DEBUG) << "Failed to build shape for " << cnode->fullname_with_scope() << ": " << e.what();
    s = nullptr;
  }
  if (s == nullptr) {
    support_infer_ = false;
    MS_LOG(DEBUG) << "Node " << cnode->fullname_with_scope() << " does not support BuildShape.";
    return digital_shape->BuildSymbolicShape();
  }
  return s;
}

SymbolPtr SymbolEngineImpl::BuildCNodeSymbolicValue(OperationBuilder *builder, const PrimitivePtr &prim,
                                                    const AbstractBasePtrList &inputs, const AbstractBasePtr &abstract,
                                                    const CNodePtr &cnode) {
  if (builder == nullptr) {
    support_infer_ = false;
    MS_LOG(DEBUG) << "Node " << cnode->fullname_with_scope() << " does not support BuildValue, builder not found.";
    return BuildSymbolicValue(abstract);
  }
  SymbolPtr s = nullptr;
  try {
    MS_LOG_TRY_CATCH_SCOPE;
    s = builder->BuildValue(prim, inputs, abstract);
  } catch (std::exception &e) {
    MS_LOG(DEBUG) << "Failed to build value for " << cnode->fullname_with_scope() << ": " << e.what();
    s = nullptr;
  }
  if (s == nullptr) {
    support_infer_ = false;
    MS_LOG(DEBUG) << "Node " << cnode->fullname_with_scope() << " does not support BuildValue.";
    return BuildSymbolicValue(abstract);
  }
  return s;
}

AbstractBasePtrList SymbolEngineImpl::ExtractInputsAbstract(const CNodePtr &cnode) {
  CNodePtr real_node = cnode;
  if (cnode->input(0)->isa<CNode>()) {
    real_node = cnode->input(0)->cast<CNodePtr>();
  }
  AbstractBasePtrList abs_list;
  abs_list.reserve(real_node->size());
  (void)std::transform(real_node->inputs().cbegin() + 1, real_node->inputs().cend(), std::back_inserter(abs_list),
                       [](const AnfNodePtr &node) {
                         MS_EXCEPTION_IF_NULL(node);
                         return node->abstract();
                       });
  return abs_list;
}

void SymbolEngineImpl::BuildCNodeSymbol(const CNodePtr &cnode) {
  PrimitivePtr prim = GetCNodePrimitive(cnode);
  if (prim == nullptr && cnode->input(0)->isa<CNode>()) {
    prim = std::make_shared<Primitive>(ops::kControlFlowJoin);
  }
  MS_EXCEPTION_IF_NULL(prim);
  auto inputs = ExtractInputsAbstract(cnode);
  auto builder = OperationBuilderInfoRegistry::GetBuilder(prim->name(), emitter_.get());
  CloneAbstractIfSymbolExists(cnode);
  auto abstract = cnode->abstract();
  MS_EXCEPTION_IF_NULL(abstract);

  // theoretically, it's possible that both shape and value are required for a same node.
  const auto &depend_status = depend_status_map_[cnode];
  if (depend_status.value) {
    MS_LOG(DEBUG) << "Build value for node " << cnode->fullname_with_scope() << ".   " << cnode->DebugString();
    auto sym_value = BuildCNodeSymbolicValue(builder.get(), prim, inputs, abstract, cnode);
    MS_LOG(DEBUG) << "Set value for node: " << cnode->fullname_with_scope() << ". symbol: " << sym_value->ToString();
    abstract->SetSymbolicValue(sym_value);
  }

  if (depend_status.shape) {
    MS_LOG(DEBUG) << "Build shape for node " << cnode->fullname_with_scope() << ".   " << cnode->DebugString();
    auto sym_shape = BuildCNodeSymbolicShape(builder.get(), prim, inputs, abstract, cnode);
    MS_EXCEPTION_IF_NULL(sym_shape);
    MS_LOG(DEBUG) << "Set shape for node: " << cnode->fullname_with_scope() << ". symbol: " << sym_shape->ToString();
    abstract->SetSymbolicShape(sym_shape->as_sptr<ListSymbol>());
  }
}

std::string SymbolEngineImpl::DumpText() const {
  std::ostringstream oss;
  oss << ToString() << " {\n";
  for (auto op : ops_) {
    oss << "  " << op->DumpText();
  }
  oss << "}\n";
  return oss.str();
}

void CloneAbstractIfSymbolExists(const AnfNodePtr &node) {
  auto old_abs = node->abstract();
  MS_EXCEPTION_IF_NULL(old_abs);
  if (old_abs->GetSymbolicShape() == nullptr && old_abs->GetSymbolicValue() == nullptr) {
    return;
  }
  try {
    MS_LOG_TRY_CATCH_SCOPE;
    auto new_abs = old_abs->Clone();
    new_abs->SetSymbolicShape(nullptr);
    new_abs->SetSymbolicValue(nullptr);
    node->set_abstract(new_abs);
  } catch (std::exception &e) {
    std::string sym_shape = old_abs->GetSymbolicShape() == nullptr ? "" : old_abs->GetSymbolicShape()->ToString();
    std::string sym_value = old_abs->GetSymbolicValue() == nullptr ? "" : old_abs->GetSymbolicValue()->ToString();
    MS_LOG(WARNING) << "For node " << node->DebugString() << ", the abstract has symbol (S:" << sym_shape
                    << ", V:" << sym_value << ") but cannot be cloned. abstract: " << old_abs->ToString()
                    << ", failed info:" << e.what();
    return;
  }
}
}  // namespace symshape
}  // namespace mindspore
