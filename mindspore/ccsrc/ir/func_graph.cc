/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "ir/func_graph.h"

#include <algorithm>
#include <sstream>
#include <utility>

#include "debug/trace.h"
#include "ir/manager.h"
#include "operator/ops.h"
#include "pybind_api/export_flags.h"
#include "utils/ordered_set.h"
#include "utils/convert_utils.h"

namespace mindspore {
/*
 * Methods of Graph
 */
FuncGraph::FuncGraph()
    : flags_(),
      transforms_(),
      parameter_default_value_(),
      seen_(0),
      parameters_(),
      has_vararg_(false),
      has_kwarg_(false),
      kwonlyargs_count_(0),
      hyper_param_count_(0),
      is_generated_(false),
      return_(nullptr),
      manager_(std::weak_ptr<FuncGraphManager>()) {
  debug_info_ = std::make_shared<GraphDebugInfo>();
}

AnfNodePtr FuncGraph::output() const {
  // If return value is set, return should have two inputs.
  if (return_ != nullptr && return_->inputs().size() == 2) {
    return return_->input(1);
  } else {
    // If not set yet, return nullptr.
    return nullptr;
  }
}

ParameterPtr FuncGraph::add_parameter() {
  FuncGraphPtr this_func_graph = shared_from_base<FuncGraph>();
  ParameterPtr p = std::make_shared<Parameter>(this_func_graph);
  add_parameter(p);
  return p;
}

void FuncGraph::add_parameter(const ParameterPtr &p) {
  if (manager_.lock()) {
    std::vector<AnfNodePtr> new_params = parameters_;
    new_params.push_back(p);
    manager_.lock()->SetParameters(shared_from_base<FuncGraph>(), new_params);
  } else {
    parameters_.push_back(p);
  }
}

ParameterPtr FuncGraph::AddWeightParameter(const std::string &name) {
  FuncGraphPtr this_graph = shared_from_base<FuncGraph>();
  ParameterPtr p = std::make_shared<Parameter>(this_graph);
  p->set_name(name);
  p->debug_info()->set_name(name);

  std::vector<AnfNodePtr> new_params = parameters_;
  // append parameter
  new_params.push_back(p);

  if (manager_.lock()) {
    manager_.lock()->SetParameters(shared_from_base<FuncGraph>(), new_params);
  } else {
    parameters_.push_back(p);
  }
  hyper_param_count_++;
  return p;
}

bool FuncGraph::has_flag(const std::string &flag) {
  if (flags_.count(flag)) {
    return flags_[flag];
  }
  return false;
}

CNodePtr FuncGraph::NewCNode(const std::vector<AnfNodePtr> &inputs) {
  CNodePtr cnode = std::make_shared<CNode>(inputs, shared_from_base<FuncGraph>());
  if (has_flag(GRAPH_FLAG_HAS_EFFECT)) {
    order_.push_back(cnode);
    MS_LOG(INFO) << "Graph: " << ToString() << ", push back " << cnode->DebugString() << " in order.";
  }
  return cnode;
}

CNodePtr FuncGraph::NewCNodeWithScope(const std::vector<AnfNodePtr> &inputs, const ScopePtr &scope) {
  CNodePtr app = NewCNode(inputs);
  app->set_scope(scope);
  return app;
}

void FuncGraph::DumpCNodeList() {
  MS_LOG(INFO) << "FuncGraph " << ToString() << " has following CNode in code order:";
  for (const auto &cnode : order_) {
    MS_LOG(INFO) << cnode->DebugString();
  }
}

std::string FuncGraph::ToString() const {
  return mindspore::label_manage::Label(const_cast<FuncGraph *>(this)->shared_from_base<FuncGraph>()->debug_info());
}

GraphDebugInfoPtr FuncGraph::debug_info() {
  MS_EXCEPTION_IF_NULL(this->debug_info_);
  if (this->debug_info_->get_graph() == nullptr) {
    this->debug_info_->set_graph(shared_from_base<FuncGraph>());
  }
  return this->debug_info_;
}

const AnfNodeSet &FuncGraph::nodes() { return nodes_; }

void FuncGraph::CopyNodes(const FuncGraphPtr &source) { nodes_ = source->nodes(); }

void FuncGraph::ClearNodes() { nodes_.clear(); }

void FuncGraph::AddNode(AnfNodePtr node) { nodes_.add(node); }

void FuncGraph::DropNode(AnfNodePtr node) {
  nodes_.erase(node);
  auto graph = node->func_graph();
  // Remove the node from order list.
  if (graph) {
    graph->EraseUnusedNodeInOrder(node);
  }
}

const AnfNodeCounterMap &FuncGraph::value_nodes() { return value_nodes_; }

void FuncGraph::CopyValueNodes(const FuncGraphPtr &source) {
  auto &others = source->value_nodes();
  for (auto it = others.begin(); it != others.end(); it++) {
    AddValueNode(it->first, it->second);
  }
}

void FuncGraph::ClearValueNodes() { value_nodes_.clear(); }

void FuncGraph::AddValueNode(AnfNodePtr node, int count) {
  if (value_nodes_.count(node) == 0) {
    value_nodes_[node] = count;
  } else {
    value_nodes_[node] += count;
  }
}

void FuncGraph::DropValueNode(AnfNodePtr node) {
  if (value_nodes_.count(node) != 0) {
    if (value_nodes_[node] == 1) {
      (void)value_nodes_.erase(node);
    } else {
      value_nodes_[node]--;
      if (value_nodes_[node] < 0) {
        MS_LOG(EXCEPTION) << "Count of ValueNode '" << node
                          << "' dec from 0. NodeInfo: " << trace::GetDebugInfo(debug_info());
      }
    }
  }
}

const AnfNodeCounterMap &FuncGraph::free_variables() { return free_variables_; }

void FuncGraph::CopyFreeVariables(const FuncGraphPtr &source) {
  auto &others = source->free_variables();
  for (auto it = others.begin(); it != others.end(); it++) {
    if (it->first->func_graph().get() != this) {
      (void)AddFreeVariable(it->first, it->second);
    }
  }
}

void FuncGraph::ClearFreeVariables() { free_variables_.clear(); }

bool FuncGraph::AddFreeVariable(AnfNodePtr node, int count) {
  if (free_variables_.count(node) == 0) {
    free_variables_[node] = count;
    return true;
  } else {
    free_variables_[node] += count;
    return false;
  }
}

bool FuncGraph::DropFreeVariable(AnfNodePtr node) {
  if (free_variables_.count(node) != 0) {
    if (free_variables_[node] == 1) {
      (void)free_variables_.erase(node);
      return true;
    } else {
      free_variables_[node]--;
      if (free_variables_[node] < 0) {
        MS_LOG(EXCEPTION) << "Count of free variable '" << node
                          << "' dec from 0. NodeInfo: " << trace::GetDebugInfo(debug_info());
      }
    }
  }
  return false;
}

const BaseRefCounterMap &FuncGraph::free_variables_total() {
  auto mng = manager_.lock();
  MS_EXCEPTION_IF_NULL(mng);
  auto &fv_total = mng->free_variables_total();
  return fv_total[shared_from_base<FuncGraph>()];
}

std::vector<AnfNodePtr> FuncGraph::free_variables_nodes() {
  std::vector<AnfNodePtr> nodes;
  const auto &fv_total = this->free_variables_total();
  for (auto &p : fv_total) {
    auto key = p.first;
    if (utils::isa<AnfNodePtr>(key)) {
      nodes.push_back(utils::cast<AnfNodePtr>(key));
    }
  }

  return nodes;
}

std::vector<FuncGraphPtr> FuncGraph::free_variables_func_graphs() {
  std::vector<FuncGraphPtr> func_graphs;
  const auto &fv_total = this->free_variables_total();
  for (auto &p : fv_total) {
    auto key = p.first;
    if (utils::isa<FuncGraphPtr>(key)) {
      func_graphs.push_back(utils::cast<FuncGraphPtr>(key));
    }
  }

  return func_graphs;
}

const FuncGraphCounterMap &FuncGraph::func_graphs_used() { return func_graphs_used_; }

void FuncGraph::CopyFuncGraphsUsed(const FuncGraphPtr &source) {
  auto &others = source->func_graphs_used();
  for (auto it = others.begin(); it != others.end(); it++) {
    (void)AddFuncGraphUsed(it->first, it->second);
  }
  func_graphs_used_.erase(source);
}

void FuncGraph::ClearFuncGraphsUsed() { func_graphs_used_.clear(); }

bool FuncGraph::AddFuncGraphUsed(FuncGraphPtr fg, int count) {
  if (func_graphs_used_.count(fg) == 0) {
    func_graphs_used_[fg] = count;
    return true;
  } else {
    func_graphs_used_[fg] += count;
    return false;
  }
}

bool FuncGraph::DropFuncGraphUsed(FuncGraphPtr fg) {
  if (func_graphs_used_.count(fg) != 0) {
    if (func_graphs_used_[fg] == 1) {
      (void)func_graphs_used_.erase(fg);
      return true;
    } else {
      func_graphs_used_[fg]--;
      if (func_graphs_used_[fg] < 0) {
        MS_LOG(EXCEPTION) << "Count of FuncGraph '" << fg
                          << "' dec from 0. NodeInfo: " << trace::GetDebugInfo(debug_info());
      }
    }
  }
  return false;
}

const FuncGraphSet &FuncGraph::func_graphs_used_total() {
  auto mng = manager_.lock();
  MS_EXCEPTION_IF_NULL(mng);
  auto &used = mng->func_graphs_used_total(shared_from_base<FuncGraph>());
  return used;
}

const CNodeIndexCounterMap &FuncGraph::func_graph_cnodes_index() { return func_graph_cnodes_index_; }

void FuncGraph::CopyFuncGraphCNodesIndex(const FuncGraphPtr &source) {
  auto &others = source->func_graph_cnodes_index();
  for (auto it = others.begin(); it != others.end(); it++) {
    // Ignore the user graph who may own itself.
    auto fg = it->first->first->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    if (fg.get() != this) {
      AddFuncGraphCNodeIndex(it->first, it->second);
    }
  }
}

void FuncGraph::ClearFuncGraphCNodesIndex() { func_graph_cnodes_index_.clear(); }

void FuncGraph::AddFuncGraphCNodeIndex(CNodeIndexPairPtr pair, int count) {
  if (func_graph_cnodes_index_.count(pair) == 0) {
    func_graph_cnodes_index_[pair] = count;
  } else {
    func_graph_cnodes_index_[pair] += count;
  }
}

void FuncGraph::DropFuncGraphCNodeIndex(CNodeIndexPairPtr pair) {
  if (func_graph_cnodes_index_.count(pair) != 0) {
    if (func_graph_cnodes_index_[pair] == 1) {
      (void)func_graph_cnodes_index_.erase(pair);
    } else {
      func_graph_cnodes_index_[pair]--;
      if (func_graph_cnodes_index_[pair] < 0) {
        MS_LOG(EXCEPTION) << "Count of CNode/Index '" << pair->first << "/" << pair->second
                          << "' dec from 0. NodeInfo: " << trace::GetDebugInfo(debug_info());
      }
    }
  }
}

const FuncGraphCounterMap &FuncGraph::j_func_graphs() { return j_func_graphs_; }

void FuncGraph::CopyJFuncGraphs(const FuncGraphPtr &source) {
  auto &others = source->j_func_graphs();
  for (auto it = others.begin(); it != others.end(); it++) {
    AddJFuncGraph(it->first, it->second);
  }
}

void FuncGraph::ClearJFuncGraphs() { j_func_graphs_.clear(); }

void FuncGraph::AddJFuncGraph(FuncGraphPtr fg, int count) {
  if (j_func_graphs_.count(fg) == 0) {
    j_func_graphs_[fg] = count;
  } else {
    j_func_graphs_[fg] += count;
  }
}

void FuncGraph::DropJFuncGraph(FuncGraphPtr fg) {
  if (j_func_graphs_.count(fg) != 0) {
    if (j_func_graphs_[fg] == 1) {
      (void)j_func_graphs_.erase(fg);
    } else {
      j_func_graphs_[fg]--;
      if (j_func_graphs_[fg] < 0) {
        MS_LOG(EXCEPTION) << "Count of J FuncGraph '" << fg
                          << "' dec from 0. NodeInfo: " << trace::GetDebugInfo(debug_info());
      }
    }
  }
}

FuncGraphPtr FuncGraph::parent() {
  // report the bug early.
  if (manager_.lock() == nullptr) {
    MS_LOG(EXCEPTION) << "BUG: no manager for this func graph: " << ToString()
                      << " NodeInfo: " << trace::GetDebugInfo(debug_info());
  }
  auto mng = manager_.lock();
  MS_EXCEPTION_IF_NULL(mng);
  return mng->parent(shared_from_base<FuncGraph>());
}

const FuncGraphSet &FuncGraph::children() {
  auto mng = manager_.lock();
  MS_EXCEPTION_IF_NULL(mng);
  return mng->children(shared_from_base<FuncGraph>());
}

const FuncGraphSet &FuncGraph::scope() {
  auto mng = manager_.lock();
  MS_EXCEPTION_IF_NULL(mng);
  return mng->scopes(shared_from_base<FuncGraph>());
}

bool FuncGraph::recursive() {
  auto mng = manager_.lock();
  MS_EXCEPTION_IF_NULL(mng);
  return mng->recursive(shared_from_base<FuncGraph>());
}

std::shared_ptr<std::list<FuncGraphPtr>> FuncGraph::recursive_graphs() {
  auto mng = manager_.lock();
  MS_EXCEPTION_IF_NULL(mng);
  return mng->recursive_graphs(shared_from_base<FuncGraph>());
}

AnfNodePtr FuncGraph::GetDefaultValueByName(const std::string &name) {
  auto itr = this->parameter_default_value_.find(name);
  if (itr == parameter_default_value_.end()) {
    return nullptr;
  }
  auto default_value = itr->second;
  if (default_value == nullptr) {
    MS_LOG(EXCEPTION) << "Graph parameter " << name << " not exist";
  }
  if (IsValueNode<NullObj>(default_value)) {
    return nullptr;
  }
  return default_value;
}

// set the default values
void FuncGraph::SetDefaultValues(const std::vector<std::string> &name_list, const std::vector<AnfNodePtr> &value_list) {
  auto all_is_null = std::all_of(value_list.begin(), value_list.end(),
                                 [](const AnfNodePtr &node) { return IsValueNode<NullObj>(node); });
  if (value_list.empty()) {
    all_is_null = true;
  }
  for (size_t i = 0; i < name_list.size(); ++i) {
    if (!all_is_null) {
      this->parameter_default_value_[name_list[i]] = value_list[i];
    }
  }
}

void FuncGraph::ClearDefaultValues() { parameter_default_value_.clear(); }

size_t FuncGraph::GetDefaultValueCount() {
  int null_count =
    std::count_if(parameter_default_value_.begin(), parameter_default_value_.end(),
                  [](const std::pair<std::string, AnfNodePtr> &pair) { return IsValueNode<NullObj>(pair.second); });
  return parameter_default_value_.size() - IntToSize(null_count);
}

AnfNodePtr FuncGraph::GetVariableArgParameter() {
  if (!has_vararg_) {
    return nullptr;
  }

  if (has_kwarg_) {
    if (parameters_.size() < hyper_param_count_ + 2) {
      MS_LOG(EXCEPTION) << "Length of parameters is " << parameters_.size() << ", hyper_param_count is "
                        << hyper_param_count_ << ", parameters is less than 2 + hyper_param_count";
    }
    return parameters_[parameters_.size() - hyper_param_count_ - 2];
  }

  if (parameters_.size() < hyper_param_count_ + 1) {
    MS_LOG(EXCEPTION) << "Length of parameters is " << parameters_.size() << ", hyper_param_count is "
                      << hyper_param_count_ << ", parameters is less than 1 + hyper_param_count";
  }
  return parameters_[parameters_.size() - hyper_param_count_ - 1];
}

std::string FuncGraph::GetVariableArgName() {
  if (!has_vararg_) {
    return "";
  }

  if (has_kwarg_) {
    if (parameters_.size() < hyper_param_count_ + 2) {
      MS_LOG(EXCEPTION) << "Length of parameters is " << parameters_.size() << ", hyper_param_count is "
                        << hyper_param_count_ << ", parameters is less than 2 + hyper_param_count";
    }
    return parameters_[parameters_.size() - hyper_param_count_ - 2]->cast<ParameterPtr>()->name();
  }

  if (parameters_.size() < hyper_param_count_ + 1) {
    MS_LOG(EXCEPTION) << "Length of parameters is " << parameters_.size() << ", hyper_param_count is "
                      << hyper_param_count_ << ", parameters is less than 1 + hyper_param_count";
  }
  return parameters_[parameters_.size() - hyper_param_count_ - 1]->cast<ParameterPtr>()->name();
}

AnfNodePtr FuncGraph::GetVariableKwargParameter() {
  if (has_kwarg_) {
    if (parameters_.size() < hyper_param_count_ + 1) {
      MS_LOG(EXCEPTION) << "Length of parameters is " << parameters_.size() << ", hyper_param_count is "
                        << hyper_param_count_ << ", parameters is less than 1 + hyper_param_count";
    }
    return parameters_[parameters_.size() - hyper_param_count_ - 1];
  }
  return nullptr;
}

std::string FuncGraph::GetVariableKwargName() {
  if (has_kwarg_) {
    if (parameters_.size() < hyper_param_count_ + 1) {
      MS_LOG(EXCEPTION) << "Length of parameters is " << parameters_.size() << ", hyper_param_count is "
                        << hyper_param_count_ << ", parameters is less than 1 + hyper_param_count";
    }
    return parameters_[parameters_.size() - hyper_param_count_ - 1]->cast<ParameterPtr>()->name();
  }
  return "";
}

int FuncGraph::GetPositionalArgsCount() const {
  int count = SizeToInt(parameters_.size());
  if (has_kwarg_) {
    count--;
  }
  if (has_vararg_) {
    count--;
  }
  return count - kwonlyargs_count_ - SizeToInt(hyper_param_count_);
}

AnfNodePtr FuncGraph::GetParameterByName(const std::string &name) {
  for (size_t i = 0; i < parameters_.size(); ++i) {
    MS_EXCEPTION_IF_NULL(parameters_[i]);
    auto param_cast = parameters_[i]->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_cast);
    if (param_cast->name() == name) {
      return parameters_[i];
    }
  }
  return nullptr;
}

void FuncGraph::add_parameter_obj_node(const AnfNodePtr &p) { paramter_obj_nodes_.push_back(p); }

std::list<CNodePtr> FuncGraph::GetOrderedCnodes() {
  if (has_flag(GRAPH_FLAG_HAS_EFFECT)) {
    MS_LOG(DEBUG) << "Return ordered cnodes.";
    return order_;
  } else {
    auto this_ptr = shared_from_base<FuncGraph>();
    auto BelongSameGraph = std::bind(IncludeBelongGraph, this_ptr, std::placeholders::_1);
    auto SuccDepends = std::bind(SuccIncludeFV, this_ptr, std::placeholders::_1);

    std::list<CNodePtr> cnodes;
    auto nodes = TopoSort(get_return(), SuccDepends, BelongSameGraph);
    for (const auto &node : nodes) {
      auto cnode = dyn_cast<CNode>(node);
      if (cnode) {
        cnodes.push_back(cnode);
      }
    }
    return cnodes;
  }
}

void FuncGraph::EraseUnusedNodeInOrder() {
  if (has_flag(GRAPH_FLAG_HAS_EFFECT)) {
    auto mng = manager_.lock();
    if (mng) {
      auto &all_nodes = nodes();
      // Erase unused cnode.
      for (auto it = order_.begin(); it != order_.end();) {
        if (all_nodes.count(*it)) {
          (void)it++;
        } else {
          MS_LOG(DEBUG) << "Remove node " << (*it)->ToString() << " in graph " << ToString() << " order.";
          it = order_.erase(it);
        }
      }
    }
  }
}

void FuncGraph::EraseUnusedNodeInOrder(const AnfNodePtr &n) {
  if (has_flag(GRAPH_FLAG_HAS_EFFECT) && n && n->isa<CNode>()) {
    order_.remove(n->cast<CNodePtr>());
    MS_LOG(DEBUG) << "Remove the node" << n->DebugString() << " from order list.";
  }
}

void FuncGraph::CheckOrder() {
  if (has_flag(GRAPH_FLAG_HAS_EFFECT)) {
    MS_LOG(DEBUG) << "Check graph " << ToString();
    for (auto it = order_.begin(); it != order_.end(); (void)it++) {
      for (const auto &input_node : (*it)->inputs()) {
        if (input_node && input_node->isa<CNode>() && input_node->func_graph() == shared_from_base<FuncGraph>()) {
          // Need to reorder the wrong order node.
          auto found = std::find(order_.begin(), it, input_node);
          if (found == it) {
            DumpCNodeList();
            MS_LOG(EXCEPTION) << "The cnode " << (*it)->DebugString() << " order in " << ToString()
                              << " doesn't obey the input dependency, "
                              << "as input " << input_node->DebugString() << " is not ahead of itself.";
          }
        }
      }
    }
    auto mng = manager_.lock();
    if (mng != nullptr) {
      const auto &all_nodes = nodes();
      if (all_nodes.size() != (order_.size() + parameters_.size())) {
        DumpCNodeList();
        MS_LOG(EXCEPTION) << "CNode order size " << order_.size() << " is not equal to managed node size "
                          << all_nodes.size() - parameters_.size() << ".";
      }
    }
    MS_LOG(DEBUG) << "Check order okay.";
  }
}

size_t NewFgSeenGeneration() {
  static size_t fg_seen_generation = 0;
  return ++fg_seen_generation;
}

const PrimitivePtr FuncGraphTransform::func_graph_prim_ = std::make_shared<Primitive>("FuncGraph");
const char kFuncGraphFlagUndetermined[] = "Undeterminate";
}  // namespace mindspore
