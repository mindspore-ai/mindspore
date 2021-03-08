/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "utils/trace_base.h"
#include "ir/manager.h"
#include "utils/flags.h"
#include "utils/ordered_set.h"
#include "utils/convert_utils_base.h"
#include "abstract/abstract_function.h"

namespace mindspore {
/*
 * Methods of Graph
 */
FuncGraph::FuncGraph()
    : attrs_(),
      transforms_(),
      parameter_default_value_(),
      seen_(0),
      parameters_(),
      has_vararg_(false),
      has_kwarg_(false),
      kwonlyargs_count_(0),
      hyper_param_count_(0),
      is_generated_(false),
      is_bprop_(false),
      return_(nullptr),
      manager_(std::weak_ptr<FuncGraphManager>()),
      stub_(false),
      stage_(-1) {
  debug_info_ = std::make_shared<GraphDebugInfo>();
  switch_layer_input_ = std::make_shared<bool>(false);
}

abstract::AbstractBasePtr FuncGraph::ToAbstract() {
  auto temp_context = abstract::AnalysisContext::DummyContext();
  return std::make_shared<abstract::FuncGraphAbstractClosure>(shared_from_base<FuncGraph>(), temp_context);
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

const std::vector<AnfNodePtr> FuncGraph::get_inputs() const {
  std::vector<AnfNodePtr> input_params;
  for (auto const &node : parameters_) {
    MS_EXCEPTION_IF_NULL(node);
    auto parameter = dyn_cast<Parameter>(node);
    if (!parameter->has_default()) {
      input_params.push_back(parameter);
    }
  }
  return input_params;
}

ParameterPtr FuncGraph::add_parameter() {
  FuncGraphPtr this_func_graph = shared_from_base<FuncGraph>();
  ParameterPtr p = std::make_shared<Parameter>(this_func_graph);
  add_parameter(p);
  return p;
}

void FuncGraph::add_parameter(const ParameterPtr &p) {
  if (manager_.lock()) {
    manager_.lock()->AddParameter(shared_from_base<FuncGraph>(), p);
  } else {
    parameters_.push_back(p);
  }
}

ParameterPtr FuncGraph::AddWeightParameter(const std::string &name) {
  FuncGraphPtr this_graph = shared_from_base<FuncGraph>();
  ParameterPtr p = std::make_shared<Parameter>(this_graph);
  p->set_name(name);
  p->debug_info()->set_name(name);

  if (manager_.lock()) {
    manager_.lock()->AddParameter(shared_from_base<FuncGraph>(), p);
  } else {
    parameters_.push_back(p);
  }
  hyper_param_count_++;
  return p;
}

bool FuncGraph::has_flag(const std::string &key) {
  auto iter = attrs_.find(key);
  if (iter != attrs_.cend()) {
    if (iter->second->isa<BoolImm>()) {
      return GetValue<bool>(iter->second);
    }
    MS_LOG(WARNING) << "key " << key << " is not a flag, please use has_attr function.";
  }
  return false;
}

bool FuncGraph::has_attr(const std::string &key) {
  auto iter = attrs_.find(key);
  return !(iter == attrs_.cend());
}

ValuePtr FuncGraph::get_attr(const std::string &key) {
  auto iter = attrs_.find(key);
  return iter == attrs_.cend() ? nullptr : iter->second;
}

CNodePtr FuncGraph::NewCNode(const std::vector<AnfNodePtr> &inputs) {
  return std::make_shared<CNode>(inputs, shared_from_base<FuncGraph>());
}

CNodePtr FuncGraph::NewCNodeInOrder(const std::vector<AnfNodePtr> &inputs) {
  CNodePtr cnode = NewCNode(inputs);
  order_.push_back(cnode);
  return cnode;
}

CNodePtr FuncGraph::NewCNodeInFront(const std::vector<AnfNodePtr> &inputs) {
  CNodePtr cnode = NewCNode(inputs);
  order_.push_front(cnode);
  return cnode;
}

CNodePtr FuncGraph::NewCNodeBefore(const AnfNodePtr &position, const std::vector<AnfNodePtr> &inputs) {
  CNodePtr cnode = NewCNode(inputs);
  CNodePtr pos_cnode = dyn_cast<CNode>(position);
  auto iter = order_.find(pos_cnode);
  order_.insert(iter, cnode);
  return cnode;
}

CNodePtr FuncGraph::NewCNodeAfter(const AnfNodePtr &position, const std::vector<AnfNodePtr> &inputs) {
  CNodePtr cnode = NewCNode(inputs);
  CNodePtr pos_cnode = dyn_cast<CNode>(position);
  auto iter = order_.find(pos_cnode);
  if (iter == order_.end()) {
    order_.push_front(cnode);
  } else {
    order_.insert(std::next(iter), cnode);
  }
  return cnode;
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

const std::unordered_map<AnfNodePtr, int> &FuncGraph::j_value_nodes() { return j_value_nodes_; }

void FuncGraph::CopyJValueNodes(const FuncGraphPtr &source) {
  auto &others = source->j_value_nodes();
  for (const auto &other : others) {
    AddJValueNode(other.first, other.second);
  }
}

void FuncGraph::ClearJValueNodes() { j_value_nodes_.clear(); }

void FuncGraph::AddJValueNode(const AnfNodePtr &value_node, int count) {
  if (j_value_nodes_.count(value_node) == 0) {
    j_value_nodes_[value_node] = count;
  } else {
    j_value_nodes_[value_node] += count;
  }
}

void FuncGraph::DropJValueNode(const AnfNodePtr &value_node) {
  if (j_value_nodes_.count(value_node) != 0) {
    if (j_value_nodes_[value_node] == 1) {
      (void)j_value_nodes_.erase(value_node);
    } else {
      j_value_nodes_[value_node]--;
      if (j_value_nodes_[value_node] < 0) {
        MS_LOG(EXCEPTION) << "Count of J ValueNode '" << value_node->DebugString()
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

void FuncGraph::ClearAllManagerInfo() {
  ClearNodes();
  ClearValueNodes();
  ClearFuncGraphCNodesIndex();
  ClearFreeVariables();
  ClearFuncGraphsUsed();
  ClearJValueNodes();
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
  if (IsValueNode<Null>(default_value)) {
    return nullptr;
  }
  return default_value;
}

// set the default values
void FuncGraph::SetDefaultValues(const std::vector<std::string> &name_list, const std::vector<AnfNodePtr> &value_list) {
  auto all_is_null =
    std::all_of(value_list.begin(), value_list.end(), [](const AnfNodePtr &node) { return IsValueNode<Null>(node); });
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
                  [](const std::pair<std::string, AnfNodePtr> &pair) { return IsValueNode<Null>(pair.second); });
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

std::list<CNodePtr> FuncGraph::GetOrderedCnodes() {
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

void FuncGraph::EraseUnusedNodeInOrder() {
  auto mng = manager_.lock();
  if (mng) {
    auto &all_nodes = nodes();
    // Erase unused cnode.
    for (auto it = order_.begin(); it != order_.end();) {
      if (!all_nodes.contains(*it)) {
        MS_LOG(DEBUG) << "Remove node: " << (*it)->ToString() << " in graph " << ToString() << " order.";
        it = order_.erase(it);
        continue;
      }
      (void)it++;
    }
  }
}

void FuncGraph::EraseUnusedNodeInOrder(const AnfNodePtr &node) {
  if (node) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode) {
      order_.erase(cnode);
      MS_LOG(DEBUG) << "Remove node: " << node->DebugString() << " from order list.";
    }
  }
}

// Maintain cnode order list when a cnode is replaced by a new one.
void FuncGraph::ReplaceInOrder(const AnfNodePtr &old_node, const AnfNodePtr &new_node) {
  MS_EXCEPTION_IF_NULL(old_node);
  MS_EXCEPTION_IF_NULL(new_node);
  if (order_.empty()) {
    // Skip if order list is empty.
    return;
  }
  auto old_cnode = old_node->cast<CNodePtr>();
  if (old_cnode == nullptr) {
    // Skip if old node is not cnode, since order list contains cnode only.
    return;
  }
  // Search old node in order list.
  auto iter = order_.find(old_cnode);
  if (iter == order_.end()) {
    // Skip if old node not found in order list.
    return;
  }
  auto new_cnode = new_node->cast<CNodePtr>();
  if (new_cnode != nullptr) {
    // Insert new node just before the old node.
    order_.insert(iter, new_cnode);
  }
  // Remove old node from order list.
  // Unused children nodes can be cleared by EraseUnusedNodeInOrder().
  order_.erase(iter);
}

static std::vector<AnfNodePtr> MakeInputNodes(const PrimitivePtr &primitive, const std::vector<AnfNodePtr> &inputs) {
  std::vector<AnfNodePtr> input_node_list;
  input_node_list.reserve(inputs.size() + 1);
  input_node_list.emplace_back(std::make_shared<ValueNode>(primitive));
  input_node_list.insert(input_node_list.end(), inputs.begin(), inputs.end());
  return input_node_list;
}

CNodePtr FuncGraph::NewCNode(const PrimitivePtr &primitive, const std::vector<AnfNodePtr> &inputs) {
  auto input_node_list = MakeInputNodes(primitive, inputs);
  return NewCNode(input_node_list);
}

CNodePtr FuncGraph::NewCNodeInOrder(const PrimitivePtr &primitive, const std::vector<AnfNodePtr> &inputs) {
  auto input_node_list = MakeInputNodes(primitive, inputs);
  return NewCNodeInOrder(input_node_list);
}

ParameterPtr FuncGraph::add_weight(const tensor::MetaTensorPtr &meta_tensor) {
  auto parameter = add_parameter();
  parameter->set_default_param(MakeValue(meta_tensor));
  parameter->set_abstract(meta_tensor->ToAbstract());
  return parameter;
}

bool FuncGraph::ContainMultiTarget() const {
  auto graph_manager = manager();
  MS_EXCEPTION_IF_NULL(graph_manager);
  FuncGraphSet graphs = graph_manager->func_graphs();
  for (auto &g : graphs) {
    auto nodes = TopoSort(g->get_return());
    if (mindspore::ContainMultiTarget(nodes)) {
      return true;
    }
  }
  return false;
}

size_t NewFgSeenGeneration() {
  static size_t fg_seen_generation = 0;
  return ++fg_seen_generation;
}

const PrimitivePtr FuncGraphTransform::func_graph_prim_ = std::make_shared<Primitive>("FuncGraph");
}  // namespace mindspore
