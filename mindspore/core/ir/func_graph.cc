/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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
#include "mindspore/core/ops/framework_ops.h"
#include "utils/trace_base.h"
#include "ir/manager.h"
#include "utils/ordered_set.h"
#include "utils/convert_utils_base.h"
#include "abstract/abstract_function.h"
#include "ir/func_graph_cloner.h"
#include "utils/phase.h"
#include "frontend/operator/composite/unpack_call.h"
#include "mindspore/core/ops/sequence_ops.h"

namespace mindspore {
/*
 * Methods of Graph
 */
FuncGraph::FuncGraph() : FuncGraph(std::make_shared<GraphDebugInfo>()) {}

FuncGraph::FuncGraph(GraphDebugInfoPtr &&debug_info)
    : attrs_(),
      transforms_(),
      parameter_default_value_(),
      seen_(0),
      parameters_(),
      has_vararg_(false),
      has_kwarg_(false),
      exist_multi_target_(false),
      kw_only_args_count_(0),
      fv_param_count_(0),
      is_generated_(false),
      manager_(),
      debug_info_(std::move(debug_info)),
      stub_(false),
      stage_(-1),
      segment_(1),
      phase_(PhaseManager::GetInstance().phase()) {}

FuncGraph::~FuncGraph() { subclass_destruct_flag_ = true; }

void FuncGraph::DoBreakLoop() {
  if (attached_mng_cnt() > 0) {
    MS_LOG(INFO) << "Current Graph is holding by FuncGraphManager, can't DoBreakLoop now.";
    return;
  }
  ClearOrderList();
  python_obj_ = nullptr;
  used_forward_nodes_.clear();
  func_graph_cache_.clear();
  parameters_.clear();
  parameter_obj_nodes_.clear();
  set_dropped(true);
}

abstract::AbstractBasePtr FuncGraph::ToAbstract() {
  auto temp_context = abstract::AnalysisContext::DummyContext();
  return std::make_shared<abstract::FuncGraphAbstractClosure>(shared_from_base<FuncGraph>(), temp_context);
}

AnfNodePtr FuncGraph::output() const {
  constexpr size_t return_input_num = 2;
  // If return value is set, return should have two inputs.
  if (return_node() != nullptr && return_node()->size() == return_input_num) {
    return return_node()->input(1);
  } else {
    // If not set yet, return nullptr.
    return nullptr;
  }
}

const AnfNodePtrList FuncGraph::get_inputs() const {
  AnfNodePtrList input_params;
  for (auto const &node : parameters_) {
    MS_EXCEPTION_IF_NULL(node);
    auto parameter = dyn_cast<Parameter>(node);
    MS_EXCEPTION_IF_NULL(parameter);
    if (!parameter->has_default()) {
      input_params.push_back(parameter);
    }
  }
  return input_params;
}

ParameterPtr FuncGraph::add_parameter() {
  FuncGraphPtr this_func_graph = shared_from_base<FuncGraph>();
  ParameterPtr param = std::make_shared<Parameter>(this_func_graph);
  add_parameter(param);
  return param;
}

ParameterPtr FuncGraph::add_parameter(NodeDebugInfoPtr &&debug_info) {
  FuncGraphPtr this_func_graph = shared_from_base<FuncGraph>();
  ParameterPtr param = std::make_shared<Parameter>(this_func_graph, std::move(debug_info));
  add_parameter(param);
  return param;
}

void FuncGraph::add_parameter(const ParameterPtr &param) {
  if (manager_.lock()) {
    manager_.lock()->AddParameter(shared_from_base<FuncGraph>(), param);
  } else {
    parameters_.push_back(param);
  }
}

ParameterPtr FuncGraph::InsertFrontParameter() {
  FuncGraphPtr this_func_graph = shared_from_base<FuncGraph>();
  ParameterPtr param = std::make_shared<Parameter>(this_func_graph);
  InsertFrontParameter(param);
  return param;
}

void FuncGraph::InsertFrontParameter(const ParameterPtr &param) {
  if (manager_.lock()) {
    manager_.lock()->InsertFrontParameter(shared_from_base<FuncGraph>(), param);
  } else {
    PrependParameter(param);
  }
}

ParameterPtr FuncGraph::AddFvParameter(const std::string &name, const ValuePtr &default_value) {
  FuncGraphPtr this_graph = shared_from_base<FuncGraph>();
  ParameterPtr param = std::make_shared<Parameter>(this_graph);
  param->set_name(name);
  MS_EXCEPTION_IF_NULL(param->debug_info());
  param->debug_info()->set_name(name);
  param->debug_info()->set_trace_info(nullptr);
  MS_EXCEPTION_IF_NULL(default_value);
  param->set_default_param(default_value);
  param->set_abstract(default_value->ToAbstract());
  if (manager_.lock()) {
    manager_.lock()->AddParameter(shared_from_base<FuncGraph>(), param);
  } else {
    parameters_.push_back(param);
  }
  ++fv_param_count_;
  return param;
}

bool FuncGraph::has_flag(const std::string &key) const {
  auto iter = attrs_.find(key);
  if (iter != attrs_.cend()) {
    MS_EXCEPTION_IF_NULL(iter->second);
    if (iter->second->isa<BoolImm>()) {
      return GetValue<bool>(iter->second);
    }
    MS_LOG(WARNING) << "key " << key << " is not a flag, please use has_attr function.";
  }
  return false;
}

bool FuncGraph::has_attr(const std::string &key) const {
  auto iter = attrs_.find(key);
  return !(iter == attrs_.cend());
}

ValuePtr FuncGraph::get_attr(const std::string &key) const {
  auto iter = attrs_.find(key);
  return iter == attrs_.cend() ? nullptr : iter->second;
}

CNodePtr FuncGraph::NewCNodeWeak(AnfNodeWeakPtrList &&weak_inputs) {
  return std::make_shared<CNode>(std::move(weak_inputs), shared_from_base<FuncGraph>());
}

CNodePtr FuncGraph::NewCNodeWeak(const AnfNodeWeakPtrList &weak_inputs) {
  return std::make_shared<CNode>(weak_inputs, shared_from_base<FuncGraph>());
}

CNodePtr FuncGraph::NewCNode(AnfNodePtrList &&inputs) {
  std::vector<AnfNodeWeakPtr> weak_inputs;
  weak_inputs.reserve(inputs.size());
  std::transform(inputs.cbegin(), inputs.cend(), std::back_inserter(weak_inputs),
                 [](const AnfNodePtr &node) { return AnfNodeWeakPtr(node); });
  return std::make_shared<CNode>(std::move(weak_inputs), shared_from_base<FuncGraph>());
}

CNodePtr FuncGraph::NewCNode(const AnfNodePtrList &inputs) {
  std::vector<AnfNodeWeakPtr> weak_inputs;
  weak_inputs.reserve(inputs.size());
  std::transform(inputs.cbegin(), inputs.cend(), std::back_inserter(weak_inputs),
                 [](const AnfNodePtr &node) { return AnfNodeWeakPtr(node); });
  return std::make_shared<CNode>(std::move(weak_inputs), shared_from_base<FuncGraph>());
}

CNodePtr FuncGraph::NewCNodeInOrderWeak(AnfNodeWeakPtrList &&weak_inputs) {
  CNodePtr cnode = NewCNodeWeak(std::move(weak_inputs));
  (void)order_.emplace_back(CNodeWeakPtr(cnode));
  return cnode;
}

CNodePtr FuncGraph::NewCNodeInOrderWeak(const AnfNodeWeakPtrList &weak_inputs) {
  CNodePtr cnode = NewCNodeWeak(weak_inputs);
  (void)order_.emplace_back(CNodeWeakPtr(cnode));
  return cnode;
}

CNodePtr FuncGraph::NewCNodeInOrder(AnfNodePtrList &&inputs) { return NewCNodeInOrder(inputs); }

CNodePtr FuncGraph::NewCNodeInOrder(const AnfNodePtrList &inputs) {
  std::vector<AnfNodeWeakPtr> weak_inputs;
  weak_inputs.reserve(inputs.size());
  std::transform(inputs.cbegin(), inputs.cend(), std::back_inserter(weak_inputs),
                 [](const AnfNodePtr &node) { return AnfNodeWeakPtr(node); });
  CNodePtr cnode = NewCNodeWeak(std::move(weak_inputs));
  (void)order_.emplace_back(CNodeWeakPtr(cnode));
  return cnode;
}

CNodePtr FuncGraph::NewCNodeInFront(const AnfNodePtrList &inputs) {
  std::vector<AnfNodeWeakPtr> weak_inputs;
  weak_inputs.reserve(inputs.size());
  std::transform(inputs.cbegin(), inputs.cend(), std::back_inserter(weak_inputs),
                 [](const AnfNodePtr &node) { return AnfNodeWeakPtr(node); });
  CNodePtr cnode = NewCNodeWeak(std::move(weak_inputs));
  (void)order_.emplace_front(CNodeWeakPtr(cnode));
  return cnode;
}

CNodePtr FuncGraph::NewCNodeBefore(const AnfNodePtr &position, const AnfNodePtrList &inputs) {
  std::vector<AnfNodeWeakPtr> weak_inputs;
  weak_inputs.reserve(inputs.size());
  std::transform(inputs.cbegin(), inputs.cend(), std::back_inserter(weak_inputs),
                 [](const AnfNodePtr &node) { return AnfNodeWeakPtr(node); });
  CNodePtr cnode = NewCNodeWeak(std::move(weak_inputs));
  CNodePtr pos_cnode = dyn_cast<CNode>(position);
  auto iter = std::find_if(order_.cbegin(), order_.cend(), [&pos_cnode](const CNodeWeakPtr &node) {
    return node.lock() != nullptr && node.lock() == pos_cnode;
  });
  (void)order_.insert(iter, CNodeWeakPtr(cnode));
  return cnode;
}

CNodePtr FuncGraph::NewCNodeAfter(const AnfNodePtr &position, const AnfNodePtrList &inputs) {
  std::vector<AnfNodeWeakPtr> weak_inputs;
  weak_inputs.reserve(inputs.size());
  std::transform(inputs.cbegin(), inputs.cend(), std::back_inserter(weak_inputs),
                 [](const AnfNodePtr &node) { return AnfNodeWeakPtr(node); });
  CNodePtr cnode = NewCNodeWeak(std::move(weak_inputs));
  CNodePtr pos_cnode = dyn_cast<CNode>(position);
  auto iter = std::find_if(order_.cbegin(), order_.cend(), [&pos_cnode](const CNodeWeakPtr &node) {
    return node.lock() != nullptr && node.lock() == pos_cnode;
  });
  if (iter == order_.cend()) {
    order_.push_front(CNodeWeakPtr(cnode));
  } else {
    (void)order_.insert(std::next(iter), CNodeWeakPtr(cnode));
  }
  return cnode;
}

const std::list<AnfNodePtr> &FuncGraph::own_nodes() const { return own_nodes_; }

void FuncGraph::AddOwnNode(const AnfNodePtr &node) { (void)own_nodes_.emplace_back(node); }

void FuncGraph::AddOwnNode(const AnfNodePtrList &nodes) {
  (void)own_nodes_.insert(own_nodes_.end(), nodes.cbegin(), nodes.cend());
}

void FuncGraph::AddOwnNode(const AnfNodeWeakPtrList &weak_nodes) {
  std::transform(weak_nodes.cbegin(), weak_nodes.cend(), std::back_inserter(own_nodes_),
                 [](const AnfNodeWeakPtr &weak_node) -> AnfNodePtr { return weak_node.lock(); });
}

void FuncGraph::RemoveOwnNode(const AnfNodePtr &node) {
  auto iter = std::find(own_nodes_.cbegin(), own_nodes_.cend(), node);
  if (iter != own_nodes_.cend()) {
    own_nodes_.erase(iter);
  }
}

void FuncGraph::ResetOwnNodes() { own_nodes_.clear(); }

void FuncGraph::DumpCNodeList() {
  MS_LOG(INFO) << "FuncGraph " << ToString() << " has following CNode in code order:";
  for (const auto &weak_cnode : order_) {
    const auto &cnode = weak_cnode.lock();
    if (cnode == nullptr) {
      continue;
    }
    MS_LOG(INFO) << cnode->DebugString();
  }
}

std::string FuncGraph::ToString() const {
  std::ostringstream buffer;
  auto debug_info = const_cast<FuncGraph *>(this)->debug_info();
  buffer << mindspore::trace::Label(debug_info);
  buffer << "_" << debug_info->get_id();
  return buffer.str();
}

GraphDebugInfoPtr FuncGraph::debug_info() {
  MS_EXCEPTION_IF_NULL(this->debug_info_);
  if (this->debug_info_->get_graph() == nullptr) {
    this->debug_info_->set_graph(shared_from_base<FuncGraph>());
  }
  return this->debug_info_;
}

const AnfNodeSet &FuncGraph::nodes() const { return nodes_; }

const AnfNodeSet &FuncGraph::switch_nodes() const { return switch_nodes_; }

void FuncGraph::CopyNodes(const FuncGraphPtr &source) {
  nodes_.update(source->nodes());
  switch_nodes_.update(source->switch_nodes());
}

void FuncGraph::ClearNodes() {
  nodes_.clear();
  switch_nodes_.clear();
}

void FuncGraph::AddNode(const AnfNodePtr &node) {
  nodes_.add(node);
  if (IsPrimitiveCNode(node, prim::kPrimSwitch)) {
    switch_nodes_.add(node);
  }
}

void FuncGraph::DropNode(const AnfNodePtr &node) {
  if (node == nullptr) {
    MS_LOG(ERROR) << "Node is nullptr";
    return;
  }
  (void)nodes_.erase(node);
  if (IsPrimitiveCNode(node, prim::kPrimSwitch)) {
    switch_nodes_.erase(node);
  }
  auto graph = node->func_graph();
  if (node->isa<Parameter>()) {
    (void)parameters_.erase(std::remove(parameters_.begin(), parameters_.end(), node), parameters_.end());
  }
  // Remove the node from order list.
  if (graph != nullptr) {
    graph->EraseUnusedNodeInOrder(node);
  }
}

const AnfNodeCounterMap &FuncGraph::value_nodes() const { return value_nodes_; }

void FuncGraph::CopyValueNodes(const FuncGraphPtr &source) {
  MS_EXCEPTION_IF_NULL(source);
  auto &others = source->value_nodes();
  for (auto it = others.begin(); it != others.end(); ++it) {
    AddValueNode(it->first, it->second);
  }
}

void FuncGraph::ClearValueNodes() { value_nodes_.clear(); }

void FuncGraph::AddValueNode(const AnfNodePtr &node, int count) {
  if (value_nodes_.count(node) == 0) {
    value_nodes_[node] = count;
  } else {
    value_nodes_[node] += count;
  }
}

void FuncGraph::DropValueNode(const AnfNodePtr &node) {
  if (value_nodes_.count(node) != 0) {
    if (value_nodes_[node] == 1) {
      (void)value_nodes_.erase(node);
    } else {
      value_nodes_[node]--;
      if (value_nodes_[node] < 0) {
        MS_LOG(INTERNAL_EXCEPTION) << "Count of ValueNode '" << node
                                   << "' dec from 0. NodeInfo: " << trace::GetDebugInfoStr(debug_info());
      }
    }
  }
}

const AnfNodeCounterMap &FuncGraph::free_variables() const { return free_variables_; }

void FuncGraph::CopyFreeVariables(const FuncGraphPtr &source) {
  MS_EXCEPTION_IF_NULL(source);
  auto &others = source->free_variables();
  for (auto it = others.begin(); it != others.end(); ++it) {
    const auto &free_var = it->first;
    MS_EXCEPTION_IF_NULL(free_var);
    if (free_var->func_graph().get() != this) {
      (void)AddFreeVariable(free_var, it->second);
    }
  }
}

void FuncGraph::ClearFreeVariables() { free_variables_.clear(); }

bool FuncGraph::AddFreeVariable(const AnfNodePtr &node, int count) {
  if (free_variables_.count(node) == 0) {
    free_variables_[node] = count;
    return true;
  } else {
    free_variables_[node] += count;
    return false;
  }
}

bool FuncGraph::DropFreeVariable(const AnfNodePtr &node) {
  if (free_variables_.count(node) != 0) {
    if (free_variables_[node] == 1) {
      (void)free_variables_.erase(node);
      return true;
    } else {
      free_variables_[node]--;
      if (free_variables_[node] < 0) {
        MS_LOG(INTERNAL_EXCEPTION) << "Count of free variable '" << node
                                   << "' dec from 0. NodeInfo: " << trace::GetDebugInfoStr(debug_info());
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

AnfNodePtrList FuncGraph::free_variables_nodes() {
  AnfNodePtrList nodes;
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

const FuncGraphCounterMap &FuncGraph::func_graphs_used() const { return func_graphs_used_; }

void FuncGraph::CopyFuncGraphsUsed(const FuncGraphPtr &source) {
  auto &others = source->func_graphs_used();
  for (auto it = others.begin(); it != others.end(); ++it) {
    (void)AddFuncGraphUsed(it->first, it->second);
  }
  (void)func_graphs_used_.erase(source);
}

void FuncGraph::ClearFuncGraphsUsed() { func_graphs_used_.clear(); }

bool FuncGraph::AddFuncGraphUsed(const FuncGraphPtr &fg, int count) {
  if (func_graphs_used_.count(fg) == 0) {
    func_graphs_used_[fg] = count;
    return true;
  } else {
    func_graphs_used_[fg] += count;
    return false;
  }
}

bool FuncGraph::DropFuncGraphUsed(const FuncGraphPtr &fg) {
  if (func_graphs_used_.count(fg) != 0) {
    if (func_graphs_used_[fg] == 1) {
      (void)func_graphs_used_.erase(fg);
      return true;
    } else {
      func_graphs_used_[fg]--;
      if (func_graphs_used_[fg] < 0) {
        MS_LOG(INTERNAL_EXCEPTION) << "Count of FuncGraph '" << fg
                                   << "' dec from 0. NodeInfo: " << trace::GetDebugInfoStr(debug_info());
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

const CNodeIndexCounterMap &FuncGraph::func_graph_cnodes_index() const { return func_graph_cnodes_index_; }

void FuncGraph::CopyFuncGraphCNodesIndex(const FuncGraphPtr &source) {
  MS_EXCEPTION_IF_NULL(source);
  auto &others = source->func_graph_cnodes_index();
  for (auto it = others.begin(); it != others.end(); ++it) {
    // Ignore the user graph who may own itself.
    MS_EXCEPTION_IF_NULL(it->first);
    MS_EXCEPTION_IF_NULL(it->first->first);
    auto fg = it->first->first->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    if (fg.get() != this) {
      AddFuncGraphCNodeIndex(it->first, it->second);
    }
  }
}

void FuncGraph::ClearFuncGraphCNodesIndex() { func_graph_cnodes_index_.clear(); }

void FuncGraph::AddFuncGraphCNodeIndex(const CNodeIndexPairPtr &pair, int count) {
  if (func_graph_cnodes_index_.count(pair) == 0) {
    func_graph_cnodes_index_[pair] = count;
  } else {
    func_graph_cnodes_index_[pair] += count;
  }
}

void FuncGraph::DropFuncGraphCNodeIndex(const CNodeIndexPairPtr &pair) {
  if (func_graph_cnodes_index_.count(pair) != 0) {
    if (func_graph_cnodes_index_[pair] == 1) {
      (void)func_graph_cnodes_index_.erase(pair);
    } else {
      func_graph_cnodes_index_[pair]--;
      if (func_graph_cnodes_index_[pair] < 0) {
        MS_LOG(INTERNAL_EXCEPTION) << "Count of CNode/Index '" << pair->first << "/" << pair->second
                                   << "' dec from 0. NodeInfo: " << trace::GetDebugInfoStr(debug_info());
      }
    }
  }
}

const mindspore::HashMap<AnfNodePtr, int> &FuncGraph::meta_fg_prim_value_nodes() const {
  return meta_fg_prim_value_nodes_;
}

void FuncGraph::CopyMetaFgPrimValueNodes(const FuncGraphPtr &source) {
  MS_EXCEPTION_IF_NULL(source);
  auto &others = source->meta_fg_prim_value_nodes();
  for (const auto &other : others) {
    AddMetaFgPrimValueNode(other.first, other.second);
  }
}

void FuncGraph::ClearMetaFgPrimValueNodes() { meta_fg_prim_value_nodes_.clear(); }

void FuncGraph::AddMetaFgPrimValueNode(const AnfNodePtr &value_node, int count) {
  if (meta_fg_prim_value_nodes_.count(value_node) == 0) {
    meta_fg_prim_value_nodes_[value_node] = count;
  } else {
    meta_fg_prim_value_nodes_[value_node] += count;
  }
}

void FuncGraph::DropMetaFgPrimValueNode(const AnfNodePtr &value_node) {
  if (meta_fg_prim_value_nodes_.count(value_node) != 0) {
    if (meta_fg_prim_value_nodes_[value_node] == 1) {
      (void)meta_fg_prim_value_nodes_.erase(value_node);
    } else {
      meta_fg_prim_value_nodes_[value_node]--;
      if (meta_fg_prim_value_nodes_[value_node] < 0) {
        MS_LOG(INTERNAL_EXCEPTION) << "Count of MetaFgPrim ValueNode '" << value_node->DebugString()
                                   << "' dec from 0. NodeInfo: " << trace::GetDebugInfoStr(debug_info());
      }
    }
  }
}

FuncGraphPtr FuncGraph::parent() {
  // report the bug early.
  if (manager_.lock() == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "BUG: no manager for this func graph: " << ToString()
                               << " NodeInfo: " << trace::GetDebugInfoStr(debug_info());
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

void FuncGraph::ClearAllResource() {
  ClearNodes();
  ClearValueNodes();
  ClearFuncGraphCNodesIndex();
  ClearFreeVariables();
  ClearFuncGraphsUsed();
  ClearMetaFgPrimValueNodes();
}

AnfNodePtr FuncGraph::GetDefaultValueByName(const std::string &name) {
  auto itr = this->parameter_default_value_.find(name);
  if (itr == parameter_default_value_.end()) {
    return nullptr;
  }
  auto default_value = itr->second;
  if (default_value == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Graph parameter " << name << " not exist";
  }
  if (IsValueNode<Null>(default_value)) {
    return nullptr;
  }
  return default_value;
}

// set the default values
void FuncGraph::SetDefaultValues(const std::vector<std::string> &name_list, const AnfNodePtrList &value_list) {
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
  int64_t null_count =
    std::count_if(parameter_default_value_.begin(), parameter_default_value_.end(),
                  [](const std::pair<std::string, AnfNodePtr> &pair) { return IsValueNode<Null>(pair.second); });
  return parameter_default_value_.size() - LongToSize(null_count);
}

AnfNodePtr FuncGraph::GetVariableArgParameter() {
  if (!has_vararg_) {
    return nullptr;
  }

  size_t min_param_num = 1;
  if (has_kwarg_) {
    min_param_num += 1;
  }
  min_param_num += IntToSize(kw_only_args_count_);
  min_param_num += fv_param_count_;

  if (parameters_.size() < min_param_num) {
    MS_LOG(INTERNAL_EXCEPTION) << "Length of parameters is " << parameters_.size()
                               << " which less than the sum of following: fv_param_count: " << fv_param_count_
                               << ", has_vararg: " << has_vararg_ << ", has_kwarg: " << has_kwarg_
                               << ", kw_only_args_count_: " << kw_only_args_count_;
  }
  return parameters_[parameters_.size() - min_param_num];
}

std::string FuncGraph::GetVariableArgName() {
  if (!has_vararg_) {
    return "";
  }

  const auto &param_node = GetVariableArgParameter();
  MS_EXCEPTION_IF_NULL(param_node);
  auto parameter = param_node->cast_ptr<Parameter>();
  MS_EXCEPTION_IF_NULL(parameter);
  return parameter->name();
}

AnfNodePtr FuncGraph::GetVariableKwargParameter() {
  if (has_kwarg_) {
    if (parameters_.size() < fv_param_count_ + 1) {
      MS_LOG(INTERNAL_EXCEPTION) << "Length of parameters is " << parameters_.size() << ", fv_param_count is "
                                 << fv_param_count_ << ", parameters is less than 1 + fv_param_count";
    }
    return parameters_[(parameters_.size() - fv_param_count_) - 1];
  }
  return nullptr;
}

std::string FuncGraph::GetVariableKwargName() {
  auto kwarg_param = GetVariableKwargParameter();
  if (kwarg_param != nullptr) {
    auto parameter = kwarg_param->cast_ptr<Parameter>();
    MS_EXCEPTION_IF_NULL(parameter);
    return parameter->name();
  }
  return "";
}

AnfNodePtrList FuncGraph::GetKwOnlyArgsParameters() {
  AnfNodePtrList kw_only_args;
  if (kw_only_args_count_ == 0) {
    return kw_only_args;
  }

  size_t min_param_num = 0;
  size_t varargs_kwargs_num = 0;
  if (has_kwarg_) {
    min_param_num += 1;
    varargs_kwargs_num += 1;
  }
  min_param_num += IntToSize(kw_only_args_count_);
  min_param_num += fv_param_count_;

  if (parameters_.size() < min_param_num) {
    MS_LOG(INTERNAL_EXCEPTION) << "Length of parameters is " << parameters_.size()
                               << " which less than the sum of following: fv_param_count: " << fv_param_count_
                               << ", has_vararg: " << has_vararg_ << ", has_kwarg: " << has_kwarg_
                               << ", kw_only_args_count: " << kw_only_args_count_;
  }
  size_t kw_only_args_start_offset = parameters_.size() - min_param_num;
  std::copy(parameters_.cbegin() + kw_only_args_start_offset, parameters_.cend() - fv_param_count_ - varargs_kwargs_num,
            std::back_inserter(kw_only_args));
  return kw_only_args;
}

int FuncGraph::GetPositionalArgsCount() const {
  int count = SizeToInt(parameters_.size());
  if (has_kwarg_) {
    count--;
  }
  if (has_vararg_) {
    count--;
  }
  return (count - kw_only_args_count_) - SizeToInt(fv_param_count_);
}

AnfNodePtr FuncGraph::GetParameterByName(const std::string &name) {
  for (size_t i = 0; i < parameters_.size(); ++i) {
    MS_EXCEPTION_IF_NULL(parameters_[i]);
    auto param_cast = parameters_[i]->cast_ptr<Parameter>();
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
  auto nodes = mindspore::TopoSort(return_node(), SuccDepends, BelongSameGraph);
  for (const auto &node : nodes) {
    auto cnode = dyn_cast<CNode>(node);
    if (cnode != nullptr) {
      (void)cnodes.emplace_back(std::move(cnode));
    }
  }
  return cnodes;
}

void FuncGraph::EraseUnusedNodeInOrder() {
  auto mng = manager_.lock();
  if (mng != nullptr) {
    auto &all_nodes = nodes();
    // Erase unused cnode.
    for (auto it = order_.begin(); it != order_.cend();) {
      const auto &cnode = it->lock();
      if (cnode == nullptr) {
        it = order_.erase(it);
        continue;
      }
      if (!all_nodes.contains(cnode)) {
        MS_EXCEPTION_IF_NULL(cnode);
        MS_LOG(DEBUG) << "Remove node: " << cnode->DebugString() << " in graph " << ToString() << " order.";
        it = order_.erase(it);
        continue;
      }
      (void)++it;
    }
  }
}

void FuncGraph::EraseUnusedNodeInOrder(const AnfNodePtr &node) {
  if (node == nullptr) {
    return;
  }
  auto cnode = node->cast<CNodePtr>();
  if (cnode != nullptr) {
    auto iter = std::find_if(order_.cbegin(), order_.cend(), [&cnode](const CNodeWeakPtr &node) {
      return node.lock() != nullptr && node.lock() == cnode;
    });
    if (iter != order_.cend()) {
      (void)order_.erase(iter);
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
  auto iter = std::find_if(order_.cbegin(), order_.cend(), [&old_cnode](const CNodeWeakPtr &node) {
    return node.lock() != nullptr && node.lock() == old_cnode;
  });
  if (iter == order_.cend()) {
    // Skip if old node not found in order list.
    return;
  }
  auto new_cnode = new_node->cast<CNodePtr>();
  if (new_cnode != nullptr) {
    // Insert new node just before the old node.
    (void)order_.insert(iter, CNodeWeakPtr(new_cnode));
  }
  // Remove old node from order list.
  // Unused children nodes can be cleared by EraseUnusedNodeInOrder().
  (void)order_.erase(iter);
}

static AnfNodePtrList MakeInputNodes(const PrimitivePtr &primitive, const AnfNodePtrList &inputs) {
  AnfNodePtrList input_node_list;
  input_node_list.reserve(inputs.size() + 1);
  input_node_list.emplace_back(std::make_shared<ValueNode>(primitive));
  input_node_list.insert(input_node_list.end(), inputs.begin(), inputs.end());
  return input_node_list;
}

CNodePtr FuncGraph::NewCNode(const PrimitivePtr &primitive, const AnfNodePtrList &inputs) {
  auto input_node_list = MakeInputNodes(primitive, inputs);
  return NewCNode(std::move(input_node_list));
}

CNodePtr FuncGraph::NewCNodeInOrder(const PrimitivePtr &primitive, const AnfNodePtrList &inputs) {
  auto input_node_list = MakeInputNodes(primitive, inputs);
  return NewCNodeInOrder(std::move(input_node_list));
}

void FuncGraph::SetMultiTarget() const {
  auto graph_manager = manager();
  MS_EXCEPTION_IF_NULL(graph_manager);
  FuncGraphSet graphs = graph_manager->func_graphs();
  AnfNodePtrList all_nodes;
  for (auto &g : graphs) {
    auto nodes = mindspore::TopoSort(g->get_return());
    (void)std::copy(nodes.begin(), nodes.end(), std::back_inserter(all_nodes));
  }

  bool exist_multi_target = false;
  if (mindspore::ContainMultiTarget(all_nodes)) {
    exist_multi_target = true;
    MS_LOG(INFO) << "The graph " << ToString() << " exists the multi target.";
  }

  for (auto &g : graphs) {
    g->set_exist_multi_target(exist_multi_target);
  }
}

void FuncGraph::set_used_forward_nodes(const AnfNodePtrList &used_forward_nodes) {
  (void)std::for_each(used_forward_nodes.begin(), used_forward_nodes.end(), [this](const AnfNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    (void)used_forward_nodes_.insert(node);
  });
}

AnfNodePtrList FuncGraph::TopoSort(const AnfNodePtr &node) { return mindspore::TopoSort(node); }

bool FuncGraph::IsSideEffectCNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &primitive = GetCNodePrimitiveWithoutDoSignature(node);
  if (primitive != nullptr) {
    auto effect_info = GetPrimEffectInfo(primitive);
    if (effect_info.memory || effect_info.io) {
      MS_LOG(DEBUG) << "Side Effect Primitive CNode: " << node->DebugString();
      node->cast<CNodePtr>()->set_has_side_effect_node(true);
      return true;
    }
  } else if (node->isa<CNode>()) {
    // Call side effect node.
    auto first_node = node->cast<CNodePtr>()->input(0);
    if (first_node->isa<CNode>() && IsSideEffectCNode(first_node)) {
      first_node->cast<CNodePtr>()->set_has_side_effect_node(true);
      node->cast<CNodePtr>()->set_has_side_effect_node(true);
      MS_LOG(DEBUG) << "Side Effect Primitive CNode: " << first_node->DebugString();
      MS_LOG(DEBUG) << "Side Effect Primitive CNode: " << node->DebugString();
      return true;
    }
  }
  return false;
}

bool FuncGraph::CheckSideEffect(const AnfNodePtr &input) {
  if (IsSideEffectCNode(input)) {
    MS_LOG(DEBUG) << "Multiple side-effect node: " << input->DebugString();
    return true;
  }
  // Process {Depend -> StopGradient -> MakeTuple(call function, ...)}.
  if (input->isa<CNode>()) {
    auto fn_input = input->cast<CNodePtr>()->input(0);
    if (IsValueNode<prim::UnpackCall>(fn_input)) {
      fn_input = input->cast<CNodePtr>()->input(1);
    }
    if (IsValueNode<FuncGraph>(fn_input)) {
      auto func = GetValueNode<FuncGraphPtr>(fn_input);
      if (IsSideEffectCNode(func->output()) || func->HasIsolatedSideEffectNode()) {
        MS_LOG(DEBUG) << "Single nested side-effect node: " << input->DebugString();
        input->cast<CNodePtr>()->set_has_side_effect_node(true);
        if (func->output()->isa<CNode>()) {
          func->output()->cast<CNodePtr>()->set_has_side_effect_node(true);
        }
        return true;
      }
    }
  }
  return false;
}

bool FuncGraph::HasIsolatedSideEffectNode() {
  const auto node = output();
  if (!IsPrimitiveCNode(node, prim::kPrimDepend)) {
    return false;
  }
  auto cnode = dyn_cast<CNode>(node);
  MS_EXCEPTION_IF_NULL(cnode);
  auto attr_sort_rhs_first = cnode->GetAttr(kAttrTopoSortRhsFirst);
  auto sort_rhs_first =
    attr_sort_rhs_first != nullptr && attr_sort_rhs_first->isa<BoolImm>() && GetValue<bool>(attr_sort_rhs_first);
  if (!sort_rhs_first) {
    // Return false if it's definitely not side-effect Depend CNode.
    return false;
  }

  // To check side-effect nodes in {Depend -> StopGradient -> MakeTuple(...)}.
  constexpr size_t stop_gradient_pos = 2;
  auto stop_gradient_node = cnode->input(stop_gradient_pos);
  auto stop_gradient_cnode = dyn_cast<CNode>(stop_gradient_node);
  MS_EXCEPTION_IF_NULL(stop_gradient_cnode);
  constexpr size_t isolated_node_pos = 1;
  auto isolated_node = stop_gradient_cnode->input(isolated_node_pos);
  MS_EXCEPTION_IF_NULL(isolated_node);
  if (CheckSideEffect(isolated_node)) {
    stop_gradient_cnode->set_has_side_effect_node(true);
    return true;
  }
  if (IsPrimitiveCNode(isolated_node, prim::kPrimMakeTuple)) {
    auto isolated_cnode = dyn_cast<CNode>(isolated_node);
    MS_EXCEPTION_IF_NULL(isolated_cnode);
    for (size_t i = 1; i < isolated_cnode->size(); ++i) {
      auto input = isolated_cnode->input(i);
      if (CheckSideEffect(input)) {
        return true;
      }
    }
  }
  return false;
}

// Mark the side effect at output and func graph for later constant folding.
void FuncGraph::PresetCertainSideEffect() {
  if (!HasIsolatedSideEffectNode()) {
    return;
  }
  set_has_side_effect_node(true);
  MS_LOG(DEBUG) << "Set isolated side-effect node flag for " << ToString();
}

SeenNum NewFgSeenGeneration() {
  static SeenNum fg_seen_generation = 0;
  ++fg_seen_generation;
  // 0 is invalid number.
  if (fg_seen_generation == 0) {
    ++fg_seen_generation;
  }
  return fg_seen_generation;
}
}  // namespace mindspore
