/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "utils/func_graph_analyzer.h"

#include <algorithm>
#include <string>
#include <memory>
#include <vector>
#include "base/core_ops.h"
#include "utils/utils.h"

namespace mindspore {
const int64_t kGetAll = -1;

using FuncClosurePtr = std::shared_ptr<FuncClosure>;

class ValueGetter;
using ValueGetterPtr = std::shared_ptr<ValueGetter>;

class ValueManager;
using ValueManagerPtr = std::shared_ptr<ValueManager>;
ValueGetterPtr CreateValueGetter(const AnfNodePtr &node, const ValueManagerPtr &manager);
class ValueManager : public std::enable_shared_from_this<ValueManager> {
 public:
  ValueManager() = default;
  ~ValueManager() = default;
  ValueGetterPtr GetValueGetter(const AnfNodePtr &node) {
    MS_LOG(DEBUG) << "Try get value getter of node: " << node->DebugString();
    const auto &it = values_getters_.find(node);
    if (it == values_getters_.end()) {
      auto new_value_getter = CreateValueGetter(node, shared_from_this());
      values_getters_[node] = new_value_getter;
      MS_LOG(DEBUG) << "Create new value getter of node: " << node->DebugString();
      return new_value_getter;
    }
    return it->second;
  }

  bool UpdateGraphRelations(const std::vector<FuncClosurePtr> &func_closures, const AnfNodePtr &call) {
    MS_LOG(DEBUG) << "Func closure size: " << func_closures.size() << ", call: " << call->DebugString();
    auto change1 = UpdateGraphRealCallers(func_closures, call);
    auto change2 = UpdateCallerClosures(func_closures, call);
    return change1 || change2;
  }

  std::vector<FuncClosurePtr> GetCallClosures(const AnfNodePtr &call, const FuncGraphPtr &fg) {
    const auto &it = caller_closures_.find(call);
    if (it == caller_closures_.end()) {
      return {};
    }
    const auto &closures = it->second;
    std::vector<FuncClosurePtr> ret;
    (void)std::copy_if(closures.begin(), closures.end(), std::back_inserter(ret),
                       [&fg](const FuncClosurePtr &closure) { return closure->func_graph_ == fg; });
    return ret;
  }

  std::vector<AnfNodePtr> GetArg(const AnfNodePtr &param, const AnfNodePtr &call) {
    MS_EXCEPTION_IF_NULL(param);
    auto fg = param->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    const auto &parameters = fg->parameters();
    int64_t param_index = -1;
    for (size_t i = 0; i < parameters.size(); i++) {
      if (parameters[i] == param) {
        param_index = i;
      }
    }
    if (param_index == -1) {
      MS_LOG(EXCEPTION) << "Failed failed arg of parameter: " << param->DebugString()
                        << ",call: " << call->DebugString();
    }
    std::vector<AnfNodePtr> ret_args;
    auto call_cnode = call->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(call_cnode);
    auto closures = GetCallClosures(call, fg);
    for (const auto &closure : closures) {
      auto args = closure->GetArgs();
      (void)std::copy(call_cnode->inputs().begin() + 1, call_cnode->inputs().end(), std::back_inserter(args));
      if (parameters.size() != args.size()) {
        MS_LOG(EXCEPTION) << "Parameters size and args size are not equal, parameters size: " << parameters.size()
                          << ", args size: " << args.size() << ". Parameter: " << param->DebugString()
                          << ", call: " << call_cnode->DebugString();
      }
      ret_args.emplace_back(args[param_index]);
    }
    return ret_args;
  }

  HashMap<FuncGraphPtr, std::vector<CNodePtr>> func_graph_real_users_;
  HashMap<AnfNodePtr, std::vector<FuncClosurePtr>> caller_closures_;
  bool has_incorporate_call_ = false;

 private:
  HashMap<AnfNodePtr, ValueGetterPtr> values_getters_;
  bool UpdateGraphRealCallers(const std::vector<FuncClosurePtr> &func_closures, const AnfNodePtr &call) {
    bool change = false;
    for (const auto &fg_closure : func_closures) {
      auto map_it = func_graph_real_users_.find(fg_closure->func_graph_);
      if (map_it != func_graph_real_users_.end()) {
        const auto &real_callers = map_it->second;
        if (std::find(real_callers.begin(), real_callers.end(), call->cast<CNodePtr>()) != real_callers.end()) {
          continue;
        }
      }
      MS_LOG(DEBUG) << "Fg: " << fg_closure->func_graph_->ToString() << ", user: " << call->DebugString();
      func_graph_real_users_[fg_closure->func_graph_].push_back(call->cast<CNodePtr>());
      change = true;
    }
    return change;
  }

  bool UpdateCallerClosures(const std::vector<FuncClosurePtr> &func_closures, const AnfNodePtr &call) {
    auto map_it = caller_closures_.find(call);
    if (map_it != caller_closures_.end()) {
      bool change = false;
      auto &closures = map_it->second;
      std::copy_if(func_closures.begin(), func_closures.end(), std::back_inserter(closures),
                   [&closures, &change](const FuncClosurePtr &fg_closure) {
                     if (!fg_closure->ExistInList(closures)) {
                       change = true;
                       return true;
                     }
                     return false;
                   });
      return change;
    }
    caller_closures_[call] = func_closures;
    return true;
  }
};

class ValueGetter {
 public:
  ValueGetter(const AnfNodePtr &anf_node, const ValueManagerPtr &manager) : anf_node_(anf_node), manager_(manager) {}
  ~ValueGetter() = default;
  virtual ValueGetterPtr Visit(int64_t index, const std::shared_ptr<HashSet<AnfNodePtr>> &visit_path);
  virtual std::vector<FuncClosurePtr> GetFuncGraphs();

 protected:
  AnfNodePtr anf_node_ = nullptr;
  ValueManagerPtr manager_ = nullptr;
};

ValueGetterPtr ValueGetter::Visit(int64_t index, const std::shared_ptr<HashSet<AnfNodePtr>> &visit_path) {
  return nullptr;
}
std::vector<FuncClosurePtr> ValueGetter::GetFuncGraphs() { return {}; }

class MultipleValueGetter : public ValueGetter {
 public:
  explicit MultipleValueGetter(const ValueManagerPtr &manager) : ValueGetter(nullptr, manager) {}
  ~MultipleValueGetter() = default;
  MultipleValueGetter(const std::vector<ValueGetterPtr> &value_getters, const ValueManagerPtr &manager)
      : ValueGetter(nullptr, manager), value_getters_(value_getters) {}
  void AddValueGetter(const ValueGetterPtr &value_getter) {
    if (std::find(value_getters_.begin(), value_getters_.end(), value_getter) == value_getters_.end()) {
      value_getters_.push_back(value_getter);
    }
  }

  ValueGetterPtr Visit(int64_t index, const std::shared_ptr<HashSet<AnfNodePtr>> &visit_path) override;

  std::vector<FuncClosurePtr> GetFuncGraphs() override;

 private:
  std::vector<ValueGetterPtr> value_getters_;
};

ValueGetterPtr MultipleValueGetter::Visit(int64_t index, const std::shared_ptr<HashSet<AnfNodePtr>> &visit_path) {
  auto new_multiple_value_getter = std::make_shared<MultipleValueGetter>(manager_);
  for (auto &value_getter : value_getters_) {
    // Copy a new path.
    auto new_path = std::make_shared<HashSet<AnfNodePtr>>(*visit_path);
    new_multiple_value_getter->AddValueGetter(value_getter->Visit(index, new_path));
  }
  return new_multiple_value_getter;
}

std::vector<FuncClosurePtr> MultipleValueGetter::GetFuncGraphs() {
  std::vector<FuncClosurePtr> ret_func_closures;
  for (const auto &value_getter : value_getters_) {
    const auto &func_closures = value_getter->GetFuncGraphs();
    (void)std::copy_if(func_closures.begin(), func_closures.end(), std::back_inserter(ret_func_closures),
                       [&ret_func_closures](const auto &closure) { return !closure->ExistInList(ret_func_closures); });
  }
  return ret_func_closures;
}

class MakeTupleValueGetter : public ValueGetter {
 public:
  MakeTupleValueGetter(const AnfNodePtr &make_tuple, const ValueManagerPtr &manager)
      : ValueGetter(make_tuple, manager) {}
  ~MakeTupleValueGetter() = default;
  ValueGetterPtr Visit(int64_t index, const std::shared_ptr<HashSet<AnfNodePtr>> &visit_path) override;

  std::vector<FuncClosurePtr> GetFuncGraphs() override;
};

ValueGetterPtr MakeTupleValueGetter::Visit(int64_t index, const std::shared_ptr<HashSet<AnfNodePtr>> &visit_path) {
  (void)visit_path->insert(anf_node_);
  const auto &make_tuple = anf_node_->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(make_tuple);
  if (index == kGetAll) {
    auto multiple_value_getter = std::make_shared<MultipleValueGetter>(manager_);
    for (size_t i = 1; i < make_tuple->size(); i++) {
      auto new_path = std::make_shared<HashSet<AnfNodePtr>>(*visit_path);
      multiple_value_getter->AddValueGetter(manager_->GetValueGetter(make_tuple->input(i))->Visit(0, new_path));
    }
    return multiple_value_getter;
  }

  const auto &input_i = make_tuple->input(LongToSize(index + 1));
  auto input_i_getter = manager_->GetValueGetter(input_i);
  if (input_i_getter == nullptr) {
    MS_LOG(EXCEPTION) << "Make tuple: " << anf_node_->DebugString()
                      << " get input value getter failed. Index: " << index << ", input_i: " << input_i->DebugString();
  }
  return input_i_getter;
}

std::vector<FuncClosurePtr> MakeTupleValueGetter::GetFuncGraphs() {
  MS_LOG(EXCEPTION) << "MakeTupleValueGetter has no func graphs, anf_node_:" << anf_node_->DebugString();
}

class TupleGetItemValueGetter : public ValueGetter {
 public:
  TupleGetItemValueGetter(const AnfNodePtr &tuple_getitem, const ValueManagerPtr &manager)
      : ValueGetter(tuple_getitem, manager) {}
  ~TupleGetItemValueGetter() = default;
  ValueGetterPtr Visit(int64_t index, const std::shared_ptr<HashSet<AnfNodePtr>> &visit_path) override;

  std::vector<FuncClosurePtr> GetFuncGraphs() override;
};

ValueGetterPtr TupleGetItemValueGetter::Visit(int64_t index, const std::shared_ptr<HashSet<AnfNodePtr>> &visit_path) {
  (void)visit_path->insert(anf_node_);
  auto tuple_getitem = anf_node_->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(tuple_getitem);
  // Get cur index
  auto output_index_value_node = tuple_getitem->input(kInputNodeOutputIndexInTupleGetItem);
  auto value_node = output_index_value_node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto cur_index = LongToSize(GetValue<int64_t>(value_node->value()));
  // Get real input value getter
  auto real_input = tuple_getitem->input(kRealInputNodeIndexInTupleGetItem);
  const auto &real_input_value_getter = manager_->GetValueGetter(real_input).get();
  MS_EXCEPTION_IF_NULL(real_input_value_getter);
  return real_input_value_getter->Visit(cur_index, visit_path)->Visit(index, visit_path);
}

std::vector<FuncClosurePtr> TupleGetItemValueGetter::GetFuncGraphs() {
  MS_LOG(EXCEPTION) << "TupleGetItemValueGetter has no func graphs, anf_node_:" << anf_node_->DebugString();
}

class DependValueGetter : public ValueGetter {
 public:
  DependValueGetter(const AnfNodePtr &depend, const ValueManagerPtr &manager) : ValueGetter(depend, manager) {}
  ~DependValueGetter() = default;
  ValueGetterPtr Visit(int64_t index, const std::shared_ptr<HashSet<AnfNodePtr>> &visit_path) override;
  std::vector<FuncClosurePtr> GetFuncGraphs() override;
};

ValueGetterPtr DependValueGetter::Visit(int64_t index, const std::shared_ptr<HashSet<AnfNodePtr>> &visit_path) {
  (void)visit_path->insert(anf_node_);
  auto depend = anf_node_->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(depend);
  auto real_input = depend->input(kRealInputIndexInDepend);
  return manager_->GetValueGetter(real_input)->Visit(index, visit_path);
}
std::vector<FuncClosurePtr> DependValueGetter::GetFuncGraphs() {
  MS_LOG(EXCEPTION) << "DependValueGetter has no func graphs, anf_node_: " << anf_node_->DebugString();
}

class PartialValueGetter : public ValueGetter, public std::enable_shared_from_this<PartialValueGetter> {
 public:
  PartialValueGetter(const AnfNodePtr &partial, const ValueManagerPtr &manager) : ValueGetter(partial, manager) {}
  ~PartialValueGetter() = default;
  ValueGetterPtr Visit(int64_t index, const std::shared_ptr<HashSet<AnfNodePtr>> &visit_path) override;

  std::vector<FuncClosurePtr> GetFuncGraphs() override;

 private:
  ValueGetterPtr real_value_getter_ = nullptr;
};

ValueGetterPtr PartialValueGetter::Visit(int64_t index, const std::shared_ptr<HashSet<AnfNodePtr>> &visit_path) {
  (void)visit_path->insert(anf_node_);
  auto partial = anf_node_->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(partial);
  auto constexpr function_input_index = 1;
  auto real_input = partial->input(function_input_index);
  real_value_getter_ = manager_->GetValueGetter(real_input)->Visit(index, visit_path);
  return shared_from_this();
}

std::vector<FuncClosurePtr> PartialValueGetter::GetFuncGraphs() {
  if (real_value_getter_ == nullptr) {
    MS_LOG(EXCEPTION) << "Real value getter is null, please visit before get func graphs.node:"
                      << anf_node_->DebugString();
  }
  auto input_closures = real_value_getter_->GetFuncGraphs();
  constexpr auto arg_start_idx = 2;
  auto partial = anf_node_->cast<CNodePtr>();
  std::vector<FuncClosurePtr> closures;
  for (const auto &closure : input_closures) {
    auto arg_indexes = closure->arg_indexes_;
    auto arg_users = closure->arg_users_;
    for (size_t i = arg_start_idx; i < partial->inputs().size(); i++) {
      arg_indexes.emplace_back(i);
      arg_users.emplace_back(partial);
    }
    closures.emplace_back(std::make_shared<FuncClosure>(closure->func_graph_, arg_indexes, arg_users));
  }
  return closures;
}

class CallerValueGetter : public ValueGetter {
 public:
  CallerValueGetter(const AnfNodePtr &call, const ValueManagerPtr &manager) : ValueGetter(call, manager) {}
  ~CallerValueGetter() = default;
  ValueGetterPtr Visit(int64_t index, const std::shared_ptr<HashSet<AnfNodePtr>> &visit_path) override;

  std::vector<FuncClosurePtr> GetFuncGraphs() override;
};

ValueGetterPtr CallerValueGetter::Visit(int64_t index, const std::shared_ptr<HashSet<AnfNodePtr>> &visit_path) {
  // Get the func_graph called.
  auto call = anf_node_->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(call);
  auto input0 = call->input(0);
  std::vector<FuncClosurePtr> called_func_graphs;
  const auto &input0_getter = manager_->GetValueGetter(input0).get();
  // If get func graph from caller output, incorporate call exist.
  manager_->has_incorporate_call_ = true;
  if (input0_getter != nullptr) {
    auto new_path = std::make_shared<HashSet<AnfNodePtr>>();
    auto value_getter = input0_getter->Visit(0, new_path);
    called_func_graphs = value_getter->GetFuncGraphs();
  }
  if (called_func_graphs.empty()) {
    MS_LOG(EXCEPTION) << "Call node get value failed,node: " << anf_node_->DebugString();
  }
  // Get the call return value getters
  std::vector<ValueGetterPtr> output_value_getters;
  for (const auto &fg_closure : called_func_graphs) {
    auto new_path = std::make_shared<HashSet<AnfNodePtr>>(*visit_path);
    const auto &output_value_getter =
      manager_->GetValueGetter(fg_closure->func_graph_->output())->Visit(index, new_path);
    output_value_getters.push_back(output_value_getter);
  }
  if (output_value_getters.size() == 1) {
    return output_value_getters.back();
  } else {
    auto new_multiple_value_getter = std::make_shared<MultipleValueGetter>(manager_);
    for (const auto &output_value_getter : output_value_getters) {
      new_multiple_value_getter->AddValueGetter(output_value_getter);
    }
    return new_multiple_value_getter;
  }
}

std::vector<FuncClosurePtr> CallerValueGetter::GetFuncGraphs() {
  MS_LOG(EXCEPTION) << "Caller node can't call the func get func graphs, call node: " << anf_node_->DebugString();
}

class SwitchValueGetter : public ValueGetter {
 public:
  SwitchValueGetter(const AnfNodePtr &switch_node, const ValueManagerPtr &manager)
      : ValueGetter(switch_node, manager) {}
  ~SwitchValueGetter() = default;
  ValueGetterPtr Visit(int64_t index, const std::shared_ptr<HashSet<AnfNodePtr>> &visit_path) override;

  std::vector<FuncClosurePtr> GetFuncGraphs() override;
};

ValueGetterPtr SwitchValueGetter::Visit(int64_t index, const std::shared_ptr<HashSet<AnfNodePtr>> &visit_path) {
  auto constexpr true_branch_index = 2;
  auto constexpr false_branch_index = 3;
  auto switch_node = anf_node_->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(switch_node);
  auto true_branch_node = switch_node->input(true_branch_index);
  auto false_branch_node = switch_node->input(false_branch_index);
  auto true_value_getter = manager_->GetValueGetter(true_branch_node)->Visit(index, visit_path);
  auto new_path = std::make_shared<HashSet<AnfNodePtr>>(*visit_path);
  auto false_value_getter = manager_->GetValueGetter(false_branch_node)->Visit(index, new_path);

  auto multiple_value_getter = std::make_shared<MultipleValueGetter>(manager_);
  multiple_value_getter->AddValueGetter(true_value_getter);
  multiple_value_getter->AddValueGetter(false_value_getter);
  return multiple_value_getter;
}

std::vector<FuncClosurePtr> SwitchValueGetter::GetFuncGraphs() {
  MS_LOG(EXCEPTION) << "Switch node can't call the func get func graphs, switch: " << anf_node_->DebugString();
}

class SwitchLayerValueGetter : public ValueGetter {
 public:
  SwitchLayerValueGetter(const AnfNodePtr &switch_layer, const ValueManagerPtr &manager)
      : ValueGetter(switch_layer, manager) {}
  ~SwitchLayerValueGetter() = default;
  ValueGetterPtr Visit(int64_t index, const std::shared_ptr<HashSet<AnfNodePtr>> &visit_path) override;

  std::vector<FuncClosurePtr> GetFuncGraphs() override;
};

ValueGetterPtr SwitchLayerValueGetter::Visit(int64_t index, const std::shared_ptr<HashSet<AnfNodePtr>> &visit_path) {
  constexpr auto funcs_make_tuple_index = 2;
  const auto &switch_layer = anf_node_->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(switch_layer);
  auto switch_layer_input1 = switch_layer->input(funcs_make_tuple_index);
  return manager_->GetValueGetter(switch_layer_input1)->Visit(kGetAll, visit_path);
}

std::vector<FuncClosurePtr> SwitchLayerValueGetter::GetFuncGraphs() {
  MS_LOG(EXCEPTION) << "SwitchLayer node can't call the func get func graphs, switch layer: "
                    << anf_node_->DebugString();
}

// ParameterValueGetter should be analysis after others caller
class ParameterValueGetter : public ValueGetter {
 public:
  ParameterValueGetter(const AnfNodePtr &parameter, const ValueManagerPtr &manager) : ValueGetter(parameter, manager) {}
  ~ParameterValueGetter() = default;
  ValueGetterPtr Visit(int64_t index, const std::shared_ptr<HashSet<AnfNodePtr>> &visit_path) override;
  std::vector<FuncClosurePtr> GetFuncGraphs() override;
};

ValueGetterPtr ParameterValueGetter::Visit(int64_t index, const std::shared_ptr<HashSet<AnfNodePtr>> &visit_path) {
  // If get func graph from parameter, incorporate call exist.
  manager_->has_incorporate_call_ = true;
  // If anf_node_ in visit path, it remarks there is a recursive call
  if (visit_path->find(anf_node_) != visit_path->end()) {
    MS_LOG(INFO) << "Node: " << anf_node_->DebugString() << " has been visited.";
    return std::make_shared<MultipleValueGetter>(manager_);
  }
  // Add parameter to path
  (void)visit_path->insert(anf_node_);
  // Find node users
  auto param_func = anf_node_->func_graph();

  MS_EXCEPTION_IF_NULL(param_func);
  auto multiple_value_getter = std::make_shared<MultipleValueGetter>(manager_);
  const auto &it = manager_->func_graph_real_users_.find(param_func);
  if (it != manager_->func_graph_real_users_.end()) {
    const auto &calls = it->second;
    for (const auto &call : calls) {
      const auto &args = manager_->GetArg(anf_node_, call);
      for (const auto &arg : args) {
        auto new_path = std::make_shared<HashSet<AnfNodePtr>>(*visit_path);
        auto arg_value_getter = manager_->GetValueGetter(arg)->Visit(index, new_path);
        multiple_value_getter->AddValueGetter(arg_value_getter);
      }
    }
  }
  return multiple_value_getter;
}

std::vector<FuncClosurePtr> ParameterValueGetter::GetFuncGraphs() {
  // If parameter has not find it's arg value getter, we return a empty func graphs.
  MS_LOG(INFO) << "Undetermined parameter function,node: " << anf_node_->DebugString();
  return {};
}

class DirectValueGetter : public ValueGetter, public std::enable_shared_from_this<DirectValueGetter> {
 public:
  DirectValueGetter(const AnfNodePtr &value_node, const ValueManagerPtr &manager) : ValueGetter(value_node, manager) {}
  ~DirectValueGetter() = default;
  ValueGetterPtr Visit(int64_t index, const std::shared_ptr<HashSet<AnfNodePtr>> &) override;
  std::vector<FuncClosurePtr> GetFuncGraphs() override;

 private:
  std::vector<FuncClosurePtr> func_graphs_;
};

ValueGetterPtr DirectValueGetter::Visit(int64_t index, const std::shared_ptr<HashSet<AnfNodePtr>> &) {
  MS_LOG(DEBUG) << "Visit direct value getter: " << anf_node_->DebugString();
  return shared_from_this();
}
std::vector<FuncClosurePtr> DirectValueGetter::GetFuncGraphs() {
  if (func_graphs_.empty()) {
    func_graphs_.emplace_back(std::make_shared<FuncClosure>(GetValueNode<FuncGraphPtr>(anf_node_),
                                                            std::vector<size_t>(), std::vector<CNodePtr>()));
  }
  return func_graphs_;
}

bool IsFuncGraphCallNode(const AnfNodePtr &node) {
  if (!node->isa<CNode>()) {
    return false;
  }
  auto input0 = node->cast<CNodePtr>()->input(0);
  if (IsValueNode<Primitive>(input0)) {
    return false;
  }
  return true;
}

ValueGetterPtr CreateValueGetter(const AnfNodePtr &node, const ValueManagerPtr &manager) {
  if (IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
    return std::make_shared<MakeTupleValueGetter>(node, manager);
  }
  if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
    return std::make_shared<TupleGetItemValueGetter>(node, manager);
  }
  if (IsPrimitiveCNode(node, prim::kPrimDepend)) {
    return std::make_shared<DependValueGetter>(node, manager);
  }
  if (IsPrimitiveCNode(node, prim::kPrimPartial)) {
    return std::make_shared<PartialValueGetter>(node, manager);
  }
  if (IsPrimitiveCNode(node, prim::kPrimSwitch)) {
    return std::make_shared<SwitchValueGetter>(node, manager);
  }
  if (IsPrimitiveCNode(node, prim::kPrimSwitchLayer)) {
    return std::make_shared<SwitchLayerValueGetter>(node, manager);
  }
  if (IsFuncGraphCallNode(node)) {
    return std::make_shared<CallerValueGetter>(node, manager);
  }
  if (node->isa<Parameter>()) {
    return std::make_shared<ParameterValueGetter>(node, manager);
  }
  if (node->isa<ValueNode>()) {
    return std::make_shared<DirectValueGetter>(node, manager);
  }
  // Others are prim cnode.
  MS_LOG(EXCEPTION) << "Unexpected value getter node: " << node->DebugString();
}

std::vector<AnfNodePtr> GetAllCallNodes(const FuncGraphPtr &func_graph) {
  std::vector<AnfNodePtr> calls;
  const auto &all_nodes = TopoSort(func_graph->return_node(), SuccDeeperSimple, AlwaysInclude);
  std::copy_if(all_nodes.begin(), all_nodes.end(), std::back_inserter(calls),
               [](const AnfNodePtr &node) { return IsFuncGraphCallNode(node); });
  return calls;
}

bool FuncClosure::ExistInList(const std::vector<std::shared_ptr<FuncClosure>> &list) const {
  return std::any_of(list.begin(), list.end(), [this](const auto &list_item) { return *list_item == *this; });
}

std::vector<AnfNodePtr> FuncClosure::GetArgs() const {
  std::vector<AnfNodePtr> args;
  for (size_t i = 0; i < arg_indexes_.size(); i++) {
    args.emplace_back(arg_users_[i]->input(arg_indexes_[i]));
  }
  return args;
}

std::string FuncClosure::ToString() const {
  std::ostringstream buffer;
  buffer << "\nfg:," << func_graph_->ToString();
  for (size_t i = 0; i < arg_users_.size(); i++) {
    buffer << "\narg[" << i << "]:" << arg_users_[i]->input(arg_indexes_[i])->ToString();
  }
  buffer << "\n===================================================";
  return buffer.str();
}

FuncGraphAnalyzer::FuncGraphAnalyzer(const FuncGraphPtr &func_graph) {
  root_graph_ = func_graph;
  value_manager_ = std::make_shared<ValueManager>();
}

void FuncGraphAnalyzer::Run() {
  MS_LOG(INFO) << "Start.";
  const auto &calls = GetAllCallNodes(root_graph_);
  size_t cycle = 0;
  bool change = true;
  while (change) {
    change = false;
    MS_LOG(INFO) << "Func graph call analysis cycle:" << cycle;
    for (const auto &call : calls) {
      MS_LOG(INFO) << "Start analysis call node: " << call->DebugString();
      auto input0 = call->cast<CNodePtr>()->input(0);
      auto value_getter = value_manager_->GetValueGetter(input0);
      value_getter = value_getter->Visit(0, std::make_shared<HashSet<AnfNodePtr>>());
      change = value_manager_->UpdateGraphRelations(value_getter->GetFuncGraphs(), call) || change;
    }
    ++cycle;
  }
  DumpFuncGraphRealUsers();
  MS_LOG(INFO) << "End.";
}

std::vector<CNodePtr> FuncGraphAnalyzer::GetFuncGraphCallers(const FuncGraphPtr &func_graph) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto it = value_manager_->func_graph_real_users_.find(func_graph);
  if (it == value_manager_->func_graph_real_users_.end()) {
    MS_LOG(INFO) << "Find func graph:" << func_graph->ToString() << " failed.";
    return {};
  }
  return it->second;
}

std::vector<FuncGraphPtr> FuncGraphAnalyzer::GetCallerFuncGraphs(const AnfNodePtr &node) const {
  const auto &closures = GetCallClosures(node);
  std::vector<FuncGraphPtr> func_graphs;
  for (const auto &closure : closures) {
    if (std::find(func_graphs.begin(), func_graphs.end(), closure->func_graph_) == func_graphs.end()) {
      func_graphs.emplace_back(closure->func_graph_);
    }
  }
  return func_graphs;
}

const std::vector<FuncClosurePtr> &FuncGraphAnalyzer::GetCallClosures(const AnfNodePtr &call) const {
  MS_EXCEPTION_IF_NULL(call);
  auto it = value_manager_->caller_closures_.find(call);
  if (it != value_manager_->caller_closures_.end()) {
    return it->second;
  }
  MS_LOG(EXCEPTION) << "Find closure of call: " << call->DebugString() << " failed.";
}

std::vector<AnfNodePtr> FuncGraphAnalyzer::GetArg(const AnfNodePtr &param, const AnfNodePtr &call) const {
  return value_manager_->GetArg(param, call);
}

void FuncGraphAnalyzer::DumpFuncGraphRealUsers() const {
  const auto &func_graph_callers = value_manager_->func_graph_real_users_;
  MS_LOG(INFO) << "Func graph size:" << func_graph_callers.size();
  size_t fg_index = 0;
  std::ostringstream buffer;
  buffer << "\n";
  for (auto &it : func_graph_callers) {
    const auto fg = it.first;
    const auto callers = it.second;
    buffer << "FuncGraph[" << fg_index++ << "]:" << fg->ToString() << "\n";
    for (size_t i = 0; i < callers.size(); i++) {
      buffer << "---->Caller[" << i << "]:" << callers[i]->DebugString() << "\n";
    }
  }
  MS_LOG(INFO) << buffer.str();
}

bool FuncGraphAnalyzer::ExistClosure() const {
  for (const auto &[call, closures] : value_manager_->caller_closures_) {
    for (const auto &closure : closures) {
      if (closure->arg_indexes_.empty()) {
        continue;
      }
      const auto &last_arg_user = closure->arg_users_.back();
      // Partial's arg and call are in same graph, this is not closure.
      if (last_arg_user->func_graph() != call->func_graph()) {
        return true;
      }
    }
  }
  return false;
}

bool FuncGraphAnalyzer::HasIncorporateCall() const { return value_manager_->has_incorporate_call_; }
}  // namespace mindspore
// namespace mindspore
