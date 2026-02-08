/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "frontend/jit/pi/graph_capture/graph_arguments_optimizer.h"
#include <algorithm>
#include <climits>
#include <iterator>
#include <list>
#include <map>
#include <regex>
#include <string>
#include "frontend/jit/pi/graph_build/build_graph_utils.h"
#include "frontend/jit/pi/graph_build/func_graph_builder.h"
#include "frontend/jit/pi/graph_capture/abstract_object.h"
#include "frontend/jit/pi/graph_capture/graph_build_helper.h"
#include "frontend/jit/pi/graph_compiler/compiler.h"
#include "frontend/jit/pi/graph_compiler/pi_ir/ctrl_flow.h"
#include "frontend/jit/pi/graph_guard/cache.h"
#include "frontend/jit/pi/utils/opcode_declare.h"
#include "include/frontend/jit/ps/executor/jit_executor_py.h"

namespace mindspore {
namespace pijit {
GraphArgumentOptimizerPtr GraphArgumentOptimizer::GetNewInstance(Graph *graph) {
  ArgsStatusMap status;
  return std::make_shared<GraphArgumentOptimizer>(graph, status);
}

void GraphArgumentOptimizer::Run(const std::vector<ValueNode *> &outputs) {
  MS_EXCEPTION_IF_NULL(graph_);
  if (outputs.empty()) {
    MS_LOG(INFO) << "No output set for graph " << graph_->GetCodeName();
    return;
  }
  const auto &builder = graph_->func_graph_builder();
  MS_EXCEPTION_IF_NULL(builder);
  auto func_graph = builder->graph();
  if (func_graph == nullptr) {
    return;
  }
  if (IsTopGraph() && status_.empty()) {
    InitializeArgumentsUsageStatus();
  }
  if (status_.empty()) {
    MS_LOG(INFO) << "No argument to optimize for graph " << graph_->GetCodeName();
    return;
  }
  if (nodes_.empty()) {
    nodes_ = CollectCapturedNodes();
  }
  related_nodes_ = CollectNodesUsingArgument(nodes_);
  AnalyzeArgumentsUsageStatus(outputs);
  if (!IsTopGraph()) {
    return;
  }
  arguments_ = CollectUsedInputsInGraph();
  for (auto &stat : status_) {
    auto is_used_in_graph = std::find(arguments_.begin(), arguments_.end(), stat.second.first) != arguments_.end();
    if (stat.second.second <= 0 && is_used_in_graph) {
      MS_LOG(INFO) << "Analyze Not Match : " << stat.second.first->ToString();
      stat.second.second = INT_MAX;
    }
    if (stat.second.second > 0 && !is_used_in_graph) {
      MS_LOG(INFO) << "Used as constant : " << stat.second.first->ToString();
    }
  }
  auto constant_args = CollectConstantArguments();
  auto replace_map = CollectDuplicateArguments();
  GuardExpandParameters();
  EliminateRedundantArguments(constant_args, replace_map);
}

void GraphArgumentOptimizer::InitializeArgumentsUsageStatus() {
  for (auto &argument : graph_->prepare().inputs_) {
    MS_EXCEPTION_IF_NULL(argument);
    auto vobj = argument->GetOwnVobj();
    duplicate_args_[vobj].insert(argument);
    if (status_.find(vobj) != status_.end()) {
      MS_LOG(INFO) << "Find duplicate argument " << argument->ToString() << " same as "
                   << status_[vobj].first->ToString();
      continue;
    }
    MS_LOG(INFO) << "Initialize argument : " << argument->ToString();
    status_[vobj] = std::make_pair(argument, 0);
  }
}

std::vector<ValueNode *> GraphArgumentOptimizer::CollectCapturedNodes() const {
  const auto &nodes = graph_->GetTracedNodes();
  auto break_bci = graph_->GetStopTraceBci();
  if (break_bci == -1) {
    return nodes;
  }
  std::vector<ValueNode *> result;
  for (const auto &node : nodes) {
    if (node->bci() > break_bci) {
      break;
    }
    result.push_back(node);
  }
  return result;
}

bool GraphArgumentOptimizer::IsUsingAnyArgument(AObject *vobj) const {
  MS_EXCEPTION_IF_NULL(vobj);
  if (status_.find(vobj) != status_.end()) {
    return true;
  }
  auto type = vobj->GetType();
  if (type == AObject::kTypeTuple || type == AObject::kTypeList) {
    auto seq = static_cast<AbstractSequence *>(vobj);
    auto elements = seq->GetElementsWithInit();
    return !elements.empty() &&
           std::any_of(elements.begin(), elements.end(), [this](auto &element) { return IsUsingAnyArgument(element); });
  }
  if (type == AObject::kTypeDict) {
    auto dict = static_cast<AbstractDict *>(vobj);
    auto elements = dict->GetElementsWithInit();
    return !elements.empty() && std::any_of(elements.begin(), elements.end(), [this](auto &element) {
      return IsUsingAnyArgument(element.first) || IsUsingAnyArgument(element.second);
    });
  }
  return false;
}

std::vector<ValueNode *> GraphArgumentOptimizer::CollectNodesUsingArgument(const std::vector<ValueNode *> &nodes) {
  std::vector<ValueNode *> related_nodes;
  std::unordered_set<const AObject *> obj_set;
  std::for_each(status_.begin(), status_.end(), [this, &obj_set](const auto &kv) {
    obj_set.insert(kv.first);
    related_nodes_.push_back(kv.second.first);
  });
  for (auto &node : nodes) {
    // Output is not a argument node,
    // There is no argument in the node's inputs, no need to analysis.
    if (obj_set.find(node->GetOwnVobj()) == obj_set.end()) {
      auto inputs = node->inputs();
      if (inputs.empty()) {
        continue;
      } else {
        if (std::all_of(inputs.begin(), inputs.end(), [this, &related_nodes, &obj_set](auto &input) {
              return std::find(related_nodes.begin(), related_nodes.end(), input) == related_nodes.end() &&
                     obj_set.find(input->GetOwnVobj()) == obj_set.end() && !IsUsingAnyArgument(input->GetOwnVobj());
            })) {
          continue;
        }
      }
    }
    obj_set.insert(node->GetOwnVobj());
    // Has not added to related_nodes.
    if (std::find(related_nodes.begin(), related_nodes.end(), node) != related_nodes.end()) {
      continue;
    }
    MS_LOG(DEBUG) << "Find using argument node : " << node->ToString();
    related_nodes.push_back(node);
  }
  return related_nodes;
}

bool IsNeedMarkAllArguments(const CallNode *call_node) {
  MS_EXCEPTION_IF_NULL(call_node);
  auto func = call_node->input(0)->GetOwnVobj()->GetPyObject();
  if (func.ptr() == nullptr || !py::hasattr(func, "__qualname__")) {
    return true;
  }
  const auto &qualname = py::getattr(func, "__qualname__").cast<std::string>();
  const std::vector<std::string> builtins = {"dict", "dict.get",    "dict.items",  "dict.keys",   "dict.values", "list",
                                             "len",  "list.append", "list.extend", "list.insert", "tuple"};
  return std::find(builtins.begin(), builtins.end(), qualname) == builtins.end();
}

void GraphArgumentOptimizer::MarkAllArguments(AObject *vobj) {
  MS_EXCEPTION_IF_NULL(vobj);
  if (status_.find(vobj) != status_.end()) {
    status_[vobj].second++;
    return;
  }
  auto seq = dynamic_cast<AbstractSequence *>(vobj);
  if (seq != nullptr) {
    auto elements = seq->GetElementsWithInit();
    std::for_each(elements.begin(), elements.end(), [this](auto &element) { MarkAllArguments(element); });
  } else {
    auto dict = dynamic_cast<AbstractDict *>(vobj);
    if (dict != nullptr) {
      auto key_values = dict->GetElementsWithInit();
      std::for_each(key_values.begin(), key_values.end(), [this](auto &key_value) {
        MarkAllArguments(key_value.first);
        MarkAllArguments(key_value.second);
      });
    }
  }
}

bool GraphArgumentOptimizer::AnalyzeCallNode(CallNode *call_node) {
  MS_EXCEPTION_IF_NULL(call_node);
  auto sub_graph = call_node->GetSubGraph();
  // module name set to mindspore, means we consider the call can not be analyzed
  // all arguments will be marked as used
  std::string module_name = "mindspore";
  auto ret_val = sub_graph == nullptr ? nullptr : sub_graph->GetRetVal();
  if (ret_val != nullptr) {
    module_name = sub_graph->GetModuleName();
  }
  bool is_need_analyze = module_name.find("mindspore") == std::string::npos;
  if (is_need_analyze) {
    // Analyze the arguments usage status in sub-graph.
    auto optimizer = std::make_shared<GraphArgumentOptimizer>(sub_graph, status_);
    optimizer->arguments_ = arguments_;
    optimizer->Run({ret_val});
    status_ = optimizer->status_;
    MS_EXCEPTION_IF_NULL(ret_val);
    if (ret_val->GetOwnVobj() != call_node->GetOwnVobj()) {
      MS_LOG(WARNING) << "The aobj of " << call_node->ToString() << " should be same as " << ret_val->ToString();
      std::for_each(ret_val->inputs().begin(), ret_val->inputs().end(),
                    [this](auto &value) { MarkAllArguments(value->GetOwnVobj()); });
    }
  }
  return !is_need_analyze && IsNeedMarkAllArguments(call_node);
}

void GraphArgumentOptimizer::AnalyzeArgumentsUsageStatus(const std::vector<ValueNode *> &ret_values) {
  std::unordered_set<const ValueNode *> visited;
  std::unordered_set<const ValueNode *> need_mark;
  std::list<ValueNode *> nodes(ret_values.begin(), ret_values.end());
  if (IsTopGraph()) {
    std::for_each(nodes.begin(), nodes.end(), [this](auto &node) { MarkAllArguments(node->GetOwnVobj()); });
  }
  while (!nodes.empty()) {
    nodes.unique();
    auto node = nodes.front();
    nodes.pop_front();
    auto vobj = node->GetOwnVobj();
    if (status_.find(vobj) != status_.end()) {
      status_[vobj].second++;
    }
    if (std::find(related_nodes_.begin(), related_nodes_.end(), node) == related_nodes_.end()) {
      continue;
    }
    if (visited.find(node) != visited.end()) {
      continue;
    }
    visited.insert(node);
    auto inputs = node->inputs();
    // Only the arguments of the top graph will be analyzed.
    if (Opcode(node->GetOpcode()).IsCall()) {
      auto call_node = static_cast<CallNode *>(node);
      if (AnalyzeCallNode(call_node)) {
        std::for_each(inputs.begin(), inputs.end(), [&need_mark](auto &input) {
          if (Opcode(input->GetOpcode()).IsBuildOp()) {
            need_mark.insert(input);
          }
        });
      }
    }
    bool is_not_actually_use = (need_mark.find(node) == need_mark.end() && Opcode(node->GetOpcode()).IsBuildOp());
    std::for_each(inputs.begin(), inputs.end(), [this, &nodes, is_not_actually_use](auto &input) {
      nodes.push_back(input);
      auto input_vobj = input->GetOwnVobj();
      if (is_not_actually_use && status_.find(input_vobj) != status_.end()) {
        status_[input_vobj].second--;
      }
    });
  }
}

std::vector<ValueNode *> GraphArgumentOptimizer::CollectConstantArguments() const {
  std::vector<ValueNode *> constant_args;
  std::for_each(status_.begin(), status_.end(), [this, &constant_args](const auto &stat) {
    if (stat.second.second <= 0) {
      return;
    }
    if (std::find(arguments_.begin(), arguments_.end(), stat.second.first) != arguments_.end()) {
      return;
    }
    auto args = duplicate_args_.at(stat.first);
    constant_args.insert(constant_args.end(), args.begin(), args.end());
  });
  return constant_args;
}

std::unordered_map<ValueNode *, ValueNode *> GraphArgumentOptimizer::CollectDuplicateArguments() const {
  std::unordered_map<ValueNode *, ValueNode *> duplicate_args;
  for (const auto &[vobj, pair] : status_) {
    if (pair.second <= 0) {
      continue;
    }
    for (auto &arg : duplicate_args_.at(vobj)) {
      if (arg == pair.first) {
        continue;
      }
      duplicate_args[arg] = pair.first;
    }
  }
  return duplicate_args;
}

static std::string GetParameterIdentity(const AnfNodePtr &param) {
  auto name = GetParameterName(param);
  auto pos = name.find("input_");
  name = pos == std::string::npos ? name : name.substr(pos);
  return name;
}

static std::string MakeCompileGraphName(const std::string &co_name) {
  static size_t id = 0;
  constexpr const char *reg_mark = "<compile\\[\\d+\\]>";
  return "<compile[" + std::to_string(id++) + "]>" + std::regex_replace(co_name, std::regex(reg_mark), "");
}

std::vector<ValueNode *> GraphArgumentOptimizer::CollectUsedInputsInGraph() const {
  const auto &builder = graph_->func_graph_builder();
  MS_EXCEPTION_IF_NULL(builder);
  builder->ClearNodeAbstract();
  auto co = graph_->GetCodeObj();
  MS_EXCEPTION_IF_NULL(co);
  auto co_name = MakeCompileGraphName(PyUnicode_AsUTF8(co->co_name));
  builder->SetGraphName(co_name);
  common::SetCompileConfig("ENABLE_ELIMINATE_UNUSED_PARAMS", "1", true);
  GraphCompiler::CompileInfo compile_info{py::cast<std::string>(co->co_filename), co_name, co->co_firstlineno,
                                          static_cast<int>(arguments_.size()),    0,       0,
                                          builder->origin_top_input_num()};
  builder->SetCompileResult(GraphCompiler::Compile(builder->graph(), compile_info));
  common::SetCompileConfig("ENABLE_ELIMINATE_UNUSED_PARAMS", "0", true);
  auto executor = pipeline::GetExecutor();
  MS_EXCEPTION_IF_NULL(executor);
  auto func_graph = executor->GetFuncGraph(builder->GetCompileResult().first);
  MS_EXCEPTION_IF_NULL(func_graph);
  auto parameters = func_graph->parameters();
  std::vector<std::string> names;
  std::transform(parameters.begin(), parameters.end(), std::back_inserter(names),
                 [](const auto &param) { return GetParameterIdentity(param); });
  auto is_used = [&names](auto &node) {
    auto param = node->GetGraph()->func_graph_builder()->FindNodeByWrapper(node->abstract_wrapper());
    return std::find(names.begin(), names.end(), GetParameterIdentity(param)) != names.end();
  };
  std::vector<ValueNode *> inputs;
  auto nodes = graph_->prepare().inputs_;
  std::for_each(nodes.begin(), nodes.end(), [&inputs, &is_used](auto &node) {
    if (is_used(node)) {
      inputs.push_back(node);
    }
  });
  auto cnt = parameters.size() - func_graph->fv_param_count();
  if (cnt != inputs.size()) {
    MS_LOG(WARNING) << "Arguments should be " << cnt << " but got " << inputs.size();
  }
  return inputs;
}

void GraphArgumentOptimizer::GuardExpandParameters() {
  using InfoIter = std::map<const AObject *, Graph::ExpandParamInfo>::iterator;
  std::vector<const AObject *> remove_items;
  std::for_each(status_.begin(), status_.end(), [&remove_items](const auto &stat) {
    if (stat.second.second <= 0) {
      remove_items.push_back(stat.first);
    }
  });
  auto &info = graph_->GetExpandParamInfo();
  while (!remove_items.empty()) {
    std::vector<const AObject *> new_items;
    for (InfoIter iter = info.begin(); iter != info.end();) {
      auto elements = iter->second.elements_;
      auto elements_remove_cond = [&remove_items](const auto &vobj) {
        return std::find(remove_items.begin(), remove_items.end(), vobj) != remove_items.end();
      };
      auto elements_remove_func = std::remove_if(elements.begin(), elements.end(), elements_remove_cond);
      elements.erase(elements_remove_func, elements.end());
      if (!elements.empty()) {
        ++iter;
      } else {
        new_items.push_back(iter->first);
        iter = info.erase(iter);
      }
    }
    remove_items.swap(new_items);
  }
  std::for_each(info.begin(), info.end(), [this](auto &iter) {
    auto seq = dynamic_cast<const AbstractSequence *>(iter.first);
    if (seq == nullptr) {
      return;
    }
    MS_LOG(INFO) << "Add len guard for " << iter.second.node_->ToString();
    graph_->GuardSequenceNodeLength(iter.second.node_, seq->size());
  });
}

void GraphArgumentOptimizer::EliminateRedundantArguments(const std::vector<ValueNode *> &constant_args,
                                                         const std::unordered_map<ValueNode *, ValueNode *> &args) {
  graph_->prepare().inputs_.clear();
  graph_->prepare().operations_.clear();
  std::for_each(arguments_.begin(), arguments_.end(), [this](auto &argument) {
    graph_->PrepareParameter(argument);
    graph_->GuardParameter(argument);
    MS_LOG(INFO) << "Add parameter guard for " << argument->ToString();
  });
  std::for_each(constant_args.begin(), constant_args.end(), [this](auto &argument) {
    graph_->PrepareParameter(argument);
    graph_->GuardValueNode(argument);
    MS_LOG(INFO) << "Add value guard for " << argument->ToString();
  });
  for (auto &[replaced, target] : args) {
    graph_->GetGuardManager()->GetGuard()->GuardOn(graph_->TraceValueNode(replaced), GuardLevel::kGuardMatchIDS);
    graph_->GetGuardManager()->GetGuard()->GuardOn(graph_->TraceValueNode(target), GuardLevel::kGuardMatchIDS);
    graph_->PrepareParameter(replaced);
    MS_LOG(INFO) << "Add match guard for " << replaced->ToString() << " vs " << target->ToString();
  }
}
}  // namespace pijit
}  // namespace mindspore
