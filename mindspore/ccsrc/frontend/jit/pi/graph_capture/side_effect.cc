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
#include "frontend/jit/pi/graph_capture/side_effect.h"

#include <algorithm>
#include <climits>
#include <iterator>
#include <list>
#include <string>
#include <utility>

#include "frontend/jit/pi/graph_capture/code_generator.h"
#include "frontend/jit/pi/graph_capture/graph.h"
#include "frontend/jit/pi/graph_capture/node.h"
#include "frontend/jit/pi/graph_guard/infer.h"
#include "frontend/jit/pi/pi_jit_config.h"
#include "frontend/jit/pi/utils/opcode_util.h"
#include "utils/compile_config.h"

namespace mindspore {
namespace pijit {

constexpr auto kSetAttr = "setattr";
constexpr auto kDelAttr = "delattr";
constexpr auto kSetItem = "__setitem__";
constexpr auto kDelItem = "__delitem__";

ValueNode *GetSelfFromKnownMethod(ValueNode *call_node, bool *is_method_descriptor) {
  ValueNode *method_node = call_node->input(0);
  PyObject *method_object = method_node->GetVobj()->GetPyObject().ptr();
  ValueNode *self = nullptr;
  bool is_not_method = PyFunction_Check(method_object) || Py_IS_TYPE(method_object, &PyMethodDescr_Type);
  if (is_not_method) {
    self = call_node->input(1);
  } else if (method_node->GetOpcode() == LOAD_ATTR) {
    self = method_node->input(0);
  }
  if (is_method_descriptor != nullptr) {
    *is_method_descriptor = is_not_method;
  }
  return self;
}

void SideEffectData::RecordModifiedAndReplacedNode(ValueNode *old_node, ValueNode *new_node) {
  ValueNode **old_record = &modified_and_replaced_map_[new_node];
  ValueNode *real_src = old_node;
  const auto &m = modified_and_replaced_map_;
  for (auto iter = m.find(real_src); iter != m.end(); iter = m.find(real_src)) {
    real_src = iter->second;
  }
  *old_record = real_src;
}

void SideEffectData::AddAttrData(const std::string &name, ValueNode *src, ValueNode *new_attr) {
  auto &map = attr_cache_.modified_attrs_[src];
  map[name] = new_attr;
}

void SideEffectData::AddGlobalData(const std::string &module_name, const std::string &name, ValueNode *node) {
  auto &dict = global_cache_.modified_globals_[module_name];
  dict[name] = node;
}

void SideEffectData::ClearCache() {
  attr_cache_.modified_attrs_.clear();
  global_cache_.modified_globals_.clear();
}

SideEffect::CacheResult SideEffect::LoadAttr(ValueNode *src, const std::string &name) const {
  const auto &cache = data_->attr_cache().modified_attrs_;
  if (cache.empty()) {
    return {};  // no attribute modified
  }

  CacheResult result{};
  auto Find = [&cache, &name, &result](ValueNode *src_node) {
    auto map_iter = cache.find(src_node);
    if (map_iter == cache.end()) {
      return false;  // not find attr map of this node
    }
    auto attr_iter = map_iter->second.empty() ? map_iter->second.end() : map_iter->second.find(name);
    if (attr_iter == map_iter->second.end()) {
      return false;  // not find attr of this node
    }
    result = {attr_iter->second, attr_iter->second == nullptr};
    return true;
  };

  PyObject *src_object = src->GetVobj() ? src->GetVobj()->GetPyObject().ptr() : nullptr;
  if (src_object == nullptr) {
    Find(src);
  } else if (!CheckConstPyObject(src_object)) {
    auto iter = data()->id_map().find(src_object);
    MS_EXCEPTION_IF_CHECK_FAIL(iter != data()->id_map().end(), "missing track for node " + src->ToString());
    (void)std::find_if(iter->second.begin(), iter->second.end(), Find);
  }
  return result;
}

SideEffect::CacheResult SideEffect::LoadGlobal(const std::string &module_name, const std::string &name) const {
  const auto &cache = data_->global_cache().modified_globals_;
  if (cache.empty()) {
    return {};  // no global modified
  }
  auto m_iter = cache.find(module_name);
  if (m_iter == cache.end()) {
    return {};  // this module global not modified
  }
  auto value_iter = m_iter->second.find(name);
  if (value_iter == m_iter->second.end()) {
    return {};  // this name not modified
  }
  return {value_iter->second, value_iter->second == nullptr};
}

static bool IsTensorOpt(SideEffect::Type type, ValueNode *oper, const std::string &method_name) {
  if (!pijit::kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kTensorSetitemSideEffectOpt)) {
    MS_LOG(DEBUG) << "Tensor setitem side-effect optimization is not enabled";
    return false;
  }
  ValueNode *tensor;
  if (type == SideEffect::Type::kBuiltinMethod) {
    tensor = GetSelfFromKnownMethod(oper);
  } else {
    return false;
  }
  // must be tensor api
  if (tensor->GetVobj()->GetType() != AObject::kTypeTensor) {
    return false;
  }
  // must be return a tensor
  if (oper->GetVobj()->GetType() != AObject::kTypeTensor) {
    return false;
  }
  // must be computed by graph, but graph can't apply side effect to tensor
  if (method_name == kSetItem) {
    MS_LOG(DEBUG) << "Enable Tensor setitem side-effect optimization for node: " << ToString(oper);
    return true;
  }
  // function Tensor.assign_value can't run in graph
  // primitive ops.assign only effect for Parameter in graph
  return false;
}

bool SideEffect::Record(ValueNode *node, Type type, std::string name) {
  MS_LOG(DEBUG) << "Record SideEffect node: " << ToString(node);
  int opcode = node->GetOpcode();
  if (opcode == STORE_ATTR || opcode == DELETE_ATTR) {
    ValueNode *src_node = opcode == DELETE_ATTR ? node->input(0) : node->input(1);
    ValueNode *attr_node = opcode == DELETE_ATTR ? nullptr : node->input(0);
    data_->AddAttrData(node->GetName(), src_node, attr_node);
    type = kBuiltinFunction;
    name = opcode == STORE_ATTR ? kSetAttr : kDelAttr;
    MS_LOG(DEBUG) << "Record attr data, src node: " << ToString(src_node) << ", attr node: " << ToString(attr_node);
  } else if (opcode == STORE_GLOBAL || opcode == DELETE_GLOBAL) {
    MS_EXCEPTION_IF_NULL(node->GetGraph());
    ValueNode *new_value = opcode == DELETE_GLOBAL ? nullptr : node->input(0);
    std::string module_name = node->GetGraph()->GetModuleName();
    if (module_name.empty()) {
      return false;  // empty module name, unknown global source
    }
    data_->AddGlobalData(module_name, node->GetName(), new_value);
    type = kSetGlobal;
  } else if (opcode == STORE_SUBSCR || opcode == DELETE_SUBSCR) {
    type = kDefault;
    name = opcode == STORE_SUBSCR ? kSetItem : kDelItem;
  } else if (Opcode(opcode).IsCall() && CheckCallRecord(node, type, name)) {
  } else if (opcode == STORE_DEREF) {
    // No action needed
  } else {
    MS_LOG(INFO) << "unimplemented side-effect " << node->ToString();
    return false;
  }
  node->MarkSideEffectNode();
  size_t order_index = nodes_.size();
  Entry entry{node, type, order_index, std::move(name)};
  if (IsTensorOpt(entry.type_, entry.node_, entry.method_name_)) {
    entry.type_ = kTensorOptMethod;
  }
  AddKeepAlive(GetKeepAlive(entry));
  nodes_[node] = std::move(entry);
  return true;
}

bool SideEffect::CheckCallRecord(ValueNode *node, SideEffect::Type type, const std::string &name) {
  if (type == kDefault) {
    return true;
  }
  if (type == kBuiltinFunction && (name == kSetAttr || name == kDelAttr)) {
    size_t index = 1;
    ValueNode *src_node = node->input(index++);
    py::object name = node->input(index++)->GetVobj()->GetPyObject();
    ValueNode *attr_node = node->inputs().size() == index ? nullptr : node->input(index);
    data_->AddAttrData(PyUnicode_AsUTF8(name.ptr()), src_node, attr_node);
    return true;
  }
  // check list.append, dict.pop, list.__setitem__, dict.__setitem__
  if (type == kBuiltinMethod && GetSelfFromKnownMethod(node) != nullptr) {
    return true;
  }
  return false;
}

std::vector<ValueNode *> SideEffect::GetKeepAlive(const Entry &e) const {
  ValueNode *node = e.node_;
  Type type = e.type_;
  int opcode = node->GetOpcode();
  std::vector<ValueNode *> alive = node->inputs();
  if (Opcode(opcode).IsCall() && type >= kBuiltinMethod) {
    alive[0] = GetSelfFromKnownMethod(node);  // replace function
  }
  if (type == kTensorOptMethod) {
    alive = {alive[0]};  // the oldest version of modified object
  }
  for (auto iter = alive.begin(); iter != alive.end(); ++iter) {
    *iter = GetSource(*iter);
  }
  if (type == kTensorOptMethod) {
    alive.push_back(node);  // the latest version of modified object
  }
  return alive;
}

std::vector<ValueNode *> SideEffect::GetKeepAlive(ValueNode *node) const {
  auto it = nodes_.find(node);
  if (it == nodes_.end()) {
    MS_LOG(DEBUG) << "Is not side-effect node! " << ToString(node);  // It shouldn't happen
    return {};
  }
  return GetKeepAlive(it->second);
}

ValueNode *SideEffect::GetSource(ValueNode *src_node) const {
  const auto &map = data()->modified_and_replaced_map();
  if (map.empty() || src_node == nullptr) {
    return src_node;
  }
  auto iter = map.find(src_node);
  return iter != map.end() && iter->second != nullptr ? iter->second : src_node;
}

namespace {
void MarkMultiVersionScope(AObject *vobj) {
  MS_EXCEPTION_IF_NULL(vobj);
  if (vobj->GetScope() != AObject::Scope::SCOPE_NOT_SPECIFIED) {
    return;
  }
  auto pre = vobj->GetPreVersion();
  if (pre != nullptr) {
    MarkMultiVersionScope(const_cast<AObject *>(pre));
    vobj->SetScope(pre->GetScope());
  } else {
    MS_LOG(INFO) << vobj->ToString() << " has not set scope.";
    vobj->SetScope(AObject::Scope::SCOPE_LOCAL);
  }
}

void FillVersionNodeMap(const std::vector<ValueNode *> &nodes, std::map<const AObject *, ValueNode *> *map,
                        bool use_first) {
  MS_EXCEPTION_IF_NULL(map);
  for (auto node : nodes) {
    auto vobj = node->GetOwnVobj();
    if (map->find(vobj) == map->end()) {
      continue;
    }
    if (use_first && map->at(vobj) != nullptr) {
      continue;
    }
    (*map)[vobj] = node;
  }
}

std::vector<ValueNode *> EliminateWeightsSideEffect(const std::vector<ValueNode *> &nodes) {
  std::vector<ValueNode *> side_effect_nodes(nodes);
  auto is_remove = [](const auto &node) {
    if (node->GetOpcode() != STORE_ATTR) {
      return false;
    }
    auto obj = node->input(1)->GetOwnVobj()->GetBaseVersion()->GetPyObject();
    auto attr_name = node->GetName().c_str();
    if (obj.ptr() == nullptr || !py::hasattr(obj, attr_name)) {
      return false;
    }
    bool eliminate = IsParameterObject(obj.attr(attr_name));
    if (eliminate) {
      MS_LOG(DEBUG) << "Eliminate Parameter SideEffect node: " << node->ToString();
    }
    return eliminate;
  };
  auto remove_if = std::remove_if(side_effect_nodes.begin(), side_effect_nodes.end(), is_remove);
  side_effect_nodes.erase(remove_if, side_effect_nodes.end());
  return side_effect_nodes;
}

bool IsEnableSubGraphBreakOptimize(const Graph *graph) {
#if IS_PYTHON_3_11_PLUS
  return false;
#else
  return graph->Config().GetBoolConfig(GraphJitConfig::kSubgraphBreakOpt) &&
         common::GetCompileConfig("PIJIT_SUBGRAPH_BREAK_OPTIMIZE") != "0";
#endif
}
}  // namespace

void SideEffectHandler::Run() {
  auto stop_bci = graph_->GetStopTraceBci();
  // side effect handler has already run, and no change
  if (stop_bci == break_bci_) {
    return;
  }
  // Not First Time
  if (break_bci_ != INT_MIN) {
    ResetRunningEnvironment();
  }
  break_bci_ = stop_bci;
  inputs_ = CollectCapturedInputs();
  nodes_ = CollectCapturedNodes();
  ScopeAnalysis();
  GroupCapturedNodes();
  auto vars = CollectModifiedExternalVariables();
  InitializeVersionNodeMaps(vars);
  auto nodes = CollectSideEffectOperations();
  nodes = RebaseObjectVersionInSideEffects(nodes);
  nodes = CorrectVariableOfStoreGlobal(nodes);
  nodes = EliminateRedundantSideEffect(nodes);
  side_effect_nodes_ = MergeSideEffect(nodes);
}

std::vector<ValueNode *> SideEffectHandler::GetSideEffectInputs() const {
  std::set<ValueNode *> inputs;
  for (const auto &node : side_effect_nodes_) {
    inputs.insert(node->inputs().begin(), node->inputs().end());
  }
  std::vector<ValueNode *> side_effect_inputs(inputs.begin(), inputs.end());
  return side_effect_inputs;
}

std::vector<ValueNode *> SideEffectHandler::OptimizeSideEffect(const std::vector<ValueNode *> &nodes) {
  return EliminateRedundantSideEffect(nodes);
}

void SideEffectHandler::ResetRunningEnvironment() {
  ex_var_base_2_node_.clear();
  ex_var_latest_2_node_.clear();
  side_effect_nodes_.clear();
}

std::vector<ValueNode *> SideEffectHandler::CollectCapturedInputs() const {
  MS_EXCEPTION_IF_NULL(graph_);
  std::vector<ValueNode *> inputs(graph_->GetParameters());
  const auto &locals = graph_->GetFrame(0).GetLocals();
  std::copy_if(locals.begin(), locals.end(), std::back_inserter(inputs),
               [](auto &local) { return local->GetType() == AbstractNode::Param; });
  auto &pre_ops = graph_->prepare().operations_;
  std::copy(pre_ops.begin(), pre_ops.end(), std::back_inserter(inputs));
  return inputs;
}

std::vector<ValueNode *> SideEffectHandler::CollectCapturedNodes() const {
  MS_EXCEPTION_IF_NULL(graph_);
  const auto &nodes = graph_->GetTracedNodes();
  auto break_bci = graph_->GetStopTraceBci();
  if (break_bci == -1) {
    return nodes;
  }
  std::vector<ValueNode *> result;
  auto enabled_opt = IsEnableSubGraphBreakOptimize(graph_);
  for (const auto &node : nodes) {
    if (node->bci() > break_bci || (!enabled_opt && node->bci() == break_bci)) {
      break;
    }
    result.push_back(node);
  }
  return result;
}

void SideEffectHandler::AnalyzeCallNodeScope(CallNode *node) const {
  auto sub_graph = node->GetSubGraph();
  if (sub_graph == nullptr) {
    if (node->GetVobj()->HasMultiVersion()) {
      MarkMultiVersionScope(node->GetOwnVobj());
    } else {
      node->SetScope(AObject::Scope::SCOPE_LOCAL);
    }
  } else {
    auto side_effect_handler = sub_graph->GetSideEffectHandler();
    side_effect_handler->Run();
    if (graph_->GetStopTraceBci() == -1) {
      auto ret = sub_graph->GetRetVal();
      if ((ret->GetScope() & AObject::Scope::SCOPE_FREE_VAR) && ret->GetGraph() == node->GetGraph()) {
        node->SetScope(AObject::Scope::SCOPE_LOCAL);
      } else {
        node->SetScope(ret->GetScope());
      }
    }
  }
}

void SideEffectHandler::AnalyzeNodeScope(ValueNode *node) const {
  using ScopeAnalyzer = std::function<void(ValueNode *)>;

  ScopeAnalyzer build_analyzer = [this](ValueNode *node) { MarkMultiVersionScope(node->GetVobj()); };

  ScopeAnalyzer call_analyzer = [this](ValueNode *node) { AnalyzeCallNodeScope(static_cast<CallNode *>(node)); };

  ScopeAnalyzer global_analyzer = [](ValueNode *node) {
    if (node->GetOpcode() == LOAD_GLOBAL) {
      node->SetScope(AObject::Scope::SCOPE_GLOBAL);
    } else {
      node->inputs().back()->AddScope(AObject::Scope::SCOPE_GLOBAL);
    }
  };

  ScopeAnalyzer load_analyzer = [](ValueNode *node) {
    auto is_free_var = node->GetType() == AbstractNode::Type::FreeVar;
    node->SetScope(is_free_var ? AObject::Scope::SCOPE_FREE_VAR : AObject::Scope::SCOPE_LOCAL);
  };

  ScopeAnalyzer subscr_analyzer = [](ValueNode *node) {
    if (node->GetOpcode() == STORE_SUBSCR) {
      if (node->inputs()[1]->GetScope() & AObject::Scope::SCOPE_GLOBAL) {
        node->inputs().front()->AddScope(AObject::Scope::SCOPE_GLOBAL);
      }
    } else {
      if (node->GetScope() == AObject::Scope::SCOPE_NOT_SPECIFIED) {
        node->SetScope(node->inputs()[0]->GetScope());
      }
    }
  };

  ScopeAnalyzer default_analyzer = [](ValueNode *node) {
    if (node->GetScope() == AObject::Scope::SCOPE_NOT_SPECIFIED) {
      auto is_param = node->GetType() == AbstractNode::Type::Param;
      node->SetScope(is_param ? AObject::Scope::SCOPE_PARAM : AObject::Scope::SCOPE_LOCAL);
    }
  };

  const std::map<int, ScopeAnalyzer> scope_analyzer_map = {
    {BUILD_LIST, build_analyzer},
    {BUILD_TUPLE, build_analyzer},
    {BUILD_SET, build_analyzer},
    {BUILD_MAP, build_analyzer},
    {CALL, call_analyzer},
    {CALL_FUNCTION, call_analyzer},
    {CALL_FUNCTION_EX, call_analyzer},
    {CALL_FUNCTION_KW, call_analyzer},
    {LOAD_GLOBAL, global_analyzer},
    {STORE_GLOBAL, global_analyzer},
    {LOAD_NAME, load_analyzer},
    {LOAD_DEREF, load_analyzer},
    {LOAD_CLOSURE, load_analyzer},
    {LOAD_CLASSDEREF, load_analyzer},
    {STORE_SUBSCR, subscr_analyzer},
    {LOAD_ATTR, subscr_analyzer},
    {BINARY_SUBSCR, subscr_analyzer},
  };

  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(DEBUG) << "Analyze Scope for " << node->ToString();
  auto opcode = node->GetOpcode();
  auto anayzer =
    scope_analyzer_map.find(opcode) == scope_analyzer_map.end() ? default_analyzer : scope_analyzer_map.at(opcode);
  anayzer(node);
  MS_LOG(DEBUG) << node->GetScopeDesc() << " : " << node->ToString();
}

void SideEffectHandler::ScopeAnalysis() const {
  std::for_each(inputs_.begin(), inputs_.end(), [](auto &input) {
    input->SetScope(AObject::Scope::SCOPE_PARAM);
    MS_LOG(DEBUG) << input->GetScopeDesc() << " : " << input->ToString();
  });
  std::for_each(nodes_.begin(), nodes_.end(), [this](auto &node) { AnalyzeNodeScope(node); });
}

void SideEffectHandler::GroupCapturedNodes() const {
  for (auto iter = nodes_.begin(); iter != nodes_.end(); iter++) {
    auto node = *iter;
    if (!node->IsSideEffectNode()) {
      node->MarkVmGraphNode();
    } else {
      auto map = graph_->GetSideEffect()->data()->modified_and_replaced_map();
      if (map.find(node) != map.end()) {
        node->MarkGraphNode();
      } else {
        node->MarkVmNode();
      }
    }
  }
}

std::vector<ValueNode *> SideEffectHandler::CollectModifiedExternalVariables() const {
  std::vector<ValueNode *> side_effect_vars;
  for (auto &node : nodes_) {
    auto scope = node->GetScope();
    if (scope == AObject::Scope::SCOPE_LOCAL) {
      continue;
    }
    auto vobj = node->GetVobj();
    if (vobj == nullptr) {
      continue;
    }
    if (!vobj->HasMultiVersion()) {
      continue;
    }
    MS_LOG(DEBUG) << "Collect Side-Effect Var : " << node->ToString();
    side_effect_vars.push_back(node);
  }
  return side_effect_vars;
}

std::vector<ValueNode *> SideEffectHandler::CollectSideEffectOperations() const {
  std::vector<ValueNode *> side_effect_ops;
  for (auto &node : nodes_) {
    auto opcode = node->GetOpcode();
    if (opcode == STORE_FAST) {
      continue;
    }
    if ((opcode == STORE_SUBSCR || opcode == STORE_ATTR) &&
        node->inputs()[1]->GetScope() == AObject::Scope::SCOPE_LOCAL) {
      MS_LOG(DEBUG) << "Skip local node: " << node->ToString();
      continue;
    }
    if (Opcode(opcode).IsCall()) {
      auto sub_graph = static_cast<CallNode *>(node)->GetSubGraph();
      if (sub_graph != nullptr) {
        auto handler = sub_graph->GetSideEffectHandler();
        handler->Run();
        auto nodes = handler->GetSideEffect();
        std::copy(nodes.begin(), nodes.end(), std::back_inserter(side_effect_ops));
      }
    }
    if (!node->IsSideEffectNode()) {
      continue;
    }
    if (node->GetScope() == AObject::Scope::SCOPE_LOCAL) {
      MS_LOG(DEBUG) << "Eliminate local scope SideEffect node: " << node->ToString();
      continue;
    }
    MS_LOG(DEBUG) << "Collect Side-Effect Operation : [" << node->GetScopeDesc() << "] " << node->ToString();
    side_effect_ops.push_back(node);
  }
  return side_effect_ops;
}

void SideEffectHandler::InitializeVersionNodeMaps(const std::vector<ValueNode *> &vars) {
  for (const auto &var : vars) {
    auto base = var->GetOwnVobj()->GetBaseVersion();
    ex_var_base_2_node_[base] = nullptr;
    auto latest = var->GetVobj()->GetLatestVersion();
    ex_var_latest_2_node_[latest] = nullptr;
  }
  FillVersionNodeMap(inputs_, &ex_var_base_2_node_, true);
  FillVersionNodeMap(nodes_, &ex_var_base_2_node_, true);
  FillVersionNodeMap(nodes_, &ex_var_latest_2_node_, false);
}

void SideEffectHandler::RebaseObjectVersion(CallNode *call_node) const {
  MS_EXCEPTION_IF_NULL(call_node);
  auto callable = call_node->inputs().front();
  auto vobj = callable->GetVobj();
  MS_EXCEPTION_IF_NULL(vobj);
  auto obj = vobj->GetPyObject().ptr();
  MS_EXCEPTION_IF_NULL(obj);
  auto is_method = PyMethod_Check(obj) || (PyCFunction_Check(obj) && PyCFunction_GET_SELF(obj) != nullptr);
  auto op = callable->GetOpcode();
  auto callable_check = !is_method || op == LOAD_ATTR || op == LOAD_METHOD;
  MS_EXCEPTION_IF_CHECK_FAIL(callable_check, "Should be a func or LoadNode, but got ." + callable->ToString());
  auto &operand = is_method ? callable->inputs().front() : call_node->inputs()[1];
  auto base = operand->GetOwnVobj()->GetBaseVersion();
  if (ex_var_base_2_node_.find(base) == ex_var_base_2_node_.end()) {
    return;
  }
  operand = ex_var_base_2_node_.at(base);
}

std::vector<ValueNode *> SideEffectHandler::RebaseObjectVersionInSideEffects(
  const std::vector<ValueNode *> &side_effect_nodes) const {
  for (auto &side_effect_node : side_effect_nodes) {
    auto opcode = side_effect_node->GetOpcode();
    if (Opcode(opcode).IsCall()) {
      RebaseObjectVersion(static_cast<CallNode *>(side_effect_node));
    } else {
      auto has_obj = opcode == DELETE_ATTR || opcode == STORE_ATTR || opcode == DELETE_SUBSCR || opcode == STORE_SUBSCR;
      auto index = (opcode == DELETE_ATTR || opcode == DELETE_SUBSCR) ? 0 : 1;
      auto base = has_obj ? side_effect_node->inputs()[index]->GetVobj()->GetBaseVersion() : nullptr;
      if (ex_var_base_2_node_.find(base) != ex_var_base_2_node_.end()) {
        side_effect_node->inputs()[index] = ex_var_base_2_node_.at(base);
      }
    }
  }
  return std::move(side_effect_nodes);
}

std::vector<ValueNode *> SideEffectHandler::CorrectVariableOfStoreGlobal(const std::vector<ValueNode *> &nodes) const {
  std::vector<ValueNode *> side_effect_nodes(nodes.begin(), nodes.end());
  std::for_each(side_effect_nodes.begin(), side_effect_nodes.end(), [this](auto &side_effect_node) {
    if (side_effect_node->GetOpcode() != STORE_GLOBAL) {
      return;
    }
    auto graph = side_effect_node->GetGraph();
    std::string module_name = graph->GetModuleName();
    if (module_name == graph_->GetModuleName()) {
      return;
    }
    MS_LOG(DEBUG) << "Side Effect operation from " << graph->GetCodeName() << " : " << side_effect_node->ToString();
    py::object obj = py::reinterpret_steal<py::object>(PyImport_ImportModule(module_name.c_str()));
    auto load = graph_->NewValueNode(AObject::Convert(obj), LOAD_CONST, -1, {});
    auto var = side_effect_node->inputs().front();
    side_effect_node = graph_->NewValueNode(nullptr, STORE_ATTR, 0, {var, load}, side_effect_node->GetName());
  });
  return side_effect_nodes;
}

std::vector<ValueNode *> SideEffectHandler::EliminateRedundantSideEffect(const std::vector<ValueNode *> &nodes) {
  // The cache for optimizing store subscr.
  std::map<const AObject *, std::map<std::string, size_t>> var_2_op;
  // The cache for optimizing store deref.
  std::map<const AObject *, std::map<std::string, size_t>> deref;
  // The cache for optimizing store global and store name.
  std::map<const AObject *, std::map<std::string, size_t>> name_2_op;
  // The cache for optimizing store attr.
  std::map<const AObject *, std::map<std::string, size_t>> attr_2_op;
  const std::map<int, std::map<const AObject *, std::map<std::string, size_t>> *> cache_map = {
    {DELETE_DEREF, &deref},     {STORE_DEREF, &deref},    {DELETE_GLOBAL, &name_2_op}, {STORE_GLOBAL, &name_2_op},
    {DELETE_NAME, &name_2_op},  {STORE_NAME, &name_2_op}, {DELETE_ATTR, &attr_2_op},   {STORE_ATTR, &attr_2_op},
    {DELETE_SUBSCR, &var_2_op}, {STORE_SUBSCR, &var_2_op}};
  std::vector<ValueNode *> side_effect_nodes(nodes);
  for (size_t index = 0; index < side_effect_nodes.size(); ++index) {
    auto node = side_effect_nodes[index];
    auto opcode = node->GetOpcode();
    if (!node->IsSideEffectNode() || Opcode(opcode).IsCall()) {
      continue;
    }
    MS_EXCEPTION_IF_CHECK_FAIL(cache_map.find(opcode) != cache_map.end(),
                               "Should be a side-effect node, but got " + node->ToString());
    auto cache = cache_map.at(opcode);
    auto has_obj = opcode == DELETE_ATTR || opcode == STORE_ATTR || opcode == DELETE_SUBSCR || opcode == STORE_SUBSCR;
    auto idx = (opcode == DELETE_ATTR || opcode == DELETE_SUBSCR) ? 0 : 1;
    auto base = has_obj ? node->inputs()[idx]->GetVobj()->GetBaseVersion() : nullptr;
    auto is_int = opcode == DELETE_DEREF || opcode == STORE_DEREF || opcode == DELETE_SUBSCR || opcode == STORE_SUBSCR;
    auto arg = is_int ? std::to_string(node->GetOparg()) : node->GetName();
    if (cache->find(base) != cache->end() && cache->at(base).find(arg) != cache->at(base).end()) {
      MS_LOG(DEBUG) << "Eliminate redundant SideEffect node: " << ToString(side_effect_nodes[cache->at(base).at(arg)]);
      side_effect_nodes[cache->at(base).at(arg)] = nullptr;
    }
    (*cache)[base][arg] = index;
  }
  auto is_remove = [](const auto &node) { return node == nullptr; };
  auto remove_if = std::remove_if(side_effect_nodes.begin(), side_effect_nodes.end(), is_remove);
  side_effect_nodes.erase(remove_if, side_effect_nodes.end());
  return EliminateWeightsSideEffect(side_effect_nodes);
}

std::vector<ValueNode *> SideEffectHandler::MergeSideEffect(const std::vector<ValueNode *> &nodes) const {
  // All the side effect nodes that maybe present optimization opportunity.
  std::map<const AObject *, std::vector<size_t>> to_be_opt;
  std::vector<ValueNode *> side_effect_nodes(nodes.begin(), nodes.end());
  for (size_t index = 0; index < side_effect_nodes.size(); ++index) {
    auto opcode = side_effect_nodes[index]->GetOpcode();
    if (Opcode(opcode).IsCall()) {
      auto base = side_effect_nodes[index]->GetOwnVobj()->GetBaseVersion();
      if (base->GetType() == AObject::kTypeTensor) {
        to_be_opt[base].push_back(index);
      }
    } else {
      auto has_obj = opcode == DELETE_ATTR || opcode == STORE_ATTR || opcode == DELETE_SUBSCR || opcode == STORE_SUBSCR;
      auto idx = (opcode == DELETE_ATTR || opcode == DELETE_SUBSCR) ? 0 : 1;
      auto base = has_obj ? side_effect_nodes[index]->inputs()[idx]->GetVobj()->GetBaseVersion() : nullptr;
      if (base != nullptr && base->GetType() == AObject::kTypeTensor) {
        to_be_opt[base].push_back(index);
      }
    }
  }
  for (auto &[vobj, indexes] : to_be_opt) {
    if (indexes.size() <= 1) {
      continue;
    }
    auto latest = vobj->GetLatestVersion();
    auto check = ex_var_latest_2_node_.find(latest) != ex_var_latest_2_node_.end();
    MS_EXCEPTION_IF_CHECK_FAIL(check, "Can't find a value for " + latest->ToString());
    auto latest_value = ex_var_latest_2_node_.at(latest);
    check = ex_var_base_2_node_.find(vobj) != ex_var_base_2_node_.end();
    MS_EXCEPTION_IF_CHECK_FAIL(check, "Can't find a value for " + vobj->ToString());
    auto obj = ex_var_base_2_node_.at(vobj);
    auto none = graph_->NewValueNode(AObject::Convert(Py_None), LOAD_CONST, -1, {});
    auto slice_obj = py::reinterpret_steal<py::object>(PySlice_New(Py_None, Py_None, nullptr));
    auto slice = graph_->NewValueNode(AObject::Convert(slice_obj), BUILD_SLICE, 2, {none, none});
    auto save = graph_->NewValueNode(nullptr, STORE_SUBSCR, 3, {latest_value, obj, slice});
    for (const auto index : indexes) {
      MS_LOG(INFO) << "Delete side effect operation: " << ToString(side_effect_nodes[index]);
      side_effect_nodes[index] = nullptr;
    }
    side_effect_nodes[indexes.back()] = save;
    MS_LOG(INFO) << "Merge side effect operation: " << save->ToString();
  }
  auto is_remove = [](const auto &node) { return node == nullptr; };
  auto remove_if = std::remove_if(side_effect_nodes.begin(), side_effect_nodes.end(), is_remove);
  side_effect_nodes.erase(remove_if, side_effect_nodes.end());
  return side_effect_nodes;
}
}  // namespace pijit
}  // namespace mindspore
