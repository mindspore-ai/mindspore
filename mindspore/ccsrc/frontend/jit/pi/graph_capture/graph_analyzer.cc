/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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
#include "frontend/jit/pi/graph_capture/graph_analyzer.h"

#include <algorithm>
#include <iterator>
#include <list>
#include <optional>
#include <memory>
#include <utility>
#include <unordered_set>
#include <stack>
#include <string>
#include <vector>

#include "frontend/jit/pi/capture_context.h"
#include "frontend/jit/pi/pi_jit_config.h"
#include "frontend/jit/pi/graph_guard/infer.h"
#include "frontend/jit/pi/graph_capture/graph.h"
#include "frontend/jit/pi/graph_capture/graph_build.h"
#include "frontend/jit/pi/graph_capture/side_effect.h"
#include "frontend/jit/pi/graph_build/build_graph_utils.h"
#include "frontend/jit/pi/graph_capture/graph_arguments_optimizer.h"
#include "utils/compile_config.h"
#include "utils/convert_utils_base.h"

#define ADD_NODE(container, node)                                                   \
  do {                                                                              \
    auto _tmp_node = (node);                                                        \
    (container).push_back(_tmp_node);                                               \
    MS_LOG(INFO) << "Add node to " #container " [" << _tmp_node->ToString() << "]"; \
  } while (0)

namespace mindspore {
namespace pijit {
namespace {
void CollectAllSubGraphs(const Graph *root, int break_bci, std::unordered_set<const Graph *> *sub_graphs) {
  for (ValueNode *node : root->GetTracedNodes()) {
    if (node->GetType() == ValueNode::Call && (break_bci == -1 || node->bci() < break_bci)) {
      auto call_node = static_cast<CallNode *>(node);
      if (call_node->GetSubGraph() != nullptr) {
        Graph *sub_graph = call_node->GetSubGraph();
        sub_graphs->insert(sub_graph);
        CollectAllSubGraphs(sub_graph, sub_graph->GetStopTraceBci(), sub_graphs);
      }
    }
  }
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

void GraphAnalyzer::CapturedInfo::Info::clear() {
  values.clear();
  inputs.clear();
  operations.clear();
  outputs.clear();
}

void GraphAnalyzer::CapturedInfo::clear() {
  captured_.clear();
  interpret_.clear();
  outputs_optimize_.clear();
}

std::string GraphAnalyzer::CapturedInfo::Info::ToString() {
  std::stringstream s;
  s << "inputs: {" << std::endl;
  for (auto i : inputs) {
    s << "  " << i->ToString() << std::endl;
  }
  s << "}" << std::endl;
  s << "operations: {" << std::endl;
  for (auto i : operations) {
    s << "  " << i->ToString() << std::endl;
  }
  s << "}" << std::endl;
  s << "outputs: {" << std::endl;
  for (auto i : outputs) {
    s << "  " << i->ToString() << std::endl;
  }
  s << "}" << std::endl;
  return s.str();
}

std::string GraphAnalyzer::CapturedInfo::ToString() {
  std::stringstream s;
  s << "1. captured_ info: \n";
  s << captured_.ToString();
  s << "2. outputs_optimize_ info: \n";
  s << outputs_optimize_.ToString();
  s << "3. interpret_ info: \n";
  s << interpret_.ToString();
  s << "4. has_grad_: " << has_grad_ << "\n";
  return s.str();
}

namespace {
void UpdateBreakInfo(Graph *graph) {
  Graph::BreakInfo info;
  int bci = graph->GetStopTraceBci();
  info.bci_ = bci;
  info.reason_ = graph->GetStopTraceReason();
  info.alive_nodes_ = graph->CollectAliveNode(graph->GetStopTraceBci(), &info.alive_locals_);
  if (info.bci_ != -1) {
    MS_EXCEPTION_IF_CHECK_FAIL(bci >= 0 && bci < SizeToInt(graph->GetCFG()->instr_pool().size()),
                               "bci error: " + std::to_string(bci));
    info.break_point_ = graph->GetCFG()->instr_pool()[bci].get();

    auto &nodes = graph->GetTracedNodes();
    auto it = std::find_if(nodes.begin(), nodes.end(), [bci](auto *node) { return node->bci() == bci; });
    info.break_point_node_ = it == nodes.end() ? nullptr : *it;
  }
  graph->set_break_info(info);
}
}  // namespace

void GraphAnalyzer::BeforeAnalyze() {
#if IS_PYTHON_3_11_PLUS
  Graph *g = this->graph_;
  int break_bci = g->GetStopTraceBci();
  if (break_bci == -1) {
    return;
  }
  auto iter = g->GetCFG()->FindTryWithBlock(break_bci);
  if (iter == g->GetCFG()->exc_table().end()) {
    return;
  }
  int new_break_bci = iter->second.begin_;
  if (new_break_bci != 0 && g->GetCFG()->instr_pool()[new_break_bci - 1]->op() == BEFORE_WITH) {
    new_break_bci--;
  }
  MS_LOG(INFO) << "break at try/with block, reset break bci to try/with syntax start: " << break_bci << " -> "
               << new_break_bci;
  g->StopTraceAt(new_break_bci, g->GetStopTraceReason());
#endif
}

void GraphAnalyzer::GraphArgumentOpt() {
  if (graph_->Config().GetBoolConfig(GraphJitConfig::kInterpretCapturedCode)) {
    return;
  }
  if (!graph_->Config().GetBoolConfig(GraphJitConfig::kExpandGraphInput)) {
    return;
  }
  if (!graph_->Config().GetBoolConfig(GraphJitConfig::kEliminateRedundantArgs)) {
    return;
  }
  auto optimizer = GraphArgumentOptimizer::GetNewInstance(graph_);
  optimizer->SetGraphNodes(GetCaptureInfo().captured_.operations);
  optimizer->Run({GetCaptureInfo().captured_.outputs});
  auto inputs = GetCaptureInfo().captured_.inputs;
  GetCaptureInfo().captured_.inputs = optimizer->GetArguments();
  for (const auto &input : GetCaptureInfo().outputs_optimize_.inputs) {
    if (std::find(inputs.begin(), inputs.end(), input) != inputs.end()) {
      graph_->PrepareParameter(input);
    }
  }
}

void GraphAnalyzer::Analyze() {
  MS_LOG(INFO) << "Start graph analyze";
  BeforeAnalyze();
  auto collect_trace_nodes = [this]() {
    const auto &nodes = graph_->GetTracedNodes();
    if (graph_->GetStopTraceBci() == -1) {
      return nodes;
    }
    std::vector<ValueNode *> result;
    for (const auto &node : nodes) {
      if (node->bci() >= graph_->GetStopTraceBci()) {
        break;
      }
      result.push_back(node);
    }
    return result;
  };

  CollectClosureSideEffect();

  // assume all values is captured to func_graph
  GetCaptureInfo().captured_.inputs = graph_->prepare().inputs_;
  GetCaptureInfo().captured_.operations = collect_trace_nodes();
  UseDefAnalyze();
  auto func_graph_builder = graph_builder_->FGBuilder();
  if (func_graph_builder->graph() == nullptr) {
    // Graph build failed, add all nodes to ordered_escaped_locals.
    PyCodeWrapper co(graph_->GetCodeObj());
    auto origin_stop_bci = graph_->GetStopTraceBci();
    if (origin_stop_bci == -1) {
      MS_LOG(INFO) << "no graph in " << py::str(reinterpret_cast<PyObject *>(co.ptr()));
    } else {
      MS_LOG(INFO) << "no graph captured, trace break at " << co.FileName() << ", line "
                   << PyCode_Addr2Line(co.ptr(), origin_stop_bci);
    }
    if (graph_break_info_.is_break_at_call && !graph_break_info_.captured_subgraphs.empty()) {
      GetCaptureInfo().captured_.clear();
    } else {
      graph_->StopTraceAt(origin_stop_bci, StopTraceReason::kStopTraceNoGraphCaptured);
      UpdateBreakInfo(graph_);
      need_interpret_ = true;
      GetCaptureInfo().clear();

      GetCaptureInfo().interpret_.inputs = graph_->GetFrame(0).GetLocals();
      GetCaptureInfo().interpret_.operations = collect_trace_nodes();
      GetCaptureInfo().interpret_.outputs = graph_->break_info().alive_nodes_;
      return;
    }
  }

  CollectCapturedAndInterpret();

  GraphArgumentOpt();

  need_interpret_ = true;
  if (graph_->GetStopTraceBci() != -1 || !GetCaptureInfo().interpret_.operations.empty()) {
    return;
  }
  bool support_ret = graph_->GetRetVal()->GetVobj() && graph_->GetRetVal()->GetVobj()->IsMindSporeSupportedType();
  if (!support_ret) {
    return;
  }
  int param_index = 0;
  for (const auto &node : GetCaptureInfo().captured_.inputs) {
    if (node->GetType() != ValueNode::Param /* LOAD_DEREF, LOAD_GLOBAL, LOAD_ATTR, BINARY_SUBSCR */
        || node->GetOparg() != param_index /* missing arguments */) {
      return;
    }
    param_index++;
  }
  need_interpret_ = !graph_->GetSideEffect()->IsEmpty() || !GetCaptureInfo().outputs_optimize_.operations.empty() ||
                    param_index != graph_->GetCodeObj()->co_argcount;
  MS_LOG(INFO) << "End graph analyze";
}

void GraphAnalyzer::CollectClosureSideEffect() {
  if (graph_->GetFrame(0).GetClosures().empty()) {
    return;
  }
  const std::vector<CellVarNode *> &closures = graph_->GetFrame(0).GetClosures();
  std::vector<CellVarNode *> nodes;
  if (graph_->GetStopTraceBci() == -1) {
    // If no graph break, we only need to restore free-variables. Because cell-variables are equivalent to
    // local-variables and do not need to be restored.
    int cellvar_size = PyCodeWrapper(graph_->GetCodeObj()).CellVarsSize();
    nodes.insert(nodes.begin(), closures.begin() + cellvar_size, closures.end());
  } else {
    nodes.insert(nodes.begin(), closures.begin(), closures.end());
  }

  for (const CellVarNode *cell_node : nodes) {
    if (cell_node == &ValueNode::kUnboundLocal || cell_node->GetCellOper().empty()) {
      continue;
    }
    const std::vector<ValueNode *> &cell_ops = cell_node->GetCellOper();
    auto it = std::find_if(cell_ops.rbegin(), cell_ops.rend(), [](ValueNode *node) {
      return node->GetOpcode() == STORE_DEREF || node->GetOpcode() == DELETE_DEREF;
    });
    if (it == cell_ops.rend() || (*it)->GetOpcode() == DELETE_DEREF) {
      // Currently, the recovery of DELETE_DEREF is not supported.
      continue;
    }
    graph_->GetSideEffect()->Record(*it);
  }
}

namespace {
// check whether the node can be added to the output of the graph
// or can be added to the output of the graph through transformation
// support : none, scalar, tensor, tuple, list, dict, and combination during them
inline bool IsValidGraphOutput(const AbstractBasePtr &abstract) {
  if (abstract == nullptr) {
    return false;
  }
  if (abstract->isa<abstract::AbstractSequence>()) {
    const auto elements = abstract->cast<abstract::AbstractSequencePtr>()->elements();
    return std::all_of(elements.begin(), elements.end(), IsValidGraphOutput);
  }
  if (abstract->isa<abstract::AbstractDictionary>()) {
    const auto elements = abstract->cast<abstract::AbstractDictionaryPtr>()->elements();
    return std::all_of(elements.begin(), elements.end(), [](const abstract::AbstractElementPair &elem) {
      return IsValidGraphOutput(elem.first) && IsValidGraphOutput(elem.second);
    });
  }
  return IsValidOutputAbstractScalar(abstract) || IsValidOutputAbstractTensor(abstract) ||
         abstract->isa<abstract::AbstractNone>();
}

inline bool IsValidOutput(const ValueNode *node) {
  return node != nullptr && node->abstract_wrapper() != nullptr &&
         IsValidGraphOutput(node->abstract_wrapper()->abstract());
}

std::vector<ValueNode *> CollectInputs(const std::vector<ValueNode *> &nodes) {
  std::set<ValueNode *> inputs;
  for (const auto &node : nodes) {
    inputs.insert(node->inputs().begin(), node->inputs().end());
  }
  for (const auto &node : nodes) {
    inputs.erase(node);
  }
  return std::vector<ValueNode *>(inputs.begin(), inputs.end());
}

void ReplaceSequenceNoneElementWithConst(ValueNode *node, Graph *graph) {
  auto opcode = node->GetOpcode();
  if (opcode != BUILD_LIST && opcode != BUILD_TUPLE) {
    return;
  }
  for (auto iter = node->inputs().begin(); iter != node->inputs().end(); iter++) {
    auto abstract_wrapper = (*iter)->abstract_wrapper();
    MS_EXCEPTION_IF_NULL(abstract_wrapper);
    auto abstract = abstract_wrapper->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    if (abstract->isa<abstract::AbstractNone>()) {
      *iter = graph->NewValueNode(AObject::Convert(Py_None), LOAD_CONST, 0, {});
      (*iter)->set_abstract_wrapper(abstract_wrapper);
    }
  }
}

void UpdateUseDefOrder(std::vector<ValueNode *> *nodes) {
  MS_EXCEPTION_IF_NULL(nodes);
  std::list<ValueNode *> node_list(nodes->begin(), nodes->end());
  nodes->clear();
  while (!node_list.empty()) {
    auto front = node_list.front();
    node_list.pop_front();
    auto inputs = front->inputs();
    auto independent = std::all_of(inputs.begin(), inputs.end(), [&node_list](const auto &input) {
      return std::find(node_list.begin(), node_list.end(), input) == node_list.end();
    });
    if (inputs.empty() || independent) {
      nodes->push_back(front);
    } else {
      node_list.push_back(front);
    }
  }
}
}  // namespace

ValueNode *GraphAnalyzer::MutateSequenceNode(ValueNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto abstract_wrapper = node->abstract_wrapper();
  MS_EXCEPTION_IF_NULL(abstract_wrapper);
  auto abstract = abstract_wrapper->abstract();
  MS_EXCEPTION_IF_NULL(abstract);
  if (!abstract->isa<abstract::AbstractSequence>()) {
    return node;
  }
  auto &captured = GetCaptureInfo().captured_;
  auto sequence = abstract->cast<abstract::AbstractSequencePtr>();
  auto func_graph_builder = graph_builder_->FGBuilder();
  auto graph_node = func_graph_builder->FindOrCreateNodeByWrapper(abstract_wrapper);
  auto func_graph = func_graph_builder->graph(true);
  bool is_tuple = abstract->isa<abstract::AbstractTuple>();

  auto mutated_node = graph_->NewValueNode(node->GetVobj(), is_tuple ? BUILD_TUPLE : BUILD_LIST, sequence->size(), {});
  mutated_node->set_abstract_wrapper(std::make_shared<AbstractWrapper>(abstract));  // used to remove duplicate data
  auto prim = is_tuple ? prim::kPrimTupleGetItem : prim::kPrimListGetItem;
  for (size_t index = 0; index < sequence->size(); index++) {
    auto graph_index = NewValueNode(SizeToLong(index));
    graph_index->set_abstract(GetValueNode(graph_index)->ToAbstract());
    auto graph_item = func_graph->NewCNodeInOrder(prim, {graph_node, graph_index});
    graph_item->set_abstract(sequence->elements()[index]);
    auto item_abstract_wrapper = std::make_shared<AbstractWrapper>(sequence->elements()[index]);
    auto index_abstract_wrapper = std::make_shared<AbstractWrapper>(graph_index->abstract());
    func_graph_builder->AddLocalVariableNode(item_abstract_wrapper, graph_item);
    func_graph_builder->AddLocalVariableNode(index_abstract_wrapper, graph_index);

    auto bc_index = graph_->NewValueNode(AObject::Convert(py::int_(index)), LOAD_CONST, -1, {});
    bc_index->set_abstract_wrapper(index_abstract_wrapper);
    auto vobj_item = static_cast<AbstractSequence *>(node->GetOwnVobj())->GetItem(bc_index->GetOwnVobj());
    auto bc_item = graph_->NewValueNode(vobj_item, BINARY_SUBSCR, 0, {node, bc_index});
    bc_item->set_abstract_wrapper(item_abstract_wrapper);

    bc_index->MarkGraphNode();
    ADD_NODE(captured.operations, bc_index);
    bc_item->MarkGraphNode();
    ADD_NODE(captured.operations, bc_item);
    mutated_node->AddInput(bc_item);
  }
  ReplaceSequenceNoneElementWithConst(mutated_node, graph_);
  GetCaptureInfo().replaced_nodes_[node] = mutated_node;
  return mutated_node;
}

ValueNode *GraphAnalyzer::MutateNamedtupleNode(ValueNode *tuple_node, ValueNode *namedtuple_node) {
  MS_LOG(DEBUG) << "Start mutate namedtuple node, origin namedtuple node: " << namedtuple_node->ToString()
                << ", tuple node: " << tuple_node->ToString();
  MS_EXCEPTION_IF_NULL(namedtuple_node->GetVobj());
  // Delete the abstract wrapper, then it will be executed in pynative.
  tuple_node->set_abstract_wrapper(nullptr);
  auto namedtuple_type = reinterpret_cast<PyObject *>(namedtuple_node->GetVobj()->GetTypeObject());
  MS_EXCEPTION_IF_NULL(namedtuple_type);
  auto method_node = graph_->NewValueNode(AObject::Convert(namedtuple_type), LOAD_CONST, -1, {});

  // Create namedtuple, it will be executed in pynative. eval 'namedtuple(*tuple)' expression.
  ValueNode *mutated_node = graph_->NewCallNode(CALL_FUNCTION_EX, 0, {method_node, tuple_node});
  mutated_node->SetVobj(namedtuple_node->GetVobj());  // used to remove duplicate data
  GetCaptureInfo().replaced_nodes_[namedtuple_node] = mutated_node;
  // add to interpret
  return mutated_node;
}

// return keys and values
std::pair<ValueNode *, ValueNode *> GraphAnalyzer::MutateDictNode(ValueNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto abstract_wrapper = node->abstract_wrapper();
  MS_EXCEPTION_IF_NULL(abstract_wrapper);
  auto abstract = abstract_wrapper->abstract();
  MS_EXCEPTION_IF_NULL(abstract);
  if (!abstract->isa<abstract::AbstractDictionary>()) {
    return std::make_pair(node, nullptr);
  }
  auto &captured = GetCaptureInfo().captured_;
  auto &interpret = GetCaptureInfo().interpret_;
  auto &outputs_optimize = GetCaptureInfo().outputs_optimize_;

  auto dict = abstract->cast<abstract::AbstractDictionaryPtr>();
  AbstractBasePtrList key_abstracts;
  AbstractBasePtrList value_abstracts;
  std::for_each(dict->elements().begin(), dict->elements().end(),
                [&key_abstracts, &value_abstracts](const abstract::AbstractElementPair &ele) {
                  key_abstracts.push_back(ele.first);
                  value_abstracts.push_back(ele.second);
                });
  auto keys_wrapper = std::make_shared<AbstractWrapper>(std::make_shared<abstract::AbstractTuple>(key_abstracts));
  auto values_wrapper = std::make_shared<AbstractWrapper>(std::make_shared<abstract::AbstractTuple>(value_abstracts));
  auto func_graph_builder = graph_builder_->FGBuilder();
  auto graph_node = func_graph_builder->FindOrCreateNodeByWrapper(abstract_wrapper);
  MS_EXCEPTION_IF_NULL(graph_node);
  auto func_graph = func_graph_builder->graph(true);
  auto keys = func_graph->NewCNodeInOrder(prim::kPrimDictGetKeys, {graph_node});
  auto values = func_graph->NewCNodeInOrder(prim::kPrimDictGetValues, {graph_node});
  keys->set_abstract(keys_wrapper->abstract());
  values->set_abstract(values_wrapper->abstract());
  func_graph_builder->AddLocalVariableNode(keys_wrapper, keys);
  func_graph_builder->AddLocalVariableNode(values_wrapper, values);

  // find unique node for builtin method
  ValueNode *dict_keys_method = GetBuiltinMethodNode(&captured.operations, "keys", "dict");
  ValueNode *dict_values_method = GetBuiltinMethodNode(&captured.operations, "values", "dict");
  ValueNode *zip_method = GetBuiltinMethodNode(&interpret.operations, "zip");    // always interpret
  ValueNode *dict_method = GetBuiltinMethodNode(&interpret.operations, "dict");  // always interpret
  // use 'dict(zip(dict.keys(obj), dict.value(obj)))' to restore dict
  // consider use expression 'dict(dict.items(obj))'
  auto call_op_code = IS_PYTHON_3_11_PLUS ? CALL : CALL_FUNCTION;
  auto bc_keys = graph_->NewCallNode(call_op_code, 1, {dict_keys_method, node});
  bc_keys->set_abstract_wrapper(keys_wrapper);
  bc_keys->UpdateVobj();
  auto bc_values = graph_->NewCallNode(call_op_code, 1, {dict_values_method, node});
  bc_values->set_abstract_wrapper(values_wrapper);
  bc_values->UpdateVobj();
  auto call_zip = graph_->NewCallNode(call_op_code, 2, {zip_method, bc_keys, bc_values});
  auto make_dict = graph_->NewCallNode(call_op_code, 1, {dict_method, call_zip});
  make_dict->set_abstract_wrapper(std::make_shared<AbstractWrapper>(abstract));
  make_dict->SetVobj(node->GetVobj());

  // keys and values is graph values
  bc_keys->MarkGraphNode();
  ADD_NODE(captured.operations, bc_keys);
  bc_values->MarkGraphNode();
  ADD_NODE(captured.operations, bc_values);
  // call zip and call dict is interpret operations
  ADD_NODE(outputs_optimize.operations, make_dict);
  ADD_NODE(outputs_optimize.operations, call_zip);

  GetCaptureInfo().replaced_nodes_[node] = make_dict;
  return std::make_pair(bc_keys, bc_values);
}

namespace {
constexpr auto kPiJitOutputDepthKey = "pi_jit_output_depth";

bool IsNeedExpand(const ValueNode *node) {
  auto wrapper = node->abstract_wrapper();
  MS_EXCEPTION_IF_NULL(wrapper);
  auto abs = wrapper->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  constexpr int allow_tuple_max_depth = 3;
  return abs->has_user_data(kPiJitOutputDepthKey) && *abs->user_data<int>(kPiJitOutputDepthKey) > allow_tuple_max_depth;
}
}  // namespace

void GraphAnalyzer::ExpandGraphOutput() {
  if (!graph_->Config().GetBoolConfig(GraphJitConfig::kExpandGraphOutput)) {
    return;
  }
  std::function<int(const abstract::AbstractBasePtr &)> depth_marker =
    [&depth_marker](const abstract::AbstractBasePtr &abstract) {
      MS_EXCEPTION_IF_NULL(abstract);
      if (!abstract->isa<abstract::AbstractSequence>()) {
        return 0;
      }
      std::vector<int> depths;
      const auto &elements = abstract->cast<abstract::AbstractSequencePtr>()->elements();
      std::transform(elements.begin(), elements.end(), std::back_inserter(depths),
                     [&depth_marker](const auto &element) { return depth_marker(element); });
      auto depth = (depths.empty() ? 0 : *std::max_element(depths.begin(), depths.end())) + 1;
      abstract->set_user_data<int>(kPiJitOutputDepthKey, std::make_shared<int>(depth));
      return depth;
    };
  auto func_graph_builder = graph_builder_->FGBuilder();
  MS_EXCEPTION_IF_NULL(func_graph_builder);
  func_graph_builder->ClearOutputNodes();
  auto &captured = GetCaptureInfo().captured_;
  auto &outputs_optimize = GetCaptureInfo().outputs_optimize_;
  mindspore::CompactSet<ValueNode *> nodes;
  nodes.insert(captured.outputs.begin(), captured.outputs.end());
  captured.outputs.clear();
  for (const auto &node : nodes) {
    auto wrapper = node->abstract_wrapper();
    MS_EXCEPTION_IF_NULL(wrapper);
    auto abs = wrapper->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    (void)depth_marker(abs);
  }
  while (!nodes.empty()) {
    auto node = nodes.pop();
    MS_LOG(DEBUG) << "Start process : " << node->ToString();
    if (!IsNeedExpand(node)) {
      MS_LOG(DEBUG) << "Add to output : " << node->ToString();
      captured.outputs.push_back(node);
      auto succ = func_graph_builder->AddOutput(node->abstract_wrapper(), true);
      MS_EXCEPTION_IF_CHECK_FAIL(succ, "Add " + node->ToString() + " to graph outputs failed.");
      continue;
    }
    MS_LOG(DEBUG) << "Start expand : " << node->ToString();
    auto opcode = node->GetOpcode();
    if (opcode != BUILD_LIST && opcode != BUILD_TUPLE) {
      MS_LOG(DEBUG) << "Start mutate : " << node->ToString();
      node = MutateSequenceNode(node);  // transform to build_list or build_tuple
      MS_LOG(DEBUG) << "After mutate : " << node->ToString();
    }
    ADD_NODE(outputs_optimize.operations, node);
    nodes.insert(node->inputs().begin(), node->inputs().end());
  }
}

bool GraphAnalyzer::AnalyzeTopGraphAliveNodes(const std::vector<ValueNode *> &alive_nodes) {
  auto func_graph_builder = graph_builder_->FGBuilder();
  MS_EXCEPTION_IF_NULL(func_graph_builder);
  func_graph_builder->ClearOutputNodes();
  auto &captured = GetCaptureInfo().captured_;
  captured.outputs.clear();
  auto &outputs_optimize = GetCaptureInfo().outputs_optimize_.operations;

  // use order set as work list
  mindspore::CompactSet<ValueNode *> nodes;
  nodes.insert(alive_nodes.begin(), alive_nodes.end());
  auto side_effect_handler = graph_->GetSideEffectHandler();
  side_effect_handler->Run();
  auto side_effect_inputs = side_effect_handler->GetSideEffectInputs();
  nodes.insert(side_effect_inputs.begin(), side_effect_inputs.end());
  while (!nodes.empty()) {
    auto node = *nodes.begin();
    nodes.erase(nodes.begin());
    MS_LOG(INFO) << "Start analyze : " << node->ToString() << " abs : "
                 << (node->abstract_wrapper() == nullptr ? "nullptr"
                                                         : node->abstract_wrapper()->abstract()->ToString());
    if (NeedSkipAddGraphOutput(node)) {
      continue;
    }
    // add output for func_graph
    if (func_graph_builder->AddOutput(node->abstract_wrapper(), true)) {
      MS_LOG(INFO) << "Add graph output : " << node->ToString();
      ADD_NODE(captured.outputs, node);  // must be equal as FuncGraph outputs
      continue;
    }

    // Every node that appears here should have a corresponding anf node in top func graph.
    // Unfortunately, due to some defectuve side-effect node processing, they do not have
    // This issue must be fixed, just pass-by and reminder here
    // This code will be redundant after the issue fixed.
    auto graph_node = func_graph_builder->FindNodeByWrapper(node->abstract_wrapper());
    bool is_not_in_top_graph = (graph_node == nullptr);
    // it is top graph node but not find in top func_graph

    // Contains data whose type is not supported by the graph, analyze its inputs
    if (!IsValidOutput(node) || is_not_in_top_graph) {
      auto msg = (is_not_in_top_graph ? "Not in top graph node : " : "Invalid output : ");
      MS_LOG(INFO) << msg << node->ToString();
      if (Opcode(node->GetOpcode()).IsCall()) {
        MS_LOG(INFO) << "This call node will executed in pynative : [" << node->ToString() << "]";
      }
      // The side effect node will be handled by side_effect_handler, avoid exec twice.
      if (!node->IsSideEffectNode()) {
        ADD_NODE(outputs_optimize, node);
      }
      nodes.insert(node->inputs().begin(), node->inputs().end());
      continue;
    }
    MS_EXCEPTION_IF_CHECK_FAIL(node->abstract_wrapper() && node->abstract_wrapper()->abstract(),
                               "Error check at IsValidOutput");
    if (node->abstract_wrapper()->abstract()->isa<abstract::AbstractDictionary>()) {
      // try to add keys and value to graph outputs, here is sequence, continue to handle sequence
      auto pair = MutateDictNode(node);
      nodes.insert(pair.first);
      nodes.insert(pair.second);
    } else if (node->abstract_wrapper()->abstract()->isa<abstract::AbstractSequence>()) {
      auto sequence = MutateSequenceNode(node);  // transform to build_list or build_tuple
      if (node->abstract_wrapper()->abstract()->isa<abstract::AbstractNamedTuple>()) {
        // specialize for named tuple, 1: graph output tuple, 2: python reconstruct namedtuple
        ADD_NODE(outputs_optimize, MutateNamedtupleNode(sequence, node));
      }
      ADD_NODE(outputs_optimize, sequence);
      nodes.insert(sequence->inputs().begin(), sequence->inputs().end());
    } else {
      MS_LOG(INTERNAL_EXCEPTION) << "the node can't add graph out and not handle by output optimize, it's missing ["
                                 << node->ToString();
    }
  }
  auto side_effect_nodes = side_effect_handler->GetSideEffect();
  std::for_each(side_effect_nodes.begin(), side_effect_nodes.end(),
                [&outputs_optimize](auto &node) { outputs_optimize.insert(outputs_optimize.begin(), node); });
  outputs_optimize = SideEffectHandler::OptimizeSideEffect(outputs_optimize);
  return true;
}

void GraphAnalyzer::UpdateCapturedOrder() {
  const auto &locals = graph_->GetFrame(0).GetLocals();
  GetCaptureInfo().interpret_.inputs = locals;
  GetCaptureInfo().interpret_.values.clear();
  GetCaptureInfo().interpret_.values.insert(locals.begin(), locals.end());
  GetCaptureInfo().interpret_.values.insert(graph_->prepare().inputs_.begin(), graph_->prepare().inputs_.end());
  GetCaptureInfo().interpret_.values.insert(graph_->prepare().operations_.begin(), graph_->prepare().operations_.end());
}

void GraphAnalyzer::CollectCapturedAndInterpret() {
  // check inputs is valid if break point is rollback

  GetCaptureInfo().outputs_optimize_.inputs = CollectInputs(GetCaptureInfo().outputs_optimize_.operations);
  GetCaptureInfo().interpret_.inputs = graph_->GetFrame(0).GetLocals();
  const auto &prepare = graph_->prepare().operations_;
  auto &interpret = GetCaptureInfo().interpret_.operations;
  interpret.insert(interpret.begin(), prepare.begin(), prepare.end());
  // interpret.outputs layout:
  // | top-graph alive nodes | subgraph-1 alive nodes | ... | subgraph-n alive nodes | side-effect nodes |
  GetCaptureInfo().interpret_.outputs = graph_->break_info().alive_nodes_;
  if (graph_break_info_.is_break_at_call && !graph_break_info_.captured_subgraphs.empty()) {
    for (const Graph *graph : graph_break_info_.captured_subgraphs) {
      const std::vector<ValueNode *> &alive_nodes = graph->break_info().alive_nodes_;
      std::copy(alive_nodes.begin(), alive_nodes.end(), std::back_inserter(info_.interpret_.outputs));
    }
  }
}

namespace {
// Get the alive nodes and side-effect alive nodes in this graph before the break bci.
std::vector<ValueNode *> GetAliveNodes(const Graph *graph) {
  int bci = graph->GetStopTraceBci();
  if (IsPiJitLogOn(LogCfg::kOthers)) {
    GRAPH_JIT_LOG_F("UD analyze: enter GetAliveNodes bci %d", bci);
  }
  std::vector<ValueNode *> alive_nodes = graph->CollectAliveNode(bci);
  mindspore::CompactSet<ValueNode *> uniques;
  for (auto node : alive_nodes) {
    uniques.insert(node);
  }

  if (IsPiJitLogOn(LogCfg::kOthers)) {
    GRAPH_JIT_LOG_F("UD analyze: alive node size : %ld", alive_nodes.size());
    for (auto node : alive_nodes) {
      if (node) {
        GRAPH_JIT_LOG_F("UD analyze: alive node: %s", node->ToString().c_str());
      }
    }
  }
  return alive_nodes;
}
}  // namespace

void GraphAnalyzer::UseDefAnalyze() {
  // Top-graph UD analyze
  std::vector<ValueNode *> alive_nodes = GetAliveNodes(graph_);
  if (!alive_nodes.empty()) {
    bool stop_analyze = false;
    while (!stop_analyze) {
      UpdateCapturedOrder();
      // Add graph output according to leaf nodes.
      stop_analyze = AnalyzeTopGraphAliveNodes(alive_nodes);
      if (!stop_analyze) {
        alive_nodes = GetAliveNodes(graph_);
      }
    }
  }
  UpdateBreakInfo(graph_);

  // SubGraph break optimization.
  if (IsBreakAtCall(graph_)) {
    graph_break_info_.is_break_at_call = true;
    if (IsEnableSubGraphBreakOptimize(graph_)) {
      CallNode *call_node = FindBreakAtCall(graph_);
      AnalyzeSubGraphBreakRecursive(call_node);
    }
  }

  ExpandGraphOutput();
  auto &outputs_optimize = GetCaptureInfo().outputs_optimize_;
  std::reverse(outputs_optimize.operations.begin(), outputs_optimize.operations.end());
  // avoid missing value, update use-def at last, update all inputs use new node
  UpdateUseDefNode();
}

// Check if the bytecode at bci is in a block. Defined in other source file.
bool FindBlock(int bci, const CFG *cfg);

namespace {
bool CanCapturePartialGraph(const Graph *graph) {
  if (graph->GetStopTraceBci() == -1) {  // No graph break, is not a partial graph.
    return false;
  }
  if (graph->ShouldNeverCompile()) {
    MS_LOG(INFO) << "Hit never compile, can not capture graph: " << GetNameAndLocation(graph);
    return false;
  }
  if (FindBlock(graph->GetStopTraceBci(), graph->GetCFG().get())) {
    MS_LOG(INFO) << "Is break at block, can not capture graph: " << GetNameAndLocation(graph);
    return false;
  }
  if (PyCodeWrapper(graph->GetCodeObj()).CellVarsSize() > 0) {
    MS_LOG(INFO) << "Has cellvar, can not capture graph: " << GetNameAndLocation(graph);
    return false;
  }
  const auto *capture_ctx = CaptureContext::GetInstance();
  if (capture_ctx->IsSkip(graph->GetCodeObj(), graph->GetGlobals().ptr())) {
    MS_LOG(INFO) << "Hit skip rules in CaptureContext, can not capture graph: " << GetNameAndLocation(graph);
    return false;
  }
  static const std::unordered_set<int> unsupported_break_op = {
    YIELD_VALUE,           YIELD_FROM,    GET_YIELD_FROM_ITER, SETUP_WITH,   SETUP_FINALLY, WITH_CLEANUP_START,
    WITH_CLEANUP_FINISH,   END_FINALLY,   SETUP_EXCEPT,        POP_EXCEPT,   RERAISE,       RAISE_VARARGS,
    JUMP_IF_NOT_EXC_MATCH, BEGIN_FINALLY, POP_FINALLY,         CALL_FINALLY, JUMP_ABSOLUTE, JUMP_FORWARD};
  int break_bci = graph->GetStopTraceBci();
  const auto &instr_pool = graph->GetCFG()->instr_pool();
  MS_EXCEPTION_IF_CHECK_FAIL(break_bci >= 0 && break_bci < SizeToInt(instr_pool.size()), "Illegal break bci");
  const Instr *break_point = instr_pool[break_bci].get();
  if (unsupported_break_op.find(break_point->op()) != unsupported_break_op.end()) {
    MS_LOG(INFO) << "Is unsupported break op: " << break_point->ToString()
                 << " , can not capture graph: " << GetNameAndLocation(graph);
    return false;
  }
  return true;
}

bool FgAddOutputs(const std::vector<pijit::ValueNode *> &subgraph_outputs,
                  const AbstractWrapperPtr &subgraph_output_abs, const CallNode *call_node) {
  FuncGraphBuilderPtr fg_builder = call_node->GetGraph()->func_graph_builder();
  MS_EXCEPTION_IF_NULL(fg_builder);

  auto AddOutput = [fg_builder, call_node](const AbstractWrapperPtr &output_abs, const ValueNode *output_node) {
    if (fg_builder->AddOutput(output_abs, true)) {
      AnfNodePtr node = fg_builder->ReadLocalVariable(output_abs);
      MS_EXCEPTION_IF_NULL(node);
      fg_builder->UpdateNodesMap(output_node->abstract_wrapper(), node);
      return true;
    }
    MS_LOG(INFO) << "Failed to add output from subgraph: " << GetNameAndLocation(call_node->GetSubGraph())
                 << ", output node is at line: " << output_node->GetLineNo() << ", " << ToString(output_node);
    return false;
  };

  if (subgraph_outputs.size() == 1) {
    MS_LOG(DEBUG) << "Subgraph has single output";
    FuncGraphBuilderPtr sub_fg_builder = call_node->GetSubGraph()->func_graph_builder();
    MS_EXCEPTION_IF_CHECK_FAIL(sub_fg_builder->GetOutputSize() == 1, "Graph output num should be 1");
    if (!AddOutput(subgraph_output_abs, subgraph_outputs[0])) {
      return false;
    }
  } else {
    MS_LOG(DEBUG) << "Subgraph has " << subgraph_outputs.size() << " outputs";
    // subgraph_output_abs should be an AbstractTuple.
    auto unpacked_outputs = fg_build_utils::FgTupleUnpack(fg_builder, subgraph_output_abs);
    if (!unpacked_outputs.has_value()) {
      MS_LOG(INFO) << "Fail to unpack outputs for subgraph: " << GetNameAndLocation(call_node->GetSubGraph());
      return false;
    }
    MS_EXCEPTION_IF_CHECK_FAIL(unpacked_outputs->size() == subgraph_outputs.size(), "Graph output num not equal!");
    for (size_t i = 0; i < unpacked_outputs->size(); ++i) {
      if (!AddOutput(unpacked_outputs->at(i), subgraph_outputs[i])) {
        return false;
      }
    }
  }
  return true;
}
}  // namespace

void GraphAnalyzer::AnalyzeSubGraphBreakRecursive(CallNode *root) {
  MS_EXCEPTION_IF_NULL(root);
  std::stack<CallNode *> call_stack;
  std::stack<std::vector<ValueNode *>> alive_nodes_stack;
  CallNode *call_node = root;
  // 1.From top to bottom, do UD analysis on each subgraph.
  while (call_node != nullptr) {
    Graph *sub_graph = call_node->GetSubGraph();
    MS_EXCEPTION_IF_NULL(sub_graph);
    MS_LOG(INFO) << "Analyze subgraph: " << GetNameAndLocation(sub_graph);
    if (!CanCapturePartialGraph(sub_graph)) {
      break;
    }
    const std::vector<ValueNode *> &alive_nodes = SubGraphUseDefAnalyze(sub_graph);
    if (sub_graph->GetStopTraceReason() == kStopTraceUDAnalyze_Error) {
      MS_LOG(INFO) << "UD analyze failed, can not capture subgraph: " << GetNameAndLocation(sub_graph);
      break;
    }
    UpdateBreakInfo(sub_graph);
    call_stack.push(call_node);
    alive_nodes_stack.push(alive_nodes);
    call_node = FindBreakAtCall(sub_graph);
  }
  // 2.From bottom to top, connect sub-graph to its parent-graph.
  std::vector<ValueNode *> total_outputs;
  while (!call_stack.empty()) {
    call_node = call_stack.top();
    const auto &alive_nodes = alive_nodes_stack.top();
    total_outputs.insert(total_outputs.begin(), alive_nodes.begin(), alive_nodes.end());
    call_stack.pop();
    alive_nodes_stack.pop();  // It will destroy the std::vector object on top of the stack.

    bool can_capture = true;
    if (!total_outputs.empty()) {
      // 2.1 Add AnfNode in parent-graph to call subgraph.
      AbstractWrapperPtr output_abs = fg_build_utils::FgCallSubGraph(call_node);
      if (output_abs == nullptr) {
        can_capture = false;
      } else {
        // 2.2 Add the output nodes of subgraph to the output of parent-graph.
        can_capture = FgAddOutputs(total_outputs, output_abs, call_node);
      }
    }
    if (can_capture) {
      graph_break_info_.captured_subgraphs.push_front(call_node->GetSubGraph());
    } else {
      graph_break_info_.captured_subgraphs.clear();
      total_outputs.clear();
    }
  }
  // 3.At last, add subgraphs outputs into top-graph captured_info.
  if (!graph_break_info_.captured_subgraphs.empty()) {
    auto &captured = info_.captured_;
    captured.outputs.insert(captured.outputs.end(), total_outputs.begin(), total_outputs.end());
    captured.operations.insert(captured.operations.end(), total_outputs.begin(), total_outputs.end());
  } else {
    MS_LOG(DEBUG) << "No subgraph captured";
  }
}

std::vector<ValueNode *> GraphAnalyzer::SubGraphUseDefAnalyze(Graph *graph) {
  std::vector<ValueNode *> alive_nodes = GetAliveNodes(graph);
  std::vector<ValueNode *> graph_outputs;
  if (!alive_nodes.empty()) {
    bool finish = false;
    while (!finish) {
      finish = AnalyzeSubGraphAliveNodes(alive_nodes, graph, &graph_outputs);
      if (!finish) {
        alive_nodes = GetAliveNodes(graph);
        graph_outputs.clear();
      }
    }
  }
  return graph_outputs;
}

namespace {
bool CheckNewBreakBci(const Graph *graph, const ValueNode *new_break_point) {
  const auto &traced_nodes = graph->GetTracedNodes();
  if (std::find(traced_nodes.begin(), traced_nodes.end(), new_break_point) == traced_nodes.end()) {
    MS_LOG(INFO) << "Not a traced node, cannot UD reset to: " << new_break_point;
    return false;
  }
  int new_break_bci = new_break_point->bci();
  if (new_break_bci <= 0 || new_break_bci >= graph->GetStopTraceBci()) {
    MS_LOG(INFO) << "Reset bci error, new bci: " << new_break_bci << ", old bci: " << graph->GetStopTraceBci();
    return false;
  }
  return true;
}

ValueNode *MutateFreeVarNode(ValueNode *node, std::vector<ValueNode *> *output_optimize) {
  MS_LOG(DEBUG) << "Reconstruct freevar node: " << node->ToString();
  output_optimize->push_back(node);
  MS_EXCEPTION_IF_CHECK_FAIL(node->inputs().size() == 1, "inputs.size() should be 1");
  ValueNode *binary_subscr = node->inputs()[0];
  output_optimize->push_back(binary_subscr);
  constexpr size_t kBinarySubscrInputsSize = 2;
  MS_EXCEPTION_IF_CHECK_FAIL(binary_subscr->inputs().size() == kBinarySubscrInputsSize, "inputs.size() should be 2");
  ValueNode *load_attr = binary_subscr->inputs()[0];
  output_optimize->push_back(load_attr);
  MS_EXCEPTION_IF_CHECK_FAIL(node->inputs().size() == 1, "inputs.size() should be 1");
  return load_attr->inputs()[0];
}
}  // namespace

bool GraphAnalyzer::AnalyzeSubGraphAliveNodes(const std::vector<ValueNode *> &alive_nodes, Graph *graph,
                                              std::vector<ValueNode *> *graph_outputs) {
  auto fg_builder = graph->func_graph_builder();
  MS_EXCEPTION_IF_NULL(fg_builder);
  fg_builder->ClearOutputNodes();
  std::vector<ValueNode *> output_optimize;

  std::list<ValueNode *> nodes(alive_nodes.begin(), alive_nodes.end());
  auto side_effect_handler = graph->GetSideEffectHandler();
  side_effect_handler->Run();
  auto side_effect_inputs = side_effect_handler->GetSideEffectInputs();
  std::copy(side_effect_inputs.begin(), side_effect_inputs.end(), std::back_inserter(nodes));
  while (!nodes.empty()) {
    auto node = nodes.front();
    nodes.pop_front();
    if (NeedSkipAddGraphOutput(node)) {
      MS_LOG(DEBUG) << "No need to add subgraph output: " << node->ToString();
      continue;
    }
    if (std::find(graph_outputs->begin(), graph_outputs->end(), node) != graph_outputs->end()) {
      continue;
    }
    if (node->GetOpcode() == LOAD_ATTR && node->GetName() == "cell_contents") {  // LOAD_DEREF
      nodes.push_back(MutateFreeVarNode(node, &output_optimize));
      continue;
    }
    if (fg_builder->AddOutput(node->abstract_wrapper(), true)) {
      MS_LOG(DEBUG) << "Add subgraph output success: " << node->ToString();
      graph_outputs->push_back(node);
      continue;
    }
    if (!IsValidOutput(node) && node->GetOpcode() == LOAD_ATTR) {
      MS_LOG(DEBUG) << "Reconstruct subgraph output: " << node->ToString();
      // The side effect node will be handled by side_effect_handler, avoid exec twice.
      if (!node->IsSideEffectNode()) {
        output_optimize.push_back(node);
      }
      nodes.insert(nodes.end(), node->inputs().begin(), node->inputs().end());
      continue;
    }
    MS_LOG(INFO) << "Add subgraph output failed: " << node->ToString();
    if (!CheckNewBreakBci(graph, node)) {
      graph->StopTraceAt(graph->GetStopTraceBci(), StopTraceReason::kStopTraceUDAnalyze_Error);
      fg_builder->ClearOutputNodes();
      graph_outputs->clear();
      return true;
    }
    MS_LOG(INFO) << "Reset subgraph break bci: " << node->ToString() << " at: \""
                 << pijit::GetFileName(node->GetGraph()) << ":" << node->GetLineNo() << "\"";
    graph->StopTraceAt(node->bci(), StopTraceReason::kStopTraceGraphOutput_Type_Unsupported);
    return false;
  }
  // Status que: The nodes used to optimize the outputs of the top graph have been put into info_.outputs_optimize
  //             The nodes will be exec before sub-graph
  //             info_.outputs_optimize will be reversed in UseDefAnalyze.
  // Means if the nodes put into info_.outputs_optimize will exec before the nodes of the top graph, and it is
  // incorrect. So the nodes in output_optimize should be put in the front of info_.outputs_optimize
  auto &ops = info_.outputs_optimize_.operations;
  ops.insert(ops.begin(), output_optimize.begin(), output_optimize.end());
  auto side_effect_nodes = side_effect_handler->GetSideEffect();
  std::for_each(side_effect_nodes.begin(), side_effect_nodes.end(), [this](auto &node) {
    info_.outputs_optimize_.operations.insert(info_.outputs_optimize_.operations.begin(), node);
  });
  info_.outputs_optimize_.operations = SideEffectHandler::OptimizeSideEffect(info_.outputs_optimize_.operations);
  return true;
}

namespace {
// specialize simple data, not all equal
bool IsDuplicateData(const AbstractBasePtr &left, const AbstractBasePtr &right) {
  if (left == nullptr || right == nullptr || left->tid() != right->tid()) {
    return false;
  }
  // first check ptr
  if (left == right) {
    return true;
  }
  // check type
  if (left->isa<abstract::AbstractTensor>()) {
    return false;  // tensor always not duplicate
  }
  if (left->isa<abstract::AbstractSequence>()) {
    const auto &arr_l = left->cast<abstract::AbstractSequencePtr>()->elements();
    const auto &arr_r = right->cast<abstract::AbstractSequencePtr>()->elements();
    return std::equal(arr_l.begin(), arr_l.end(), arr_r.begin(), arr_r.end(), IsDuplicateData);
  }
  if (left->isa<abstract::AbstractDictionary>()) {
    const auto &arr_l = left->cast<abstract::AbstractDictionaryPtr>()->elements();
    const auto &arr_r = right->cast<abstract::AbstractDictionaryPtr>()->elements();
    auto comp = [](const abstract::AbstractElementPair &a, const abstract::AbstractElementPair &b) {
      return IsDuplicateData(a.first, b.first) && IsDuplicateData(a.second, b.second);
    };
    return std::equal(arr_l.begin(), arr_l.end(), arr_r.begin(), arr_r.end(), comp);
  }
  if (left->isa<abstract::AbstractScalar>()) {
    return left->BuildValue() != kValueAny && *left == *right;
  }
  return false;  // invalid output
}

ValueNode *FindDuplicateData(const std::vector<ValueNode *> &nodes, size_t end_idx, ValueNode *node) {
  MS_EXCEPTION_IF_CHECK_FAIL(end_idx <= nodes.size(), "error arguments");
  const auto end_iter = nodes.begin() + end_idx;
  auto iter = std::find(nodes.begin(), end_iter, node);
  if (iter != end_iter) {
    return *iter;
  }
  iter = std::find_if(nodes.begin(), end_iter, [&node](ValueNode *k) {
    auto left = node->abstract_wrapper() ? node->abstract_wrapper()->abstract() : nullptr;
    auto right = k->abstract_wrapper() ? k->abstract_wrapper()->abstract() : nullptr;
    return IsDuplicateData(left, right) || (node->GetVobj() != nullptr && node->GetOwnVobj() == k->GetOwnVobj());
  });
  if (iter != end_iter) {
    return *iter;
  }
  return nullptr;
}
}  // namespace

bool GraphAnalyzer::NeedSkipAddGraphOutput(ValueNode *node) {
  const auto &values = GetCaptureInfo().interpret_.values;
  const auto &captured = GetCaptureInfo().captured_;
  const auto &outputs_optimize = GetCaptureInfo().outputs_optimize_;
  const auto &replaced = GetCaptureInfo().replaced_nodes_;
  // If the value can get from local, no need to add to graph output.
  if (IsNonLocalValue(node)) {
    MS_LOG(INFO) << "Skip non local value used as graph output: " << node->ToString();
    return true;
  }
  // This node is defined out of the graph
  if (std::find(values.begin(), values.end(), node) != values.end()) {
    return true;
  }
  // This node has been added to the output
  if (std::find(captured.outputs.begin(), captured.outputs.end(), node) != captured.outputs.end()) {
    return true;
  }
  // This node has been handle
  auto &handled_nodes = outputs_optimize.operations;
  if (std::find(handled_nodes.begin(), handled_nodes.end(), node) != handled_nodes.end()) {
    return true;
  }

  if (node->abstract_wrapper() && node->abstract_wrapper()->IsConstant()) {
    PyObject *op = node->GetVobj() ? node->GetVobj()->GetPyObject().ptr() : nullptr;
    if (op == nullptr) {
      return false;  // constant value no python object, can't make instruction
    } else if (CheckConstPyObject(op)) {
      // now, only python constant
      if (PyUnicode_Check(op) && !IsValidOutputAbstractScalar(node->abstract_wrapper()->abstract())) {
        // filter FakeNodeKey
        return false;
      }
      node->ClearInputs();
      node->SetOpcode(LOAD_CONST);
      return true;
    } else if (PySlice_Check(op)) {
      const auto &slice_inputs = node->abstract_wrapper()->GetSliceInputsPyObject();
      auto start = graph_->NewValueNode(AObject::Convert(slice_inputs[0]), LOAD_CONST, -1, {});
      auto stop = graph_->NewValueNode(AObject::Convert(slice_inputs[1]), LOAD_CONST, -1, {});
      auto step = graph_->NewValueNode(AObject::Convert(slice_inputs[2]), LOAD_CONST, -1, {});
      auto ret = graph_->NewValueNode(AObject::Convert(op), BUILD_SLICE, 3, {start, stop, step});
      GetCaptureInfo().replaced_nodes_[node] = ret;
      GetCaptureInfo().outputs_optimize_.operations.push_back(ret);
    }
  }
  // remove duplicate data
  auto duplicate_data = FindDuplicateData(captured.outputs, captured.outputs.size(), node);
  if (duplicate_data == nullptr) {
    duplicate_data = FindDuplicateData(outputs_optimize.outputs, outputs_optimize.outputs.size(), node);
  }
  if (duplicate_data != nullptr) {
    GetCaptureInfo().replaced_nodes_[node] = duplicate_data;
    MS_LOG(INFO) << "skip same data: [" << node->ToString() << "] and [" << duplicate_data->ToString();
    return true;
  }
  auto iter_replaced = replaced.find(node);
  if (iter_replaced != replaced.end()) {
    MS_LOG(INFO) << "duplicate node " << node->ToString();
    return NeedSkipAddGraphOutput(iter_replaced->second);
  }
  return false;
}

ValueNode *GraphAnalyzer::GetBuiltinMethodNode(std::vector<ValueNode *> *out, const std::string &name,
                                               const std::string &cls) {
  PyObject *builtin_module = PyEval_GetBuiltins();
  MS_EXCEPTION_IF_NULL(builtin_module);
  if (PyModule_Check(builtin_module)) {
    builtin_module = PyModule_GetDict(builtin_module);
    MS_EXCEPTION_IF_NULL(builtin_module);
  }
  py::object builtin_object;
  if (cls.empty()) {
    auto method = PyDict_GetItemString(builtin_module, name.c_str());
    builtin_object = py::reinterpret_borrow<py::object>(method);
  } else {
    auto cls_object = PyDict_GetItemString(builtin_module, cls.c_str());
    MS_EXCEPTION_IF_NULL(cls_object);
    auto method = PyObject_GetAttrString(cls_object, name.c_str());
    builtin_object = py::reinterpret_steal<py::object>(method);
  }
  MS_EXCEPTION_IF_NULL(builtin_object.ptr());
  auto method_node = graph_->NewValueNode(AObject::Convert(builtin_object), LOAD_CONST, 0, {});
  out->push_back(method_node);
  return method_node;
}

namespace {
void UpdateNodeInputs(Graph *graph, std::vector<ValueNode *> *nodes_p, std::map<ValueNode *, ValueNode *> *map_p) {
  const auto &map = *map_p;
  const auto &nodes = *nodes_p;
  auto latest = [&map](ValueNode *node) {
    int limit = 10000;
    for (auto iter = map.find(node); limit > 0 && iter != map.end(); iter = map.find(node), --limit) {
      node = iter->second;
    }
    MS_EXCEPTION_IF_CHECK_FAIL(limit > 0, "maybe circle map");
    return node;
  };
  bool changed;
  do {
    changed = false;
    // for each node check it's inputs
    for (auto node_iter = nodes.begin(); node_iter != nodes.end(); ++node_iter) {
      auto node = *node_iter;
      auto &in = node->inputs();
      auto in_iter = std::find_if(in.begin(), in.end(), [&map](ValueNode *k) { return map.find(k) != map.end(); });
      if (in_iter == in.end()) {
        continue;  // not find, do nothing
      }
      // collect latest node
      std::vector<ValueNode *> new_in = node->inputs();
      for (; in_iter != in.end(); ++in_iter) {
        new_in[in_iter - in.begin()] = latest(*in_iter);
      }
      // if node is a a new node, update inputs
      if (node->GetLineNo() < 0) {
        in = std::move(new_in);
        continue;
      }
      Opcode opcode(node->GetOpcode());
      ValueNode *new_node;
      if (opcode.IsCall()) {
        new_node = graph->NewCallNode(opcode, node->GetOparg(), std::move(new_in));
        new_node->SetVobj(node->GetVobj());
      } else {
        new_node = graph->NewValueNode(node->GetVobj(), opcode, node->GetOparg(), std::move(new_in), node->GetName());
      }
      (*nodes_p)[node_iter - nodes.begin()] = new_node;
      (*map_p)[node] = new_node;
      changed = true;
    }
  } while (changed);
}
}  // namespace

void GraphAnalyzer::UpdateUseDefNode() {
  auto &map = GetCaptureInfo().replaced_nodes_;
  auto &nodes = GetCaptureInfo().outputs_optimize_.operations;
  if (map.empty()) {
    UpdateUseDefOrder(&nodes);
    return;
  }
  UpdateNodeInputs(graph_, &nodes, &map);
  UpdateUseDefOrder(&nodes);
}

}  // namespace pijit
}  // namespace mindspore
