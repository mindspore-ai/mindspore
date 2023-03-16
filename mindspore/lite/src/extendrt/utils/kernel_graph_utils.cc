/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include <string>
#include <algorithm>
#include <utility>
#include <vector>
#include <map>

#include "src/extendrt/utils/kernel_graph_utils.h"
#include "ir/graph_utils.h"
#include "ir/func_graph_cloner.h"
#include "utils/ms_context.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "base/base_ref_utils.h"

namespace mindspore {
const size_t max_depth = 128;
GraphId KernelGraphUtils::graph_sum_ = 0;

std::vector<AnfNodePtr> KernelGraphUtils::GetKernelGraphOutputs(const KernelGraphPtr &func_graph) {
  if (!func_graph) {
    return {};
  }
  std::vector<AnfNodePtr> outputs = func_graph->outputs();
  while (true) {
    auto has_replace = false;
    for (auto it = outputs.begin(); it != outputs.end(); ++it) {
      auto output = *it;
      std::vector<AnfNodePtr> one_outputs;
      if (IsPrimitiveCNode(output, prim::kPrimDepend)) {
        auto depend = output->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(depend);
        output = depend->input(kRealInputIndexInDepend);
      }
      if (IsPrimitiveCNode(output, prim::kPrimMakeTuple)) {
        auto make_tuple = output->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(make_tuple);
        auto &inputs = make_tuple->inputs();
        one_outputs = std::vector<AnfNodePtr>(inputs.begin() + 1, inputs.end());
      } else {
        one_outputs = {output};
      }
      if (one_outputs.size() != 1 || one_outputs[0] != output) {
        it = outputs.erase(it);
        outputs.insert(it, one_outputs.begin(), one_outputs.end());
        has_replace = true;
        break;
      }
    }
    if (!has_replace) {
      break;
    }
  }
  return outputs;
}

KernelGraphPtr KernelGraphUtils::ConstructKernelGraph(const FuncGraphPtr &func_graph,
                                                      std::vector<KernelGraphPtr> *all_out_graph,
                                                      mindspore::device::DeviceType device_target) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(all_out_graph);
  auto node_list = TopoSort(func_graph->get_return());
  auto graph = NewKernelGraph();
  MS_EXCEPTION_IF_NULL(graph);
  front_backend_graph_map_[func_graph.get()] = graph;
  MS_LOG(INFO) << "Create graph: " << graph->graph_id();
  graph->set_device_target(device_target);
  // Create parameter
  for (const auto &node : func_graph->parameters()) {
    MS_LOG(DEBUG) << "Start create new node, node = " << node->DebugString();
    auto graph_inputs = graph->MutableInputs();
    MS_EXCEPTION_IF_NULL(graph_inputs);
    auto new_parameter = CreateNewParameter(node, graph.get());
    graph_inputs->push_back(new_parameter);
    graph->FrontBackendMapAdd(node, new_parameter);
  }
  for (const auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    if (node->isa<Parameter>()) {
      continue;
    }
    MS_LOG(DEBUG) << "Start create new node, node = " << node->DebugString();
    // Create value node
    if (node->isa<ValueNode>()) {
      // Create common value node
      if (!IsValueNode<FuncGraph>(node)) {
        (void)CreateNewValueNode(node, graph.get());
        continue;
      }
      // Create child kernel graph according ValueNode<FuncGraph>
      FuncGraphPtr child_graph = common::AnfAlgo::GetValueNodeFuncGraph(node);
      if (front_backend_graph_map_.find(child_graph.get()) == front_backend_graph_map_.end()) {
        (void)ConstructKernelGraph(child_graph, all_out_graph, device_target);
      }
      (void)CreateValueNodeKernelGraph(node, graph.get());
      continue;
    }
    // Create cnode
    if (!CreateCNodeOfKernelGraph(node, graph.get())) {
#ifdef ENABLE_DUMP_IR
      DumpIR("construct_kernel_graph_fail.ir", func_graph);
#endif
      MS_LOG(EXCEPTION) << "Construct func graph " << func_graph->ToString() << " failed.";
    }
  }

  AddParameterToGraphInputs(func_graph->parameters(), graph.get());
  FuncGraphManagerPtr manager = MakeManager({graph});
  graph->SetInputNodes();
  SetInputNodeUsage(graph, manager);
  graph->SetExecOrderByDefault();

#ifndef ENABLE_SECURITY
  if (KernelGraphUtils::ExistSummaryNode(graph.get())) {
    graph->set_summary_node_exist(true);
  }
#endif

  all_out_graph->push_back(graph);
  return graph;
}

KernelGraphPtr KernelGraphUtils::ConstructKernelGraphFromNodeList(const AnfNodePtrList &node_list,
                                                                  const AnfNodePtrList &outputs,
                                                                  mindspore::device::DeviceType device_target,
                                                                  bool common_opt) {
  mindspore::HashMap<AnfNodePtr, AnfNodePtr> other_graph_cnode;
  auto graph = NewKernelGraph();
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Create graph: " << graph->graph_id();
  for (const auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    MS_LOG(DEBUG) << "Start create new cnode, node = " << node->DebugString();
    if (!node->isa<CNode>()) {
      MS_LOG(EXCEPTION) << "Node " << node->DebugString() << " is not CNode";
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    // create a new cnode object
    auto new_cnode = CreateNewCNode(cnode, graph, &other_graph_cnode);
    MS_EXCEPTION_IF_NULL(new_cnode);
    new_cnode->set_abstract(cnode->abstract());
    new_cnode->set_scope(cnode->scope());
    new_cnode->set_attrs(cnode->attrs());
    if (IsPrimitiveCNode(cnode, prim::kPrimLoad)) {
      new_cnode->set_fullname_with_scope(cnode->input(kFirstDataInputIndex)->fullname_with_scope());
    }
    // record map relations between anf from ME and new anf node used in backend
    graph->FrontBackendMapAdd(node, new_cnode);
  }
  graph->set_device_target(device_target);
  // add a make_tuple at the end of graph as output
  FuncGraphManagerPtr manager = MakeManager({graph});
  if (manager) {
    manager->AddFuncGraph(graph);
    graph->set_manager(manager);
  }
  graph->SetExecOrderByDefault();

  return graph;
}

KernelGraphPtr KernelGraphUtils::NewKernelGraph() {
  auto graph = std::make_shared<KernelGraph>();
  graph->set_graph_id(graph_sum_);
  graphs_[graph_sum_++] = graph;
  return graph;
}

ParameterPtr KernelGraphUtils::CreateNewParameter(const AnfNodePtr &anf, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(anf);
  MS_EXCEPTION_IF_NULL(graph);
  if (!anf->isa<Parameter>()) {
    MS_LOG(EXCEPTION) << "Anf[" << anf->DebugString() << "] is not a parameter";
  }

  auto param_value = GetParamDefaultValue(anf);
  ParameterPtr new_parameter = nullptr;
  // if parameter's python parameter has been exist a backend parameter, reuse the exist parameter
  if (param_value != nullptr) {
    new_parameter = param_value->parameter();
    if (new_parameter == nullptr) {
      TraceGuard trace_guard(std::make_shared<TraceCopy>(anf->debug_info()));
      new_parameter = graph->NewParameter(anf->cast<ParameterPtr>());
      param_value->set_parameter(new_parameter);
    }
  } else {
    TraceGuard trace_guard(std::make_shared<TraceCopy>(anf->debug_info()));
    new_parameter = graph->NewParameter(anf->cast<ParameterPtr>());
  }

  new_parameter->IncreaseUsedGraphCount();

  return new_parameter;
}

ValueNodePtr KernelGraphUtils::CreateNewValueNode(const AnfNodePtr &anf, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(anf);
  MS_EXCEPTION_IF_NULL(graph);
  auto value_node = anf->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<None>()) {
    return nullptr;
  }
  auto new_value_node = graph->NewValueNode(value_node);
  graph->FrontBackendMapAdd(anf, new_value_node);
  graph->AddValueNodeToGraph(new_value_node);
  return new_value_node;
}

ParameterPtr KernelGraphUtils::CreateNewParameterFromParameter(const AnfNodePtr &anf, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(anf);
  if (!anf->isa<Parameter>()) {
    MS_LOG(EXCEPTION) << "Anf[" << anf->DebugString() << "] is not a parameter";
  }
  MS_EXCEPTION_IF_NULL(graph);
  auto param_value = GetParamDefaultValue(anf);
  auto valid_inputs = graph->MutableValidInputs();
  MS_EXCEPTION_IF_NULL(valid_inputs);
  auto graph_inputs = graph->MutableInputs();
  MS_EXCEPTION_IF_NULL(graph_inputs);
  ParameterPtr new_parameter = nullptr;
  auto func_graph = anf->func_graph();
  if (func_graph->manager() != nullptr && func_graph->exist_multi_target() &&
      graph->device_target() == device::DeviceType::kCPU) {
    auto iter = default_param_map_.find(anf);
    if (iter != default_param_map_.end()) {
      new_parameter = iter->second;
    }
    if (new_parameter != nullptr) {
      return new_parameter;
    }
    TraceGuard trace_guard(std::make_shared<TraceCopy>(anf->debug_info()));
    new_parameter = graph->NewParameter(anf->cast<ParameterPtr>());
    graph_inputs->push_back(new_parameter);
    valid_inputs->push_back(true);
    default_param_map_[anf] = new_parameter;
    return new_parameter;
  }
  // if parameter's python parameter has been exist a backend parameter, reuse the exist parameter
  if (param_value != nullptr) {
    new_parameter = param_value->parameter();
  }
  if (new_parameter == nullptr) {
    TraceGuard trace_guard(std::make_shared<TraceCopy>(anf->debug_info()));
    new_parameter = graph->NewParameter(anf->cast<ParameterPtr>());

    auto input_node_iter = partial_parameters_map_.find(anf);
    if (input_node_iter != partial_parameters_map_.end()) {
      InitInternalOutputParameter(input_node_iter->second, new_parameter);
    }

    if (param_value != nullptr) {
      param_value->set_parameter(new_parameter);
    }
  }
  new_parameter->IncreaseUsedGraphCount();
  graph_inputs->push_back(new_parameter);
  valid_inputs->push_back(true);
  return new_parameter;
}

ParamInfoPtr KernelGraphUtils::GetParamDefaultValue(const AnfNodePtr &node) {
  if (node == nullptr) {
    return nullptr;
  }
  auto parameter = node->cast<ParameterPtr>();
  if (parameter == nullptr || !parameter->has_default()) {
    return nullptr;
  }
  return parameter->param_info();
}

void KernelGraphUtils::InitInternalOutputParameter(const AnfNodePtr &out_node, const AnfNodePtr &parameter) {
  auto graph_id = GetGraphIdByNode(out_node);
  if (graph_id == kInvalidGraphId) {
    return;
  }
  auto node_graph = GetGraph(graph_id);
  if (node_graph == nullptr) {
    return;
  }
  MS_LOG(INFO) << "Init parameter with pre graph output node: " << out_node->DebugString();
  auto ref_node_with_index = node_graph->GetInternalOutputByFrontNode(out_node);
  auto ref_node = ref_node_with_index.first;
  if (ref_node == nullptr) {
    MS_LOG(INFO) << "No corresponding internal output for output node";
    return;
  }
  size_t output_idx = 0;
  if (common::AnfAlgo::CheckPrimitiveType(out_node, prim::kPrimTupleGetItem)) {
    output_idx = common::AnfAlgo::GetTupleGetItemOutIndex(out_node->cast<CNodePtr>());
  }
  auto real_kernel = common::AnfAlgo::VisitKernel(ref_node, output_idx);
  auto ref_real_node = real_kernel.first;
  auto ref_real_node_index = real_kernel.second;
  if (ref_real_node->isa<CNode>() && node_graph->IsUniqueTargetInternalOutput(ref_real_node, ref_real_node_index)) {
    auto kernel_info = ref_real_node->kernel_info();
    if (kernel_info == nullptr || !kernel_info->has_build_info()) {
      MS_LOG(INFO) << "No kernel info";
      return;
    }
    if (!common::AnfAlgo::IsNopNode(ref_real_node) && !AnfAlgo::OutputAddrExist(ref_real_node, ref_real_node_index)) {
      MS_LOG(INFO) << "No kernel address";
      return;
    }
    auto address = AnfAlgo::GetMutableOutputAddr(ref_real_node, ref_real_node_index);
    auto format = AnfAlgo::GetOutputFormat(ref_real_node, ref_real_node_index);
    auto type = AnfAlgo::GetOutputDeviceDataType(ref_real_node, ref_real_node_index);
    auto d_kernel_info = std::make_shared<device::KernelInfo>();
    MS_EXCEPTION_IF_NULL(d_kernel_info);
    parameter->set_kernel_info(d_kernel_info);
    kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
    builder.SetOutputsDeviceType({type});
    builder.SetOutputsFormat({format});
    d_kernel_info->set_select_kernel_build_info(builder.Build());
    AnfAlgo::SetOutputAddr(address, 0, parameter.get());
    auto abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(type),
                                                               parameter->Shape()->cast<abstract::BaseShapePtr>());
    parameter->set_abstract(abstract);
  }
}

AnfNodePtr KernelGraphUtils::CreateNewParameterFromCNode(const AnfNodePtr &anf, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(anf);
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Create a new parameter from cnode[" << anf->DebugString() << "]";
  if (IsPrimitiveCNode(anf, prim::kPrimLoad)) {
    auto input = common::AnfAlgo::GetInputNode(anf->cast<CNodePtr>(), 0);
    MS_EXCEPTION_IF_NULL(input);
    if (input->isa<Parameter>()) {
      auto new_param = CreateNewParameterFromParameter(input, graph);
      auto context_ptr = MsContext::GetInstance();
      MS_EXCEPTION_IF_NULL(context_ptr);
      if (context_ptr->get_param<bool>(MS_CTX_ENABLE_MINDRT) == true) {
        graph->CacheInternalParameterToFrontNode(new_param, {anf, 0});
      }
      return new_param;
    }
  }
  return CreateParameterFromTuple(anf, graph);
}

ValueNodePtr KernelGraphUtils::CreateValueNodeKernelGraph(const AnfNodePtr &anf, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(anf);
  MS_EXCEPTION_IF_NULL(graph);
  auto value_node = anf->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto sub_func_graph = common::AnfAlgo::GetValueNodeFuncGraph(anf);
  MS_EXCEPTION_IF_NULL(sub_func_graph);
  if (front_backend_graph_map_.find(sub_func_graph.get()) == front_backend_graph_map_.end()) {
    MS_LOG(EXCEPTION) << "FuncGraph: " << sub_func_graph->ToString() << " has not been transformed to KernelGraph.";
  }
  auto sub_kernel_graph = front_backend_graph_map_[sub_func_graph.get()];

  ValueNodePtr new_value_node = std::make_shared<ValueNode>(sub_kernel_graph);
  new_value_node->set_abstract(value_node->abstract());
  // create new kernel_info of new value_node
  auto kernel_info = std::make_shared<device::KernelInfo>();
  new_value_node->set_kernel_info(kernel_info);
  // create kernel_build_info for new value node
  auto kernel_build_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_builder->Build(), new_value_node.get());
  AnfAlgo::SetGraphId(graph->graph_id(), new_value_node.get());

  graph->FrontBackendMapAdd(anf, new_value_node);

  return new_value_node;
}

bool KernelGraphUtils::CreateCNodeOfKernelGraph(const AnfNodePtr &node, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // create a new cnode object
  auto new_cnode = CreateNewCNode(cnode, graph);
  if (new_cnode == nullptr) {
    return false;
  }
  new_cnode->set_abstract(cnode->abstract());
  std::string fullname;
  if (cnode->input(kAnfPrimitiveIndex)->isa<CNode>()) {
    fullname = cnode->input(kAnfPrimitiveIndex)->fullname_with_scope();
  } else {
    fullname = cnode->fullname_with_scope();
  }
  new_cnode->set_fullname_with_scope(fullname);
  new_cnode->set_scope(cnode->scope());
  graph->FrontBackendMapAdd(node, new_cnode);
  SetReturnNode(new_cnode, graph);
  return true;
}

void KernelGraphUtils::AddParameterToGraphInputs(const std::vector<AnfNodePtr> &parameters, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_inputs = graph->MutableInputs();
  MS_EXCEPTION_IF_NULL(graph_inputs);
  graph_inputs->clear();
  for (auto &parameter : parameters) {
    MS_EXCEPTION_IF_NULL(parameter);
    auto backend_parameter = graph->GetBackendAnfByFrontAnf(parameter);
    if (backend_parameter == nullptr) {
      // for example "def f(x,y,z) {return x + y}", parameter z in unused
      auto new_parameter = CreateNewParameter(parameter, graph);
      graph_inputs->push_back(new_parameter);
      graph->FrontBackendMapAdd(parameter, new_parameter);
      MS_LOG(INFO) << "Can't find parameter:" << parameter->DebugString();
      continue;
    }
    graph_inputs->push_back(backend_parameter);
  }
}

void KernelGraphUtils::SetInputNodeUsage(const KernelGraphPtr &graph, const FuncGraphManagerPtr &manager) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(manager);
  auto input_nodes = graph->input_nodes();
  for (auto &input_node : input_nodes) {
    if (input_node->isa<Parameter>()) {
      auto node_ptr = input_node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(node_ptr);
      if (!IsUsedByRealKernel(manager, input_node, graph->graph_id())) {
        node_ptr->SetNotUsedByRealKernelInGraph(graph->graph_id());
      }
      auto shape = node_ptr->Shape();
      if (IsShapeDynamic(shape->cast<abstract::ShapePtr>())) {
        node_ptr->set_has_dynamic_shape(true);
      }
    }
  }
}

#ifndef ENABLE_SECURITY
bool KernelGraphUtils::ExistSummaryNode(const KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto ret = graph->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  auto all_nodes = DeepLinkedGraphSearch(ret);
  for (auto &n : all_nodes) {
    if (IsPrimitiveCNode(n, prim::kPrimScalarSummary) || IsPrimitiveCNode(n, prim::kPrimTensorSummary) ||
        IsPrimitiveCNode(n, prim::kPrimImageSummary) || IsPrimitiveCNode(n, prim::kPrimHistogramSummary)) {
      return true;
    }
  }
  return false;
}
#endif

GraphId KernelGraphUtils::GetGraphIdByNode(const AnfNodePtr &front_anf) const {
  for (const auto &graph_item : graphs_) {
    auto graph = graph_item.second;
    MS_EXCEPTION_IF_NULL(graph);
    // if front_anf is a parameter,the backend parameter may have two
    if (graph->GetBackendAnfByFrontAnf(front_anf) != nullptr) {
      return graph_item.first;
    }
  }
  MS_EXCEPTION_IF_NULL(front_anf);
  MS_LOG(DEBUG) << "Front_anf " << front_anf->DebugString() << " is not exist in any graph";
  return kInvalidGraphId;
}

KernelGraphPtr KernelGraphUtils::GetGraph(mindspore::GraphId graph_id) const {
  auto it = graphs_.find(graph_id);
  if (it == graphs_.end()) {
    MS_LOG(INFO) << "Can't find graph " << graph_id;
    return nullptr;
  }
  return it->second;
}

AnfNodePtr KernelGraphUtils::CreateParameterFromTuple(const AnfNodePtr &node, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);
  auto new_parameter = graph->TransTupleToMakeTuple(graph->NewParameter(node->abstract()));
  auto parameters = common::AnfAlgo::GetAllOutput(new_parameter);
  std::vector<AnfNodePtr> pre_graph_out = {node};
  // If a cnode is a call, it's input0 is a cnode too, so it doesn't have primitive
  if (!pre_graph_out.empty() && !AnfUtils::IsRealKernel(node)) {
    pre_graph_out = common::AnfAlgo::GetAllOutput(node, {prim::kPrimTupleGetItem, prim::kPrimUpdateState});
  }

  for (size_t i = 0; i < parameters.size(); ++i) {
    const auto &parameter = parameters[i];
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    if (context_ptr->get_param<bool>(MS_CTX_ENABLE_MINDRT) == true) {
      // In control flow, if the input of the cnode is a call node, it will be processed as a make_tuple input,
      // which needs to be linked when processing the internal node.
      graph->CacheInternalParameterToFrontNode(parameter, {node, i});
    }
    auto valid_inputs = graph->MutableValidInputs();
    MS_EXCEPTION_IF_NULL(valid_inputs);
    auto graph_inputs = graph->MutableInputs();
    MS_EXCEPTION_IF_NULL(graph_inputs);
    valid_inputs->push_back(true);
    graph_inputs->push_back(parameter);
  }
  size_t param_index = 0;
  for (const auto &out_node : pre_graph_out) {
    size_t output_size = AnfUtils::GetOutputTensorNum(out_node);
    for (size_t i = 0; i < output_size; i++) {
      if (param_index >= parameters.size()) {
        MS_LOG(EXCEPTION) << "Parameters size:" << parameters.size() << "out of range.Node:" << node->DebugString()
                          << ",out_node:" << out_node->DebugString();
      }
      InitInternalOutputParameter(out_node, parameters[param_index++]);
    }
  }
  return new_parameter;
}

CNodePtr KernelGraphUtils::CreateNewCNode(const CNodePtr &cnode, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> cnode_inputs;
  auto attr_input = cnode->input(kAnfPrimitiveIndex);
  MS_EXCEPTION_IF_NULL(attr_input);
  if (IsValueNode<FuncGraph>(attr_input)) {
    // cnode is a graph or a call
    cnode_inputs = CreateValueNode(cnode, graph);
  } else if (attr_input->isa<CNode>()) {
    // cnode ia a call (partial/switch/switch_layer)
    // 1. take the args of call to the partial node, as the real_args to call switch's or switch_layer's child graph
    // 2. the call in frontend is map to the partial/switch/switch_layer in backend and haven't been created
    cnode_inputs = CreateSwitchOrPartialNode(cnode, graph);
    if (cnode_inputs.empty()) {
      MS_LOG_ERROR << "Create switch or partial failed, cnode:" << cnode->DebugString();
      return nullptr;
    }
  } else {
    // get primitive of old node
    auto prim = common::AnfAlgo::GetCNodePrimitive(cnode);
    MS_EXCEPTION_IF_NULL(prim);
    // push attr to inputs[0] of new cnode
    cnode_inputs = {graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(*prim)))};
  }
  // handle inputs of cnode except primitive
  CreateCNodeInputs(cnode, graph, &cnode_inputs);
  TraceGuard trace_guard(std::make_shared<TraceCopy>(cnode->debug_info()));
  auto new_cnode = graph->NewCNodeWithInfos(cnode_inputs, cnode);
  // if the cnode is call switch, remove call
  if (new_cnode->inputs().size() > 1) {
    auto first_input = new_cnode->input(kFirstDataInputIndex);
    MS_EXCEPTION_IF_NULL(first_input);
    if (common::AnfAlgo::CheckPrimitiveType(new_cnode, prim::kPrimCall) &&
        common::AnfAlgo::CheckPrimitiveType(first_input, prim::kPrimSwitch)) {
      new_cnode = first_input->cast<CNodePtr>();
    }
    if (common::AnfAlgo::CheckPrimitiveType(new_cnode, prim::kPrimCall) &&
        common::AnfAlgo::CheckPrimitiveType(first_input, prim::kPrimSwitchLayer)) {
      auto abstract = cnode->abstract();
      new_cnode = first_input->cast<CNodePtr>();
      new_cnode->set_abstract(abstract);
    }
  }
  return new_cnode;
}

void KernelGraphUtils::SetReturnNode(const AnfNodePtr &node, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimReturn)) {
    constexpr auto kReturnInputIdx = 1;
    auto return_node = node->cast<CNodePtr>();
    graph->set_return(return_node);
    auto graph_output = return_node->input(kReturnInputIdx);
    MS_EXCEPTION_IF_NULL(graph_output);

    // If return's input is value node, then the graph has no kernel, and the pass 'trans tuple to make_tuple' cannot
    // match this pattern because that pass begin with output node but return node. So we add transform value tuple
    // to make_tuple here.
    if (common::AnfAlgo::IsTupleOutput(graph_output) && graph_output->isa<ValueNode>()) {
      return_node->set_input(kReturnInputIdx, graph->TransTupleToMakeTuple(graph_output));
    }
  }
}

bool KernelGraphUtils::IsUsedByRealKernel(const FuncGraphManagerPtr &manager, const AnfNodePtr &node,
                                          const uint32_t graph_id) {
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(node);
  auto node_users = manager->node_users()[node];
  // filter nodes not in current graph
  for (auto iter = node_users.begin(); iter != node_users.end();) {
    auto func_graph = iter->first->func_graph();
    auto kernel_graph = func_graph->cast<KernelGraphPtr>();
    if (kernel_graph == nullptr) {
      MS_LOG(EXCEPTION) << "func graph cast kernel graph failed, related node is: " << iter->first->DebugString();
    }
    if (kernel_graph->graph_id() != graph_id) {
      iter = node_users.erase(iter);
    } else {
      iter++;
    }
  }

  size_t idx = 0;
  if (std::any_of(node_users.begin(), node_users.end(), [&](const std::pair<AnfNodePtr, int64_t> &kernel) {
        return RecursiveCheck(manager, kernel, &idx);
      })) {
    return true;
  }
  return false;
}

bool KernelGraphUtils::IsShapeDynamic(const abstract::ShapePtr &shape) {
  if (shape == nullptr) {
    return false;
  }
  return std::any_of(shape->shape().begin(), shape->shape().end(), [](int64_t s) { return s < 0; });
}

std::vector<AnfNodePtr> KernelGraphUtils::CreateValueNode(const CNodePtr &cnode, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> cnode_inputs;
  auto attr_input = cnode->input(kAnfPrimitiveIndex);
  MS_EXCEPTION_IF_NULL(attr_input);
  if (common::AnfAlgo::IsGraphKernel(cnode)) {
    auto fg = common::AnfAlgo::GetCNodeFuncGraphPtr(cnode);
    MS_EXCEPTION_IF_NULL(fg);
    auto new_fg = BasicClone(fg);
    cnode_inputs.push_back(std::make_shared<ValueNode>(new_fg));
  } else {
    // create primitive of cnode:call
    cnode_inputs = {graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(prim::kPrimCall->name())))};
    // create a ValueNode<KernelGraph> as input of cnode:call
    if (graph->GetBackendAnfByFrontAnf(attr_input) != nullptr) {
      cnode_inputs.emplace_back(graph->GetBackendAnfByFrontAnf(attr_input));
    } else {
      auto new_value_node = CreateValueNodeKernelGraph(attr_input, graph);
      if (new_value_node != nullptr) {
        cnode_inputs.emplace_back(new_value_node);
      }
    }
  }
  return cnode_inputs;
}

std::vector<AnfNodePtr> KernelGraphUtils::CreateSwitchOrPartialNode(const CNodePtr &cnode, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  // create primitive of cnode:call(partial or switch or switch_layer)
  std::vector<AnfNodePtr> cnode_inputs = {
    graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(prim::kPrimCall->name())))};
  auto attr_input = cnode->input(kAnfPrimitiveIndex);
  MS_EXCEPTION_IF_NULL(attr_input);
  auto cnode_input = graph->GetBackendAnfByFrontAnf(attr_input);
  if (cnode_input == nullptr) {
    MS_LOG(ERROR) << "CNode input[0] is CNode:" << attr_input->DebugString() << ", but input[0] has not been created.";
    return {};
  }
  // if the node is partial, insert the inputs of partial to the call
  if (common::AnfAlgo::CheckPrimitiveType(cnode_input, prim::kPrimPartial)) {
    auto partial_node = attr_input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(partial_node);
    auto partial_inputs = partial_node->inputs();
    (void)std::transform(partial_inputs.begin() + kFirstDataInputIndex, partial_inputs.end(),
                         std::back_inserter(cnode_inputs), [&graph](const AnfNodePtr &node) {
                           MS_EXCEPTION_IF_NULL(graph->GetBackendAnfByFrontAnf(node));
                           return graph->GetBackendAnfByFrontAnf(node);
                         });
    return cnode_inputs;
  } else if (common::AnfAlgo::CheckPrimitiveType(cnode_input, prim::kPrimSwitch)) {
    return CreateCallSwitchInputs(cnode, graph);
  } else if (common::AnfAlgo::CheckPrimitiveType(cnode_input, prim::kPrimSwitchLayer)) {
    return CreateCallSwitchLayerInputs(cnode, graph);
  }
  MS_LOG(ERROR) << "CNode:" << cnode->DebugString() << " input[0]" << cnode_input->DebugString()
                << "must be partial or switch or switch_layer.";
  return {};
}

void KernelGraphUtils::CreateCNodeInputs(const CNodePtr &cnode, KernelGraph *graph,
                                         std::vector<AnfNodePtr> *cnode_inputs) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  if (common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimSwitch)) {
    (void)cnode_inputs->emplace_back(graph->GetBackendAnfByFrontAnf(cnode->input(kFirstDataInputIndex)));
    for (size_t index = kSwitchTrueBranchIndex; index < cnode->inputs().size(); index++) {
      auto node_input = cnode->input(index);
      auto switch_input = CreateSwitchInput(cnode, node_input, graph);
      (void)cnode_inputs->emplace_back(switch_input);
    }
  } else {
    for (size_t input_idx = kFirstDataInputIndex; input_idx < cnode->inputs().size(); input_idx++) {
      auto anf = cnode->input(input_idx);
      MS_EXCEPTION_IF_NULL(anf);
      // anf has been created before
      if (graph->GetBackendAnfByFrontAnf(anf) != nullptr) {
        (void)cnode_inputs->emplace_back(graph->GetBackendAnfByFrontAnf(anf));
        continue;
      } else if (IsValueNode<None>(anf)) {
        continue;
      }
      MS_LOG(EXCEPTION) << "Unexpected input[" << anf->DebugString() << "]";
    }
  }
}

bool KernelGraphUtils::RecursiveCheck(const FuncGraphManagerPtr &manager, const std::pair<AnfNodePtr, int64_t> &kernel,
                                      size_t *idx) {
  auto node = kernel.first;
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(node);
  if (kernel.second > 1 && (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimDepend) ||
                            common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimLoad))) {
    return false;
  }
  if (AnfUtils::IsRealKernel(node) && !common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimPartial)) {
    return true;
  }
  (*idx) += 1;
  // max recursion depth
  if (*idx <= max_depth) {
    auto users = manager->node_users()[node];
    if (std::any_of(users.begin(), users.end(), [&](const std::pair<AnfNodePtr, int64_t> &kernel) {
          return RecursiveCheck(manager, kernel, idx);
        })) {
      return true;
    }
  }
  return false;
}

std::vector<AnfNodePtr> KernelGraphUtils::CreateCallSwitchInputs(const CNodePtr &cnode, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> cnode_inputs = {
    graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(prim::kPrimCall->name())))};
  auto attr_input = cnode->input(kAnfPrimitiveIndex);
  MS_EXCEPTION_IF_NULL(attr_input);
  auto cnode_input = graph->GetBackendAnfByFrontAnf(attr_input);
  auto switch_cnode = cnode_input->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(switch_cnode);
  if (cnode->inputs().size() <= 1) {
    cnode_inputs = switch_cnode->inputs();
    return cnode_inputs;
  }
  std::vector<AnfNodePtr> switch_inputs = {switch_cnode->input(kAnfPrimitiveIndex),
                                           switch_cnode->input(kFirstDataInputIndex)};
  for (size_t index = kSwitchTrueBranchIndex; index < switch_cnode->inputs().size(); index++) {
    auto node = switch_cnode->input(index);
    // there is real input in call, should put it to true and false branch in switch
    if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimPartial)) {
      auto partial_node = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(partial_node);
      std::vector<AnfNodePtr> partial_inputs = partial_node->inputs();
      // Put all call args at the end of partial inputs.
      for (size_t i = kFirstDataInputIndex; i < cnode->size(); ++i) {
        (void)partial_inputs.emplace_back(graph->GetBackendAnfByFrontAnf(cnode->input(i)));
      }
      auto new_partial = graph->NewCNode(partial_inputs);
      (void)switch_inputs.emplace_back(new_partial);
    }
  }
  if (switch_inputs.size() < kSwitchInputSize) {
    MS_LOG(EXCEPTION) << "Switch inputs size: " << switch_inputs.size() << "less than " << kSwitchInputSize;
  }
  auto switch_node = graph->NewCNode(switch_inputs);
  (void)cnode_inputs.emplace_back(switch_node);
  return cnode_inputs;
}

std::vector<AnfNodePtr> KernelGraphUtils::CreateCallSwitchLayerInputs(const CNodePtr &cnode, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> cnode_inputs = {
    graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(prim::kPrimCall->name())))};
  auto attr_input = cnode->input(kAnfPrimitiveIndex);
  MS_EXCEPTION_IF_NULL(attr_input);
  auto cnode_input = graph->GetBackendAnfByFrontAnf(attr_input);
  auto switch_layer_cnode = cnode_input->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(switch_layer_cnode);
  std::vector<AnfNodePtr> switch_layer_inputs = {switch_layer_cnode->input(kAnfPrimitiveIndex),
                                                 switch_layer_cnode->input(kFirstDataInputIndex)};
  auto make_tuple_node = switch_layer_cnode->input(kSwitchLayerBranchesIndex);
  MS_EXCEPTION_IF_NULL(make_tuple_node);
  auto node = make_tuple_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(node);
  auto make_tuple_inputs = node->inputs();
  // there are real inputs in call, should put it to make_tuple in switch_layer
  std::vector<AnfNodePtr> real_inputs;
  for (size_t idx = kFirstDataInputIndex; idx < cnode->inputs().size(); ++idx) {
    real_inputs.emplace_back(graph->GetBackendAnfByFrontAnf(cnode->input(idx)));
  }
  std::vector<AnfNodePtr> new_make_tuple_inputs = {
    graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(prim::kPrimMakeTuple->name())))};
  for (size_t idx = kFirstDataInputIndex; idx < make_tuple_inputs.size(); idx++) {
    auto partial_idx = make_tuple_inputs[idx];
    MS_EXCEPTION_IF_NULL(cnode->abstract());
    std::vector<AnfNodePtr> new_partial_inputs;
    KernelGraphPtr partial_kernel_graph;
    // switch_layer node input is partial cnode
    if (common::AnfAlgo::CheckPrimitiveType(partial_idx, prim::kPrimPartial)) {
      auto partial_node = partial_idx->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(partial_node);
      auto partial_input = partial_node->input(kFirstDataInputIndex);
      partial_kernel_graph = GetValueNode<KernelGraphPtr>(partial_input);
      new_partial_inputs = partial_node->inputs();
    } else if (IsValueNode<KernelGraph>(partial_idx)) {  // switch_layer node input is kernel graph value node
      new_partial_inputs.emplace_back(NewValueNode(std::make_shared<Primitive>(prim::kPrimPartial->name())));
      new_partial_inputs.emplace_back(partial_idx);
      partial_kernel_graph = GetValueNode<KernelGraphPtr>(partial_idx);
    }
    // when branch in swich_layer return function
    MS_EXCEPTION_IF_NULL(partial_kernel_graph);
    auto ret = partial_kernel_graph->get_return();
    MS_EXCEPTION_IF_NULL(ret);
    auto return_input = ret->input(kFirstDataInputIndex);
    if (common::AnfAlgo::CheckPrimitiveType(return_input, prim::kPrimPartial) || return_input->isa<ValueNode>()) {
      ProcessNodeRetFunc(cnode, partial_kernel_graph.get(), real_inputs);
    }
    // partial node add input args
    new_partial_inputs.insert(new_partial_inputs.end(), real_inputs.begin(), real_inputs.end());
    // create new partial node
    auto new_partial = graph->NewCNode(new_partial_inputs);
    new_make_tuple_inputs.emplace_back(new_partial);
  }
  auto new_make_tuple = graph->NewCNode(new_make_tuple_inputs);
  auto abstract = make_tuple_node->abstract();
  if (abstract == nullptr) {
    abstract = std::make_shared<abstract::AbstractTuple>(AbstractBasePtrList());
  }
  new_make_tuple->set_abstract(abstract);
  switch_layer_inputs.emplace_back(new_make_tuple);
  auto new_switch_layer = graph->NewCNode(switch_layer_inputs);
  cnode_inputs.emplace_back(new_switch_layer);
  return cnode_inputs;
}

CNodePtr KernelGraphUtils::CreateSwitchInput(const CNodePtr &cnode, const AnfNodePtr &node_input, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(node_input);
  MS_EXCEPTION_IF_NULL(graph);
  // switch input generalizes partial
  std::vector<AnfNodePtr> partial_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimPartial->name()))};
  if (common::AnfAlgo::CheckPrimitiveType(node_input, prim::kPrimPartial)) {
    auto backend_node = graph->GetBackendAnfByFrontAnf(node_input);
    return backend_node->cast<CNodePtr>();
  } else if (node_input->isa<ValueNode>() && IsValueNode<FuncGraph>(node_input)) {
    partial_inputs.emplace_back(graph->GetBackendAnfByFrontAnf(node_input));
  } else {
    KernelGraphPtr kernel_graph = NewKernelGraph();
    MS_EXCEPTION_IF_NULL(kernel_graph);
    auto parameter = CreateNewParameterFromCNode(cnode, kernel_graph.get());
    MS_EXCEPTION_IF_NULL(parameter);
    parameter->set_abstract(cnode->abstract());
    auto primitive = NewValueNode(std::make_shared<Primitive>(prim::kPrimReturn->name()));
    auto return_node = kernel_graph->NewCNode({primitive, parameter});
    return_node->set_abstract(cnode->abstract());
    kernel_graph->set_return(return_node);
    partial_inputs.emplace_back(std::make_shared<ValueNode>(kernel_graph));
    partial_inputs.emplace_back(graph->GetBackendAnfByFrontAnf(node_input));
  }
  auto partial_node = graph->NewCNode(partial_inputs);
  return partial_node;
}

void KernelGraphUtils::ProcessNodeRetFunc(const CNodePtr &cnode, KernelGraph *graph,
                                          const std::vector<AnfNodePtr> &real_inputs) {
  MS_EXCEPTION_IF_NULL(cnode);
  // process the last cnode(func2), not func1 which abstract is AbstractFunction
  if (cnode->abstract()->isa<abstract::AbstractFunction>()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(graph);
  auto ret = graph->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  auto return_input = ret->input(kFirstDataInputIndex);
  // return node is a function
  std::vector<AnfNodePtr> call_inputs = {
    graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(prim::kPrimCall->name())))};
  if (common::AnfAlgo::CheckPrimitiveType(return_input, prim::kPrimPartial)) {
    auto return_input_cnode = return_input->cast<CNodePtr>();
    auto partial_inputs = return_input_cnode->inputs();
    call_inputs.insert(call_inputs.end(), partial_inputs.begin() + kFirstDataInputIndex, partial_inputs.end());
  } else if (IsValueNode<KernelGraph>(return_input)) {  // return node is kernel graph
    call_inputs.emplace_back(return_input);
  } else {  // return node is value node
    KernelGraphPtr kernel_graph = NewKernelGraph();
    auto valid_inputs = kernel_graph->MutableValidInputs();
    MS_EXCEPTION_IF_NULL(valid_inputs);
    auto graph_inputs = kernel_graph->MutableInputs();
    MS_EXCEPTION_IF_NULL(graph_inputs);
    std::vector<AnfNodePtr> cnode_inputs = {return_input};
    for (auto &real_input : real_inputs) {
      auto new_parameter = kernel_graph->NewParameter(real_input->abstract());
      valid_inputs->push_back(true);
      graph_inputs->push_back(new_parameter);
      cnode_inputs.push_back(new_parameter);
    }
    auto new_cnode = kernel_graph->NewCNode(cnode_inputs);
    new_cnode->set_abstract(cnode->abstract());
    std::vector<AnfNodePtr> return_inputs = {
      kernel_graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(prim::kPrimReturn->name()))), new_cnode};
    auto return_node = kernel_graph->NewCNode(return_inputs);
    return_node->set_abstract(cnode->abstract());
    kernel_graph->set_return(return_node);
    call_inputs.push_back(std::make_shared<ValueNode>(kernel_graph));
  }

  // new call node inputs
  for (auto &input_node : real_inputs) {
    auto parameter_for_input = CreateNewParameterFromCNode(input_node, graph);
    call_inputs.emplace_back(parameter_for_input);
  }

  auto call_node = graph->NewCNode(call_inputs);
  call_node->set_abstract(cnode->abstract());
  // update return input
  ret->

    set_input(kFirstDataInputIndex, call_node);
}

void KernelGraphUtils::GetModelInputsInfo(uint32_t graph_id, std::vector<tensor::TensorPtr> *inputs,
                                          std::vector<std::string> *inputs_name) const {
  MS_LOG(INFO) << "Start get model inputs, graph id : " << graph_id;
  auto kernel_graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(inputs);
  MS_EXCEPTION_IF_NULL(inputs_name);
  auto kernel_graph_inputs = kernel_graph->inputs();
  // find parameters of graph inputs
  for (size_t i = 0; i < kernel_graph_inputs.size(); ++i) {
    if (!kernel_graph_inputs[i]->isa<Parameter>()) {
      MS_LOG(ERROR) << "Kernel graph inputs have anfnode which is not Parameter.";
      continue;
    }
    auto parameter = kernel_graph_inputs[i]->cast<ParameterPtr>();
    if (!common::AnfAlgo::IsParameterWeight(parameter)) {
      std::vector<int64_t> input_shape = AnfAlgo::GetOutputDeviceShape(parameter, 0);
      auto kernel_build_info = AnfAlgo::GetSelectKernelBuildInfo(parameter);
      auto data_type = kernel_build_info->GetOutputDeviceType(0);
      auto ms_tensor = std::make_shared<tensor::Tensor>(data_type, input_shape);
      auto abstract = parameter->abstract();
      MS_EXCEPTION_IF_NULL(abstract);
      if (!abstract->name().empty()) {
        ms_tensor->set_name(abstract->name());
      } else {
        ms_tensor->set_name(parameter->fullname_with_scope());
      }
      inputs->push_back(ms_tensor);
      inputs_name->push_back(abstract->name());
    }
  }
}

void KernelGraphUtils::GetOutputNames(const std::vector<AnfNodePtr> &outputs,
                                      std::vector<std::string> *output_names) const {
  MS_EXCEPTION_IF_NULL(output_names);
  for (const auto &output : outputs) {
    auto real_output_with_index = common::AnfAlgo::VisitKernelWithReturnType(output, 0);
    auto real_output = real_output_with_index.first;
    auto idx = real_output_with_index.second;
    MS_EXCEPTION_IF_NULL(real_output);
    MS_LOG(DEBUG) << " Real output info: " << real_output->DebugString();
    AbstractBasePtr abstract = real_output->abstract();
    std::string output_idx;
    if (utils::isa<abstract::AbstractTuplePtr>(abstract)) {
      auto abstract_tuple = utils::cast<abstract::AbstractTuplePtr>(abstract);
      MS_EXCEPTION_IF_NULL(abstract_tuple);
      auto abstract_list = abstract_tuple->elements();
      if (abstract_list.size() <= idx) {
        MS_LOG(ERROR) << "AbstractTuple's size[" << abstract_list.size() << "] is smaller than expect size[" << idx
                      << "]";
        return;
      }
      abstract = abstract_list[idx];
      output_idx = std::to_string(idx);
    }
    MS_EXCEPTION_IF_NULL(abstract);
    std::string output_name;
    if (abstract->name().empty()) {
      output_name = real_output->fullname_with_scope() + output_idx;
    } else {
      output_name = abstract->name();
    }
    output_names->emplace_back(output_name);
  }
}

void KernelGraphUtils::GetModelOutputsInfo(uint32_t graph_id, std::vector<tensor::TensorPtr> *outputs,
                                           std::vector<std::string> *output_names) const {
  std::vector<tensor::TensorPtr> inputs;
  std::vector<std::string> input_names;
  GetModelInputsInfo(graph_id, &inputs, &input_names);

  auto kernel_graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(output_names);

  VectorRef vector_outputs;
  std::map<tensor::TensorPtr, session::KernelWithIndex> tensor_to_node;
  session::KernelMapTensor node_to_tensor;
  auto anf_outputs = KernelGraphUtils::GetKernelGraphOutputs(kernel_graph);
  for (auto &item : anf_outputs) {
    MS_EXCEPTION_IF_NULL(item);
    MS_LOG(INFO) << "Create node output[" << item->DebugString() << "]";
    vector_outputs.emplace_back(CreateNodeOutputTensors(item, kernel_graph, inputs, &tensor_to_node, &node_to_tensor));
  }
  *outputs = TransformVectorRefToMultiTensor(vector_outputs);
  GetOutputNames(anf_outputs, output_names);
  if (outputs->size() != output_names->size()) {
    MS_LOG_EXCEPTION << "Output tensor size " << outputs->size() << " != output name size " << output_names->size();
  }
  for (size_t i = 0; i < outputs->size(); i++) {
    outputs->at(i)->set_name(output_names->at(i));
  }
}

CNodePtr KernelGraphUtils::CreateNewCNode(const CNodePtr &cnode, KernelGraphPtr graph,
                                          mindspore::HashMap<AnfNodePtr, AnfNodePtr> *other_graph_cnode) {
  return nullptr;
}

mindspore::BaseRef KernelGraphUtils::CreateNodeOutputTensors(
  const mindspore::AnfNodePtr &anf, const mindspore::KernelGraphPtr &graph,
  const mindspore::tensor::TensorPtrList &input_tensors,
  std::map<tensor::TensorPtr, session::KernelWithIndex> *tensor_to_node,
  mindspore::session::KernelMapTensor *node_to_tensor) const {
  MS_EXCEPTION_IF_NULL(tensor_to_node);
  MS_EXCEPTION_IF_NULL(node_to_tensor);
  MS_LOG(DEBUG) << "Create tensor for output[" << anf->DebugString() << "]";
  auto item_with_index = common::AnfAlgo::VisitKernelWithReturnType(anf, 0);
  MS_EXCEPTION_IF_NULL(item_with_index.first);
  MS_LOG(DEBUG) << "Create tensor for output after visit:" << item_with_index.first->DebugString();

  // if is graph return nothing ,the function should return a null anylist
  size_t size = AnfUtils::GetOutputTensorNum(item_with_index.first);
  if (size == 0) {
    return VectorRef();
  }

  //  The outputs of graph may have the same kernel node, no need to create new tensor.
  const auto &iter = node_to_tensor->find(item_with_index);
  if (iter != node_to_tensor->end()) {
    return iter->second;
  }

  const auto &tensor = CreateNodeOutputTensor(item_with_index, graph, input_tensors, tensor_to_node);
  (*node_to_tensor)[item_with_index] = tensor;
  return tensor;
}

BaseRef KernelGraphUtils::CreateNodeOutputTensor(
  const session::KernelWithIndex &node_output_pair, const KernelGraphPtr &graph,
  const std::vector<tensor::TensorPtr> &input_tensors,
  std::map<tensor::TensorPtr, session::KernelWithIndex> *tensor_to_node) const {
  auto &node = node_output_pair.first;
  size_t output_index = node_output_pair.second;
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);
  TypeId type_id = AnfAlgo::GetOutputDeviceDataType(node, output_index);
  if (type_id == kTypeUnknown) {
    type_id = common::AnfAlgo::GetOutputInferDataType(node, output_index);
  }

  auto shape = common::AnfAlgo::GetOutputInferShape(node, output_index);
  if (common::AnfAlgo::IsDynamicShape(node)) {
    auto max_shape = common::AnfAlgo::GetOutputMaxShape(node, output_index);
    if (abstract::ShapeSize(max_shape) > abstract::ShapeSize(shape)) {
      shape = max_shape;
    }
  }
  tensor::TensorPtr tensor;
  bool is_internal_output = graph->IsInternalOutput(node, output_index);
  if (is_internal_output) {
    tensor = graph->GetInternalOutputTensor(node, output_index);
    if (tensor == nullptr) {
      tensor = std::make_shared<tensor::Tensor>(type_id, shape);
      graph->AddInternalOutputTensor(node, output_index, tensor);
    }
  } else {
    tensor = std::make_shared<tensor::Tensor>(type_id, shape);
  }
  MS_EXCEPTION_IF_NULL(tensor);
  tensor->set_padding_type(AnfAlgo::GetOutputReshapeType(node, output_index));
  if (is_internal_output) {
    tensor->set_sync_status(kNoNeedSync);
  } else {
    // if in pynative mode,data only copied to host when user want to print data
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode &&
        ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET) != kGPUDevice) {
      tensor->set_sync_status(kNeedSyncDeviceToHostImmediately);
    } else {
      tensor->set_sync_status(kNeedSyncDeviceToHost);
    }
  }
  tensor->SetIsGraphOutput();
  (*tensor_to_node)[tensor] = node_output_pair;
  return tensor;
  // return BaseRef();
}
}  // namespace mindspore
