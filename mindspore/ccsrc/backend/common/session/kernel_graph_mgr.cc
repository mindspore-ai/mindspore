/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "backend/common/session/kernel_graph_mgr.h"

#include <algorithm>
#include <queue>
#include <utility>
#include <functional>
#include <unordered_map>
#include "include/common/debug/anf_ir_dump.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "backend/common/optimizer/common_backend_optimization.h"
#include "utils/trace_base.h"
#include "ir/func_graph_cloner.h"
#ifndef ENABLE_SECURITY
#include "debug/data_dump/dump_json_parser.h"
#include "debug/data_dump/e2e_dump.h"
#endif

namespace mindspore {
namespace session {
namespace {
constexpr size_t kMaxDepth = 128;
constexpr size_t kFirstIndex = 1;
constexpr int64_t kPairIdx1 = 1;

bool RecursiveCheck(const FuncGraphManagerPtr &manager, const std::pair<AnfNodePtr, int64_t> &kernel, size_t *idx) {
  auto node = kernel.first;
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(node);
  if (kernel.second > kPairIdx1 && (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimDepend) ||
                                    common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimLoad))) {
    return false;
  }
  if (AnfUtils::IsRealKernel(node) && !common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimPartial)) {
    return true;
  }
  (*idx) += 1;
  // max recursion depth
  if (*idx <= kMaxDepth) {
    auto users = manager->node_users()[node];
    if (std::any_of(users.begin(), users.end(), [&](const std::pair<AnfNodePtr, int64_t> &kernel) {
          return RecursiveCheck(manager, kernel, idx);
        })) {
      return true;
    }
  }
  return false;
}

bool IsUsedByRealKernel(const FuncGraphManagerPtr &manager, const AnfNodePtr &node, const uint32_t graph_id) {
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(node);
  auto node_users = manager->node_users()[node];
  // filter nodes not in current graph
  for (auto iter = node_users.begin(); iter != node_users.end();) {
    auto func_graph = iter->first->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
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

bool ExistGraphCaller(const AnfNodePtr &partial_node) {
  MS_EXCEPTION_IF_NULL(partial_node);
  auto partial_cnode = partial_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(partial_cnode);
  auto partial_graph = GetValueNode<FuncGraphPtr>(partial_cnode->input(kFirstDataInputIndex));
  // If graph is nullptr, it means that the funcgraph in the partial node is a deadnode, and the processing is skipped.
  if (partial_graph == nullptr) {
    return false;
  }
  auto graph_nodes = TopoSort(partial_graph->get_return());
  return std::any_of(graph_nodes.begin(), graph_nodes.end(), IsValueNode<FuncGraph>);
}

KernelGraphPtr GetValueNodeKernelGraph(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto value_node = node->cast<ValueNodePtr>();
  if (value_node == nullptr) {
    return nullptr;
  }
  auto value = value_node->value();
  if (value == nullptr) {
    return nullptr;
  }
  auto kernel_graph = value->cast<KernelGraphPtr>();
  return kernel_graph;
}
}  // namespace

ParamInfoPtr GetParamDefaultValue(const AnfNodePtr &node) {
  if (node == nullptr) {
    return nullptr;
  }
  auto parameter = node->cast<ParameterPtr>();
  if (parameter == nullptr || !parameter->has_default()) {
    return nullptr;
  }
  return parameter->param_info();
}

#ifndef ENABLE_SECURITY
bool ExistSummaryNode(const KernelGraph *graph) {
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

GraphId KernelGraphMgr::graph_sum_ = 0;

ValueNodePtr KernelGraphMgr::CreateNewValueNode(const AnfNodePtr &anf, KernelGraph *graph) const {
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

GraphId KernelGraphMgr::GetGraphIdByNode(const AnfNodePtr &front_anf) const {
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

KernelGraphPtr KernelGraphMgr::GetGraph(mindspore::GraphId graph_id) const {
  auto it = graphs_.find(graph_id);
  if (it == graphs_.end()) {
    MS_LOG(INFO) << "Can't find graph " << graph_id;
    return nullptr;
  }
  return it->second;
}

void KernelGraphMgr::ClearGraph() {
  auto graph_iter = graphs_.begin();
  while (graph_iter != graphs_.end()) {
    graph_iter->second.reset();
    graph_iter = graphs_.erase(graph_iter);
  }
  graph_sum_ = 0;
}

void KernelGraphMgr::InitInternalOutputParameter(const AnfNodePtr &out_node, const AnfNodePtr &parameter) const {
  MS_EXCEPTION_IF_NULL(out_node);
  MS_EXCEPTION_IF_NULL(parameter);
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
  size_t output_idx = ref_node_with_index.second;
  if (common::AnfAlgo::CheckPrimitiveType(out_node, prim::kPrimTupleGetItem)) {
    output_idx = common::AnfAlgo::GetTupleGetItemOutIndex(out_node->cast<CNodePtr>());
  }
  auto real_kernel = common::AnfAlgo::VisitKernel(ref_node, output_idx);
  auto ref_real_node = real_kernel.first;
  MS_EXCEPTION_IF_NULL(ref_real_node);
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
    if (!AnfAlgo::OutputAddrExist(ref_real_node, ref_real_node_index, true)) {
      return;
    }

    // Update the kernel build info.
    auto format = AnfAlgo::GetOutputFormat(ref_real_node, ref_real_node_index);
    auto type = AnfAlgo::GetOutputDeviceDataType(ref_real_node, ref_real_node_index);
    kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
    builder.SetOutputsDeviceType({type});
    builder.SetOutputsFormat({format});
    AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), parameter.get());

    // If the flag is enable, it means the graph would run in subgraph sink mode, the internal parameter cannot share
    // the same device address.
    auto address = AnfAlgo::GetMutableOutputAddr(ref_real_node, ref_real_node_index);
    if (!node_graph->has_flag(kFlagEnableZeroCopyInGraph)) {
      AnfAlgo::SetOutputAddr(address, 0, parameter.get());
    }

    abstract::AbstractBasePtr abstract;
    auto shape = parameter->Shape();
    MS_EXCEPTION_IF_NULL(shape);
    if (shape->isa<abstract::NoShape>()) {
      abstract = std::make_shared<abstract::AbstractScalar>(TypeIdToType(type));
    } else if (shape->isa<abstract::DynamicSequenceShape>()) {
      return;
    } else {
      abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(type), shape->cast<abstract::BaseShapePtr>());
    }
    parameter->set_abstract(abstract);
  }
}

AnfNodePtr KernelGraphMgr::CreateParameterFromTuple(const AnfNodePtr &node, KernelGraph *graph) const {
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
    size_t output_size = AnfAlgo::GetOutputElementNum(out_node);
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

ParameterPtr KernelGraphMgr::CreateNewParameterFromParameter(const AnfNodePtr &anf, KernelGraph *graph) {
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
  MS_EXCEPTION_IF_NULL(func_graph);
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

AnfNodePtr KernelGraphMgr::CreateNewParameterFromCNode(const AnfNodePtr &anf, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(anf);
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Create a new parameter from cnode[" << anf->DebugString() << "]";
  return CreateParameterFromTuple(anf, graph);
}

void KernelGraphMgr::GetCNodeInfo(const CNodePtr &cnode, std::vector<AnfNodePtr> *cnode_inputs) const {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(cnode_inputs);
  auto prim = common::AnfAlgo::GetCNodePrimitive(cnode);
  if (prim != nullptr) {
    // push attr to inputs[0] of new cnode
    cnode_inputs->push_back(std::make_shared<ValueNode>(std::make_shared<Primitive>(*prim)));
  } else {
    auto fg = common::AnfAlgo::GetCNodeFuncGraphPtr(cnode);
    MS_EXCEPTION_IF_NULL(fg);
    auto new_fg = BasicClone(fg);
    cnode_inputs->push_back(std::make_shared<ValueNode>(new_fg));
  }
}

void KernelGraphMgr::GetNewCNodeInputs(const CNodePtr &cnode, KernelGraph *graph, std::vector<AnfNodePtr> *cnode_inputs,
                                       mindspore::HashMap<AnfNodePtr, AnfNodePtr> *other_graph_cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(other_graph_cnode);
  MS_EXCEPTION_IF_NULL(cnode_inputs);
  auto origin_inputs = cnode->inputs();
  const bool is_depend = IsPrimitiveCNode(cnode, prim::kPrimDepend);
  // if has multiple depends,only select first depend as parameter
  for (size_t input_idx = 1; input_idx < origin_inputs.size(); input_idx++) {
    auto anf = origin_inputs[input_idx];
    MS_EXCEPTION_IF_NULL(anf);
    // anf has been created before
    if (graph->GetBackendAnfByFrontAnf(anf) != nullptr) {
      (void)cnode_inputs->emplace_back(graph->GetBackendAnfByFrontAnf(anf));
      continue;
    } else if ((is_depend && input_idx > kRealInputIndexInDepend)) {
      cnode_inputs->push_back(NewValueNode(MakeValue(SizeToInt(input_idx))));
      continue;
    } else if (other_graph_cnode->find(anf) != other_graph_cnode->end()) {
      cnode_inputs->push_back((*other_graph_cnode)[anf]);
      continue;
    } else if (anf->isa<ValueNode>() && !IsValueNode<FuncGraph>(anf)) {
      // if input is a value node,
      auto new_value_node = CreateNewValueNode(anf, graph);
      if (new_value_node != nullptr) {
        (void)cnode_inputs->emplace_back(new_value_node);
      }
      continue;
    } else if (anf->isa<Parameter>()) {
      auto new_parameter = CreateNewParameterFromParameter(anf, graph);
      cnode_inputs->push_back(new_parameter);
      graph->FrontBackendMapAdd(anf, new_parameter);
      continue;
    } else {
      // the input node is a cnode from other graph
      auto parameter_from_cnode = CreateNewParameterFromCNode(anf, graph);
      if (parameter_from_cnode == nullptr) {
        parameter_from_cnode = NewValueNode(MakeValue(SizeToLong(input_idx)));
      }
      if (parameter_from_cnode->isa<Parameter>() && IsPrimitiveCNode(anf, prim::kPrimLoad)) {
        auto para = parameter_from_cnode->cast<ParameterPtr>();
        auto load_cnode = anf->cast<CNodePtr>();
        para->set_name(load_cnode->fullname_with_scope());
      }
      cnode_inputs->push_back(parameter_from_cnode);
      (*other_graph_cnode)[anf] = parameter_from_cnode;
    }
  }
}

CNodePtr KernelGraphMgr::CreateNewCNode(const CNodePtr &cnode, KernelGraph *graph,
                                        mindspore::HashMap<AnfNodePtr, AnfNodePtr> *other_graph_cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(other_graph_cnode);
  // get primitive of old node
  std::vector<AnfNodePtr> cnode_inputs;
  GetCNodeInfo(cnode, &cnode_inputs);
  GetNewCNodeInputs(cnode, graph, &cnode_inputs, other_graph_cnode);
  TraceGuard trace_guard(std::make_shared<TraceCopy>(cnode->debug_info()));
  auto new_cnode = graph->NewCNodeWithInfos(cnode_inputs, cnode);
  return new_cnode;
}

CNodePtr KernelGraphMgr::CreateNewCNode(const CNodePtr &cnode, KernelGraph *graph) {
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
  MS_EXCEPTION_IF_NULL(new_cnode);
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

CNodePtr KernelGraphMgr::CreateSwitchInput(const CNodePtr &cnode, const AnfNodePtr &node_input, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(node_input);
  MS_EXCEPTION_IF_NULL(graph);
  // switch input generalizes partial
  std::vector<AnfNodePtr> partial_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimPartial->name()))};
  if (common::AnfAlgo::CheckPrimitiveType(node_input, prim::kPrimPartial)) {
    auto backend_node = graph->GetBackendAnfByFrontAnf(node_input);
    MS_EXCEPTION_IF_NULL(backend_node);
    return backend_node->cast<CNodePtr>();
  } else if (node_input->isa<ValueNode>() && IsValueNode<FuncGraph>(node_input)) {
    partial_inputs.emplace_back(graph->GetBackendAnfByFrontAnf(node_input));
  } else {
    KernelGraphPtr kernel_graph = NewKernelGraph();
    MS_EXCEPTION_IF_NULL(kernel_graph);
    auto parameter = CreateNewParameterFromCNode(cnode, kernel_graph.get());
    MS_EXCEPTION_IF_NULL(parameter);
    MS_EXCEPTION_IF_NULL(cnode);
    parameter->set_abstract(cnode->abstract());
    auto primitive = NewValueNode(std::make_shared<Primitive>(prim::kPrimReturn->name()));
    auto return_node = kernel_graph->NewCNode({primitive, parameter});
    MS_EXCEPTION_IF_NULL(return_node);
    return_node->set_abstract(cnode->abstract());
    kernel_graph->set_return(return_node);
    partial_inputs.emplace_back(std::make_shared<ValueNode>(kernel_graph));
    partial_inputs.emplace_back(graph->GetBackendAnfByFrontAnf(node_input));
  }
  auto partial_node = graph->NewCNode(partial_inputs);
  return partial_node;
}

std::vector<AnfNodePtr> KernelGraphMgr::CreateCallSwitchInputs(const CNodePtr &cnode, KernelGraph *graph) const {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> cnode_inputs = {
    graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(prim::kPrimCall->name())))};
  auto attr_input = cnode->input(kAnfPrimitiveIndex);
  MS_EXCEPTION_IF_NULL(attr_input);
  auto cnode_input = graph->GetBackendAnfByFrontAnf(attr_input);
  MS_EXCEPTION_IF_NULL(cnode_input);
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
    MS_EXCEPTION_IF_NULL(node);
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

void KernelGraphMgr::ProcessNodeRetFunc(const CNodePtr &cnode, KernelGraph *graph,
                                        const std::vector<AnfNodePtr> &real_inputs) {
  MS_EXCEPTION_IF_NULL(cnode);
  // func1 =switch(branch1, branch2)
  // func2 = func1(param1)
  // out = func2(param2)
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
    MS_EXCEPTION_IF_NULL(return_input_cnode);
    auto partial_inputs = return_input_cnode->inputs();
    (void)call_inputs.insert(call_inputs.cend(), partial_inputs.cbegin() + kFirstDataInputIndex, partial_inputs.cend());
  } else if (IsValueNode<KernelGraph>(return_input)) {  // return node is kernel graph
    call_inputs.emplace_back(return_input);
  } else {  // return node is value node
    KernelGraphPtr kernel_graph = NewKernelGraph();
    MS_EXCEPTION_IF_NULL(kernel_graph);
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
  MS_EXCEPTION_IF_NULL(call_node);
  call_node->set_abstract(cnode->abstract());
  // update return input
  ret->set_input(kFirstDataInputIndex, call_node);
}

std::vector<AnfNodePtr> KernelGraphMgr::CreateCallSwitchLayerInputs(const CNodePtr &cnode, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> cnode_inputs = {
    graph->NewValueNode(NewValueNode(std::make_shared<Primitive>(prim::kPrimCall->name())))};
  auto attr_input = cnode->input(kAnfPrimitiveIndex);
  MS_EXCEPTION_IF_NULL(attr_input);
  auto cnode_input = graph->GetBackendAnfByFrontAnf(attr_input);
  MS_EXCEPTION_IF_NULL(cnode_input);
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
    (void)new_partial_inputs.insert(new_partial_inputs.cend(), real_inputs.cbegin(), real_inputs.cend());
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

std::vector<AnfNodePtr> KernelGraphMgr::CreateSwitchOrPartialNode(const CNodePtr &cnode, KernelGraph *graph) {
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
  } else if (common::AnfAlgo::CheckPrimitiveType(cnode_input, prim::kPrimTupleGetItem)) {
    // only support tuple get item from a call subgraph output
    auto tuple_get_node = cnode_input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(tuple_get_node);
    auto get_from_node = tuple_get_node->input(kFirstIndex);
    MS_EXCEPTION_IF_NULL(get_from_node);
    if (common::AnfAlgo::CheckPrimitiveType(get_from_node, prim::kPrimCall)) {
      auto call_node = get_from_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(call_node);
      auto call_graph = call_node->input(kFirstIndex);
      auto sub_kernel_graph = GetValueNodeKernelGraph(call_graph);
      MS_EXCEPTION_IF_NULL(sub_kernel_graph);
      if (kernel_graph_partial_map_.find(sub_kernel_graph.get()) == kernel_graph_partial_map_.end()) {
        MS_LOG(EXCEPTION) << "Kernel Graph: " << sub_kernel_graph->ToString()
                          << " has not a return value is a Partial Func.";
      }
      auto tuple_get_idx = common::AnfAlgo::GetTupleGetItemOutIndex(tuple_get_node);
      auto info = kernel_graph_partial_map_[sub_kernel_graph.get()];
      call_node->set_abstract(info.abstract);
      (void)cnode_inputs.emplace_back(info.sub_graph);
      if (common::GetEnv("MS_DEV_GRAPH_REUSE") == "2") {
        // call_graph and info.sub_graph need inline when cell reuse.
        sub_kernel_graph->set_need_inline(true);
        auto partial_sub_graph = GetValueNodeKernelGraph(info.sub_graph);
        MS_EXCEPTION_IF_NULL(partial_sub_graph);
        partial_sub_graph->set_need_inline(true);
        MS_LOG(INFO) << "Inline graph " << sub_kernel_graph->graph_id() << " and graph "
                     << partial_sub_graph->graph_id();
      }
      MS_LOG(INFO) << "Use cell reuse: " << sub_kernel_graph->graph_id();
      if (info.param_begin != tuple_get_idx + std::max(static_cast<int>(info.multi_tuple) - 1, 0)) {
        MS_LOG(EXCEPTION) << "Call param is not a graph, the TupleGetItem index: " << tuple_get_idx
                          << ", the partial graph index: " << info.param_begin
                          << ", need idx: " << tuple_get_idx + std::max(static_cast<int>(info.multi_tuple) - 1, 0)
                          << ", call graph: " << call_graph->fullname_with_scope();
      }
      for (size_t i = info.param_begin; i < info.param_end; i++) {
        auto idx = NewValueNode(SizeToLong(i));
        MS_EXCEPTION_IF_NULL(idx);
        auto imm = std::make_shared<Int64Imm>(i);
        idx->set_abstract(std::make_shared<abstract::AbstractScalar>(imm));
        auto getitem = graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), call_node, idx});
        std::vector<TypeId> types = {common::AnfAlgo::GetOutputInferDataType(call_node, i)};
        auto shapes = {common::AnfAlgo::GetOutputInferShape(call_node, i)};
        common::AnfAlgo::SetOutputInferTypeAndShape(types, shapes, getitem.get());
        (void)cnode_inputs.emplace_back(getitem);
      }
      return cnode_inputs;
    }
  }
  MS_LOG(ERROR) << "CNode:" << cnode->DebugString() << " input[0]" << cnode_input->DebugString()
                << "must be partial or switch or switch_layer.";
  return {};
}

std::vector<AnfNodePtr> KernelGraphMgr::CreateValueNode(const CNodePtr &cnode, KernelGraph *graph) {
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

void KernelGraphMgr::CreateCNodeInputs(const CNodePtr &cnode, KernelGraph *graph,
                                       std::vector<AnfNodePtr> *cnode_inputs) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(cnode_inputs);
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

ValueNodePtr KernelGraphMgr::CreateValueNodeKernelGraph(const AnfNodePtr &anf, KernelGraph *graph) {
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
  MS_EXCEPTION_IF_NULL(new_value_node);
  new_value_node->set_abstract(value_node->abstract());
  // create new kernel_info of new value_node
  auto kernel_info = std::make_shared<device::KernelInfo>();
  MS_EXCEPTION_IF_NULL(kernel_info);
  new_value_node->set_kernel_info(kernel_info);
  // create kernel_build_info for new value node
  auto kernel_build_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  MS_EXCEPTION_IF_NULL(kernel_build_info_builder);
  AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_builder->Build(), new_value_node.get());
  AnfAlgo::SetGraphId(graph->graph_id(), new_value_node.get());

  graph->FrontBackendMapAdd(anf, new_value_node);

  return new_value_node;
}

ParameterPtr KernelGraphMgr::CreateNewParameter(const AnfNodePtr &anf, KernelGraph *graph) const {
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

void KernelGraphMgr::FlattenTuple(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimCall)) {
    auto call_graph = node->input(kFirstIndex);
    auto sub_kernel_graph = GetValueNodeKernelGraph(call_graph);
    MS_EXCEPTION_IF_NULL(sub_kernel_graph);
    auto iter = kernel_graph_partial_map_.find(sub_kernel_graph.get());
    if (iter != kernel_graph_partial_map_.end() && iter->second.multi_tuple != 0) {
      (void)need_flatten_.insert(node);
    }
  } else if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimTupleGetItem)) {
    auto input = node->input(kFirstIndex);
    auto get_idx = common::AnfAlgo::GetTupleGetItemOutIndex(node);
    if (need_flatten_.find(input) != need_flatten_.end() && get_idx == 0) {
      need_flatten_tuple_map_[node] = input;
    }
  }
  for (size_t i = 0; i < common::AnfAlgo::GetInputNum(node); i++) {
    auto input = common::AnfAlgo::GetInputNode(node, i);
    auto iter = need_flatten_tuple_map_.find(input);
    if (iter != need_flatten_tuple_map_.end()) {
      node->set_input(i + 1, iter->second);
    }
  }
}

bool KernelGraphMgr::CreateCNodeOfKernelGraph(const AnfNodePtr &node, KernelGraph *graph) {
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
  std::string fullname = cnode->fullname_with_scope();
  new_cnode->set_fullname_with_scope(fullname);
  new_cnode->set_scope(cnode->scope());
  graph->FrontBackendMapAdd(node, new_cnode);
  SetReturnNode(new_cnode, graph);
  FlattenTuple(new_cnode);
  return true;
}

void KernelGraphMgr::AddParameterToGraphInputs(const std::vector<AnfNodePtr> &parameters, KernelGraph *graph) const {
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

// 1. Convert the node to make_tuple if the node is a ValueNode<ValueTuple> and it's the input of 'return' node.
// 2. Set the return of graph if node is "Return" node.
// 3. If the return of graph has a Partial Func, should inline it in return value.
void KernelGraphMgr::SetReturnNode(const AnfNodePtr &node, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  if (!common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimReturn)) {
    return;
  }
  constexpr auto kReturnInputIdx = 1;
  auto return_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(return_node);
  graph->set_return(return_node);
  auto graph_output = return_node->input(kReturnInputIdx);
  MS_EXCEPTION_IF_NULL(graph_output);

  // If return's input is value node, then the graph has no kernel, and the pass 'trans tuple to make_tuple' cannot
  // match this pattern because that pass begin with output node but return node. So we add transform value tuple
  // to make_tuple here.
  if (common::AnfAlgo::IsTupleOutput(graph_output) && graph_output->isa<ValueNode>()) {
    return_node->set_input(kReturnInputIdx, graph->TransTupleToMakeTuple(graph_output));
  }

  // inline partial to call graph
  auto return_tuple = return_node->input(kReturnInputIdx);
  MS_EXCEPTION_IF_NULL(return_tuple);
  if (return_tuple->isa<CNode>() && common::AnfAlgo::CheckPrimitiveType(return_tuple, prim::kPrimMakeTuple)) {
    auto make_tuple = return_tuple->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(make_tuple);
    size_t tuple_input_num = common::AnfAlgo::GetInputTensorNum(make_tuple);
    // only support the last return node is a partial func now
    auto last_input_node = common::AnfAlgo::GetInputNode(make_tuple, tuple_input_num - 1);
    MS_EXCEPTION_IF_NULL(last_input_node);
    if (last_input_node->isa<CNode>() && common::AnfAlgo::CheckPrimitiveType(last_input_node, prim::kPrimPartial)) {
      size_t multi_tuple = 0;
      auto partial_node = last_input_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(partial_node);
      size_t partial_input_num = common::AnfAlgo::GetInputTensorNum(partial_node);
      std::vector<AnfNodePtr> make_tuple_inputs;
      (void)make_tuple_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
      // skip last return node (is a partial)
      size_t param_begin = 0;
      for (size_t i = 0; i < tuple_input_num - 1; i++) {
        auto input = common::AnfAlgo::GetInputNode(make_tuple, i);
        auto node_abs = input->abstract();
        MS_EXCEPTION_IF_NULL(node_abs);
        if (node_abs->isa<abstract::AbstractSequence>()) {
          MS_EXCEPTION_IF_CHECK_FAIL(
            i == 0, "Input index: " + std::to_string(i) + " is a make tuple, input node: " + input->DebugString());
          MS_LOG(DEBUG) << "Flatten the make tuple, input node: " << input->DebugString()
                        << ", output num: " << AnfUtils::GetOutputTensorNum(input);
          // flatten the make tuple
          for (size_t j = 0; j < AnfUtils::GetOutputTensorNum(input); j++) {
            auto idx = NewValueNode(SizeToLong(j));
            MS_EXCEPTION_IF_NULL(idx);
            auto imm = std::make_shared<Int64Imm>(j);
            idx->set_abstract(std::make_shared<abstract::AbstractScalar>(imm));
            auto getitem = graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), input, idx});
            std::vector<TypeId> types = {common::AnfAlgo::GetOutputInferDataType(input, j)};
            auto shapes = {common::AnfAlgo::GetOutputInferShape(input, j)};
            common::AnfAlgo::SetOutputInferTypeAndShape(types, shapes, getitem.get());
            param_begin++;
            multi_tuple++;
            (void)make_tuple_inputs.emplace_back(getitem);
          }
        } else {
          param_begin++;
          (void)make_tuple_inputs.emplace_back(input);
        }
      }
      // skip partial graph
      for (size_t i = kFirstIndex; i < partial_input_num; i++) {
        (void)make_tuple_inputs.emplace_back(common::AnfAlgo::GetInputNode(partial_node, i));
      }
      auto g_output = graph->NewCNode(make_tuple_inputs);
      MS_EXCEPTION_IF_NULL(g_output);
      std::vector<AbstractBasePtr> abstract_list;
      for (size_t i = kFirstIndex; i < make_tuple_inputs.size(); ++i) {
        auto inputs_node = make_tuple_inputs[i];
        MS_EXCEPTION_IF_NULL(inputs_node);
        (void)abstract_list.emplace_back(inputs_node->abstract());
      }
      auto abstract = std::make_shared<abstract::AbstractTuple>(abstract_list);
      MS_EXCEPTION_IF_NULL(g_output);
      g_output->set_abstract(abstract);
      graph->set_output(g_output);

      kernel_graph_partial_map_[graph] = {abstract, common::AnfAlgo::GetInputNode(partial_node, 0), param_begin,
                                          common::AnfAlgo::GetInputTensorNum(g_output), multi_tuple};
    }
  }
}

KernelGraphPtr KernelGraphMgr::ConstructKernelGraph(const AnfNodePtrList &lst, const AnfNodePtrList &outputs,
                                                    DeviceType device_target, bool common_opt,
                                                    bool is_enable_zero_copy) {
  mindspore::HashMap<AnfNodePtr, AnfNodePtr> other_graph_cnode;
  auto graph = NewKernelGraph();
  MS_EXCEPTION_IF_NULL(graph);
  // Set the zero copy flag in subgraph sink mode.
  if (is_enable_zero_copy) {
    MS_LOG(INFO) << "Set zero copy flag for graph:" << graph->ToString();
    graph->set_flag(kFlagEnableZeroCopyInGraph, true);
  }
  MS_LOG(INFO) << "Create graph: " << graph->graph_id();
  for (const auto &node : lst) {
    MS_EXCEPTION_IF_NULL(node);
    MS_LOG(DEBUG) << "Start create new cnode, node = " << node->DebugString();
    if (!node->isa<CNode>()) {
      MS_LOG(EXCEPTION) << "Node " << node->DebugString() << " is not CNode";
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    graph->set_device_target(device_target);
    // create a new cnode object
    auto new_cnode = CreateNewCNode(cnode, graph.get(), &other_graph_cnode);
    MS_EXCEPTION_IF_NULL(new_cnode);
    new_cnode->set_abstract(cnode->abstract());
    new_cnode->set_scope(cnode->scope());
    new_cnode->set_attrs(cnode->attrs());
    // record map relations between anf from ME and new anf node used in backend
    graph->FrontBackendMapAdd(node, new_cnode);
  }
  // add a make_tuple at the end of graph as output
  graph->set_output(ConstructOutput(outputs, graph));
  FuncGraphManagerPtr manager = MakeManager({graph});
  if (manager) {
    manager->AddFuncGraph(graph);
    graph->set_manager(manager);
  }
  graph->SetExecOrderByDefault();

#ifndef ENABLE_SECURITY
  if (ExistSummaryNode(graph.get())) {
    graph->set_summary_node_exist(true);
  }
#endif

  MS_EXCEPTION_IF_NULL(MsContext::GetInstance());
  if (!MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
    UnifyMindIR(graph);
    graph->UpdateGraphAquireGilAttr();
    if (common_opt) {
      opt::BackendCommonOptimization(graph);
    }
    graph->SetInputNodes();
    SetInputNodeUsage(graph, manager);
    graph->SetOptimizerFlag();
  }
  graph->set_parameters(graph->inputs());
  return graph;
}

std::shared_ptr<KernelGraph> KernelGraphMgr::ConstructKernelGraph(const FuncGraphPtr &func_graph,
                                                                  std::vector<KernelGraphPtr> *all_out_graph,
                                                                  DeviceType device_target) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(all_out_graph);
  auto node_list = TopoSort(func_graph->get_return());
  auto graph = NewKernelGraph();
  MS_EXCEPTION_IF_NULL(graph);
  front_backend_graph_map_[func_graph.get()] = graph;
  if (func_graph->has_flag(FUNC_GRAPH_FLAG_NEED_BACKEND_INLINE)) {
    MS_LOG(INFO) << "Need backend inline: " << graph->graph_id();
    graph->set_need_inline(true);
  }
  MS_LOG(INFO) << "Create graph: " << graph->graph_id();
  graph->set_device_target(device_target);
  // Create parameter
  for (const auto &node : func_graph->parameters()) {
    MS_EXCEPTION_IF_NULL(node);
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
      MS_LOG(EXCEPTION) << "Construct func graph " << func_graph->ToString() << " failed."
                        << trace::DumpSourceLines(node);
    }
  }

  AddParameterToGraphInputs(func_graph->parameters(), graph.get());
  FuncGraphManagerPtr manager = MakeManager({graph});
  graph->SetInputNodes();
  SetInputNodeUsage(graph, manager);
  graph->SetExecOrderByDefault();

#ifndef ENABLE_SECURITY
  if (ExistSummaryNode(graph.get())) {
    graph->set_summary_node_exist(true);
  }
#endif

  all_out_graph->push_back(graph);
  graph->set_parameters(graph->inputs());
  return graph;
}

void KernelGraphMgr::SetInputNodeUsage(const KernelGraphPtr &graph, const FuncGraphManagerPtr &manager) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(manager);
  auto input_nodes = graph->input_nodes();
  for (auto &input_node : input_nodes) {
    MS_EXCEPTION_IF_NULL(input_node);
    if (input_node->isa<Parameter>()) {
      auto node_ptr = input_node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(node_ptr);
      if (!IsUsedByRealKernel(manager, input_node, graph->graph_id())) {
        node_ptr->SetNotUsedByRealKernelInGraph(graph->graph_id());
      }
      auto shape = node_ptr->Shape();
      MS_EXCEPTION_IF_NULL(shape);
      if (shape->isa<abstract::Shape>() && shape->IsDynamic()) {
        node_ptr->set_has_dynamic_shape(true);
      }
      if (input_node->abstract() != nullptr && input_node->abstract()->isa<abstract::AbstractSequence>()) {
        // If the parameter is dynamic sequence, it is regard as dynamic shape.
        const auto &tuple_abs = input_node->abstract()->cast<abstract::AbstractSequencePtr>();
        MS_EXCEPTION_IF_NULL(tuple_abs);
        if (tuple_abs->dynamic_len()) {
          MS_LOG(INFO) << "Input node:" << input_node->DebugString() << " set dynamic flag to true";
          node_ptr->set_has_dynamic_shape(true);
        }
      }
    }
  }
}

namespace {
bool CNodeFirstInputIsPrimitive(const AnfNodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    return false;
  }
  auto prim = cnode->input(kAnfPrimitiveIndex);
  if (prim == nullptr || !IsValueNode<Primitive>(prim)) {
    return false;
  }
  return true;
}

std::vector<AnfNodePtr> ExtendNodeUsers(const FuncGraphManagerPtr &front_func_graph_manager,
                                        const AnfNodePtr &front_node) {
  MS_EXCEPTION_IF_NULL(front_func_graph_manager);
  auto &users = front_func_graph_manager->node_users()[front_node];
  std::vector<AnfNodePtr> result;
  for (auto &user : users) {
    if (common::AnfAlgo::CheckPrimitiveType(user.first, prim::kPrimDepend) ||
        common::AnfAlgo::CheckPrimitiveType(user.first, prim::kPrimLoad)) {
      auto depend_cnode = user.first->cast<CNodePtr>();
      if (depend_cnode == nullptr) {
        continue;
      }
      if (front_node != depend_cnode->input(1)) {
        continue;
      }
      auto res = ExtendNodeUsers(front_func_graph_manager, user.first);
      (void)result.insert(result.cend(), res.cbegin(), res.cend());
    } else if (common::AnfAlgo::CheckPrimitiveType(user.first, prim::kPrimMakeTuple)) {
      auto res = ExtendNodeUsers(front_func_graph_manager, user.first);
      (void)result.insert(result.cend(), res.cbegin(), res.cend());
    } else {
      (void)result.emplace_back(user.first);
    }
  }
  return result;
}

AnfNodePtr GetSupportedInternalNode(const AnfNodePtr &front_node) {
  MS_EXCEPTION_IF_NULL(front_node);
  if (!front_node->isa<CNode>()) {
    return nullptr;
  }
  if (AnfUtils::IsRealKernel(front_node)) {
    return front_node;
  }
  if (common::AnfAlgo::CheckPrimitiveType(front_node, prim::kPrimTupleGetItem)) {
    return front_node;
  }
  if (common::AnfAlgo::CheckPrimitiveType(front_node, prim::kPrimMakeTuple)) {
    auto cnode = front_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto &inputs = cnode->inputs();
    if (inputs.size() > 1) {
      return GetSupportedInternalNode(inputs[1]);
    }
  }
  if (common::AnfAlgo::CheckPrimitiveType(front_node, prim::kPrimDepend)) {
    auto cnode = front_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto &inputs = cnode->inputs();
    if (inputs.size() >= kDependInputSize) {
      return GetSupportedInternalNode(inputs[kRealInputIndexInDepend]);
    }
  }
  return nullptr;
}

bool IsUnusedInternlOutput(const AnfNodePtr &user) {
  if (!CNodeFirstInputIsPrimitive(user)) {
    return true;
  }
  if (IsPrimitiveCNode(user, prim::kPrimSwitch) || IsPrimitiveCNode(user, prim::kPrimSwitchLayer)) {
    return true;
  }
  if (!AnfUtils::IsRealKernel(user)) {
    return true;
  }
  return false;
}
}  // namespace

constexpr auto kMixTarget = "MixTarget";
constexpr auto kNoTarget = "NoTarget";
std::string KernelGraphMgr::AddPartialParametersMap(const AnfNodePtr &partial_node) {
  MS_EXCEPTION_IF_NULL(partial_node);
  auto iter = partial_target_map_.find(partial_node);
  if (iter != partial_target_map_.end()) {
    return iter->second;
  }
  auto partial_cnode = partial_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(partial_cnode);
  auto partial_graph = GetValueNode<FuncGraphPtr>(partial_cnode->input(kFirstDataInputIndex));
  // If graph is nullptr, it means that the funcgraph in the partial node is a deadnode, and the processing is skipped.
  if (partial_graph == nullptr) {
    return kNoTarget;
  }
  auto parameters = partial_graph->parameters();
  auto partial_inputs = partial_cnode->inputs();
  const size_t kNonParameterNum = 2;
  if (parameters.size() + kNonParameterNum != partial_inputs.size()) {
    return kMixTarget;
  }
  for (size_t i = 0; i < parameters.size(); ++i) {
    partial_parameters_map_[parameters[i]] = partial_inputs[kNonParameterNum + i];
  }
  auto graph_nodes = TopoSort(partial_graph->get_return());
  std::string graph_target = kNoTarget;
  for (auto &node : graph_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    if (!AnfUtils::IsRealKernel(node)) {
      continue;
    }
    std::string cur_target = GetCNodeTarget(node);
    if (graph_target == kNoTarget) {
      graph_target = cur_target;
    }
    if (graph_target != cur_target) {
      graph_target = kMixTarget;
      break;
    }
  }
  (void)partial_target_map_.emplace(std::pair<AnfNodePtr, std::string>(partial_node, graph_target));
  return graph_target;
}

namespace {
bool IsNeedAddPartialParameter(const AnfNodePtr &user, const std::string &kernel_target,
                               const std::shared_ptr<KernelGraph> &graph) {
  // If the flag is enable, it means the graph would run in subgraph sink mode, the real parameter on partial
  // cannot share the same device address with the formal parameter.
  MS_EXCEPTION_IF_NULL(graph);
  return common::AnfAlgo::CheckPrimitiveType(user, prim::kPrimPartial) && kernel_target != kGPUDevice &&
         !ExistGraphCaller(user) && (!graph->has_flag(kFlagEnableZeroCopyInGraph));
}
}  // namespace

void KernelGraphMgr::HandleInternalOutput(const AnfNodePtr &input_front_node, const AnfNodePtr &backend_node,
                                          const FuncGraphManagerPtr &front_func_graph_manager,
                                          const std::shared_ptr<KernelGraph> &backend_graph) {
  MS_EXCEPTION_IF_NULL(backend_graph);
  if (device::KernelRuntime::UseMemScheduler()) {
    return;
  }
  auto front_node = GetSupportedInternalNode(input_front_node);
  if (front_node == nullptr) {
    return;
  }
  auto front_real_kernel_pair = common::AnfAlgo::VisitKernel(front_node, 0);
  auto backend_real_kernel_pair = common::AnfAlgo::VisitKernel(backend_node, 0);
  auto backend_real_kernel = backend_real_kernel_pair.first;
  if (backend_real_kernel == nullptr || !backend_real_kernel->isa<CNode>()) {
    return;
  }
  auto front_real_kernel = front_real_kernel_pair.first;
  std::string kernel_target = GetCNodeTarget(front_real_kernel);
  bool internal_output = CNodeFirstInputIsPrimitive(front_real_kernel);
  bool unique_target = true;
  if (internal_output && common::AnfAlgo::IsNopNode(front_real_kernel)) {
    auto pre_node_pair = common::AnfAlgo::GetPrevNodeOutput(front_real_kernel, 0);
    auto pre_node_target = GetCNodeTarget(pre_node_pair.first);
    if (pre_node_target != kernel_target) {
      unique_target = false;
    }
  }
  if (internal_output) {
    auto users = ExtendNodeUsers(front_func_graph_manager, front_node);
    for (auto &user : users) {
      if (IsNeedAddPartialParameter(user, kernel_target, backend_graph)) {
        auto partial_target = AddPartialParametersMap(user);
        if (partial_target != kNoTarget && partial_target != kernel_target) {
          unique_target = false;
        }
        continue;
      }
      if (common::AnfAlgo::CheckPrimitiveType(user, prim::kPrimUpdateState)) {
        continue;
      }
      if (IsUnusedInternlOutput(user)) {
        internal_output = false;
        break;
      }
      if (kernel_target != GetCNodeTarget(user)) {
        unique_target = false;
      }
    }
  }
  if (internal_output) {
    MS_LOG(INFO) << "AddInternalOutput: " << front_node->DebugString() << " To " << backend_real_kernel->DebugString()
                 << ", unique_target: " << unique_target;
    backend_graph->AddInternalOutput(front_node, backend_real_kernel, backend_real_kernel_pair.second, unique_target);
  }
}

CNodePtr KernelGraphMgr::ConstructOutput(const AnfNodePtrList &outputs, const std::shared_ptr<KernelGraph> &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> output_args;
  for (const auto &output : outputs) {
    MS_EXCEPTION_IF_NULL(output);
    MS_LOG(INFO) << "Output:" << output->DebugString();
  }
  auto FindEqu = [graph, outputs, this](const AnfNodePtr &out) -> AnfNodePtr {
    auto backend_anf = graph->GetBackendAnfByFrontAnf(out);
    if (backend_anf != nullptr) {
      auto context_ptr = MsContext::GetInstance();
      MS_EXCEPTION_IF_NULL(context_ptr);
      if (context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
        return backend_anf;
      }

      MS_EXCEPTION_IF_NULL(out);
      auto out_func_graph = out->func_graph();
      MS_EXCEPTION_IF_NULL(out_func_graph);
      auto out_func_graph_manager = out_func_graph->manager();
      if (out_func_graph_manager == nullptr) {
        return backend_anf;
      }
      HandleInternalOutput(out, backend_anf, out_func_graph_manager, graph);
      return backend_anf;
    }
    MS_LOG(EXCEPTION) << "Can't find the node in the equiv map!";
  };
  output_args.push_back(mindspore::NewValueNode(std::make_shared<Primitive>(prim::kPrimMakeTuple->name())));
  (void)std::transform(outputs.begin(), outputs.end(), std::back_inserter(output_args),
                       [&](const AnfNodePtr &out) -> AnfNodePtr { return FindEqu(out); });
  auto output_node = graph->NewCNode(output_args);
  MS_EXCEPTION_IF_NULL(output_node);
  // Create abstract for output maketuple node.
  AbstractBasePtrList output_abs_list;
  const auto &inputs = output_node->inputs();
  (void)std::transform(
    inputs.begin() + 1, inputs.end(), std::back_inserter(output_abs_list), [](const AnfNodePtr &input) {
      return input->abstract() == nullptr ? std::make_shared<abstract::AbstractNone>() : input->abstract();
    });
  auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(output_abs_list);
  MS_EXCEPTION_IF_NULL(abstract_tuple);
  output_node->set_abstract(abstract_tuple);
  return output_node;
}

KernelGraphPtr KernelGraphMgr::NewKernelGraph() {
  auto graph = std::make_shared<KernelGraph>();
  MS_EXCEPTION_IF_NULL(graph);
  graph->set_graph_id(graph_sum_);
  graphs_[graph_sum_++] = graph;
  return graph;
}

void KernelGraphMgr::UnifyMindIR(const KernelGraphPtr &graph) { opt::CommonUnifyMindIR(graph); }
}  // namespace session
}  // namespace mindspore
