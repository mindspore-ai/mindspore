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

#include "runtime/framework/control_node_parser.h"
#include "abstract/utils.h"
#include "ir/tensor.h"

namespace mindspore {
namespace runtime {
namespace {
// Get all the real parameters corresponding to node.
void FetchRealParameterByNode(const KernelWithIndex &node, std::set<KernelWithIndex> *real_parameters,
                              std::set<KernelWithIndex> *invalid_call_nodes) {
  auto node_with_index = node;
  if (!node.first->isa<ValueNode>()) {
    node_with_index = AnfAlgo::VisitKernelWithReturnType(node.first, node.second);
  }
  if (node_with_index.first->isa<ValueNode>() || node_with_index.first->isa<Parameter>()) {
    // If node is a valuenode or parameter, the real parameter is itself.
    real_parameters->emplace(node_with_index);
  } else if (AnfAlgo::IsCallNode(node_with_index.first)) {
    // If node is a call node, the real parameters are the outputs of funcgraph the node called.
    if (invalid_call_nodes->find(node_with_index) != invalid_call_nodes->end()) {
      return;
    }
    invalid_call_nodes->emplace(node_with_index);
    const auto &func_graphs = AnfAlgo::GetFuncGraphbyCallNode(node_with_index.first);
    for (const auto &func_graph : func_graphs) {
      MS_EXCEPTION_IF_NULL(func_graph);
      FetchRealParameterByNode({func_graph->output(), node_with_index.second}, real_parameters, invalid_call_nodes);
    }
  } else if (AnfAlgo::CheckPrimitiveType(node_with_index.first, prim::kPrimMakeTuple)) {
    // If node is a maketuple node, the real parameters are its total inputs.
    const auto &make_tuple_cnode = node_with_index.first->cast<CNodePtr>();
    const auto &make_tuple_inputs = make_tuple_cnode->inputs();
    if (make_tuple_inputs.size() <= node_with_index.second) {
      MS_LOG(EXCEPTION) << "Invalid index:" << node_with_index.second
                        << "for tuple node:" << node_with_index.first->DebugString();
    }
  } else if (AnfAlgo::CheckPrimitiveType(node.first, prim::kPrimSwitch)) {
    // If node is a switch node, the real parameters are its both true and false branches.
    const auto cnode = node_with_index.first->cast<CNodePtr>();
    const auto inputs = cnode->inputs();
    for (size_t i = kSwitchTrueBranchPos; i < inputs.size(); ++i) {
      FetchRealParameterByNode({inputs[i], 0}, real_parameters, invalid_call_nodes);
    }
  } else if (AnfAlgo::CheckPrimitiveType(node_with_index.first, prim::kPrimSwitchLayer)) {
    // If node is a switchlyaer node, the real parameters are its total branches.
    const auto &switch_layer_cnode = node_with_index.first->cast<CNodePtr>();
    const auto &switch_layer_inputs = switch_layer_cnode->inputs();
    if (switch_layer_inputs.size() != kSwitchLayerInputNum ||
        (!AnfAlgo::CheckPrimitiveType(switch_layer_inputs[kSwitchLayerBranchPos], prim::kPrimMakeTuple))) {
      MS_LOG(EXCEPTION) << "Invalid switch layer node:" << switch_layer_cnode->DebugString();
    }
    const auto &make_tuple_cnode = switch_layer_inputs[kSwitchLayerBranchPos]->cast<CNodePtr>();
    const auto &make_tuple_inputs = make_tuple_cnode->inputs();
    for (size_t i = kSwitchTrueBranchPos; i < make_tuple_inputs.size(); ++i) {
      FetchRealParameterByNode({make_tuple_inputs[i], 0}, real_parameters, invalid_call_nodes);
    }
  } else {
    // If node is a kernel, the real parameter is itself.
    real_parameters->emplace(node_with_index);
  }
}

// Fetch all the weight parameters related to node. It runs like this:
// if we have a map like {{a, {b, c}}, {b, {d, e}}}, final we will get {{a, {b, c, d, e}}, {b, {c, d}}}.
void FetchWeightbyHostParameter(const AnfNodePtr &node, std::set<AnfNodePtr> *dest_nodes,
                                const HostParameterToWeight &front_to_front_weight) {
  if (dest_nodes->find(node) != dest_nodes->end()) {
    return;
  }
  dest_nodes->emplace(node);
  if (front_to_front_weight.find(node) == front_to_front_weight.end()) {
    return;
  }

  const auto weight_nodes = front_to_front_weight.at(node);
  for (const auto weight_node : weight_nodes) {
    FetchWeightbyHostParameter(weight_node, dest_nodes, front_to_front_weight);
  }
}

// Recursive interface, get the real kernel that UpdateState node depends on.
AnfNodePtr FetchSourceNodeByAutoMonad(const AnfNodePtr &node) {
  if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimUpdateState)) {
    const auto &cnode = node->cast<CNodePtr>();
    const auto &inputs = cnode->inputs();
    if (inputs.size() <= kUpdateStateRealInput) {
      MS_LOG(EXCEPTION) << "Invalid updatestate node:" << AnfAlgo::GetNodeDebugString(node);
    }

    return FetchSourceNodeByAutoMonad(inputs[kUpdateStateRealInput]);
  }
  return node;
}

// Topologically sort all funcgraphs according to the function call relationship.
std::vector<FuncGraphPtr> TopoSortForFuncGraph(const FuncGraphPtr &root, FuncGraphCallRelation *edges) {
  MS_EXCEPTION_IF_NULL(root->manager());
  std::set<FuncGraphPtr> nodes;
  nodes.emplace(root);

  FuncGraphSet subs = root->manager()->func_graphs();
  for (auto sub : subs) {
    if (sub != root && root != nullptr) {
      nodes.emplace(sub);
    }
  }

  std::queue<FuncGraphPtr> que;
  for (const auto &node : nodes) {
    if (edges->find(node) == edges->end()) {
      que.push(node);
    }
  }

  std::vector<FuncGraphPtr> result;
  while (!que.empty()) {
    const auto node = que.front();
    que.pop();
    result.emplace_back(node);
    for (auto iter = edges->begin(); iter != edges->end();) {
      auto &sub_edges = iter->second;
      for (auto sub_iter = sub_edges.begin(); sub_iter != sub_edges.end();) {
        if (sub_iter->find(node) != sub_iter->end()) {
          sub_edges.erase(sub_iter);
        } else {
          ++sub_iter;
        }
      }
      if (sub_edges.empty()) {
        que.push(iter->first);
        edges->erase(iter++);
      } else {
        ++iter;
      }
    }
  }

  return result;
}

// Fetch all output of node, and this function will not parse the call node.
std::vector<KernelWithIndex> FetchAllOutputWithIndex(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  std::vector<KernelWithIndex> result;

  if (node->isa<ValueNode>() && IsValueNode<ValueTuple>(node)) {
    const auto &value_node = node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    const auto &value = value_node->value();
    MS_EXCEPTION_IF_NULL(value);
    const auto &value_tuple = value->cast<ValueTuplePtr>();
    MS_EXCEPTION_IF_NULL(value_tuple);
    const auto tuple_value = value_tuple->value();
    for (size_t i = 0; i < tuple_value.size(); ++i) {
      result.emplace_back(node, i);
    }
    return result;
  }

  const auto node_with_index = AnfAlgo::VisitKernelWithReturnType(node, 0);
  if (AnfAlgo::CheckPrimitiveType(node_with_index.first, prim::kPrimMakeTuple)) {
    const auto &cnode = node_with_index.first->cast<CNodePtr>();
    const auto &inputs = cnode->inputs();

    for (size_t i = kMakeTupleInputStartPos; i < inputs.size(); ++i) {
      const auto &tmp_list = FetchAllOutputWithIndex(inputs[i]);
      result.insert(result.end(), tmp_list.begin(), tmp_list.end());
    }
  } else if (AnfAlgo::CheckPrimitiveType(node_with_index.first, prim::kPrimSwitch) ||
             AnfAlgo::CheckPrimitiveType(node_with_index.first, prim::kPrimSwitchLayer)) {
  } else if (AnfAlgo::IsCallNode(node)) {
    size_t output_num = AnfAlgo::GetOutputTensorNum(node);
    for (size_t i = 0; i < output_num; ++i) {
      result.emplace_back(node, i);
    }
  } else {
    result.emplace_back(node_with_index);
  }

  return result;
}

// Create a device tensor for the front node.
// Get the output format and select kernel build info from the backend node corresponding to the front node to
// create the device address.
void CreateDeviceTensorForValueNode(const KernelWithIndex &front_node_with_index, const AnfNodePtr &backend_node,
                                    const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(backend_node);
  MS_EXCEPTION_IF_NULL(device_context);
  const auto &front_node = front_node_with_index.first;
  MS_EXCEPTION_IF_NULL(front_node);

  const auto &node_value = front_node->cast<ValueNodePtr>()->value();
  if ((!node_value->isa<tensor::Tensor>()) && (!node_value->isa<ValueTuple>())) {
    return;
  }

  size_t tensor_size = AnfAlgo::GetOutputTensorMemSize(backend_node, 0);
  TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(backend_node, 0);
  if (output_type_id == kTypeUnknown) {
    output_type_id = AnfAlgo::GetOutputInferDataType(backend_node, 0);
  }

  if (front_node->kernel_info() == nullptr) {
    front_node->set_kernel_info(std::make_shared<device::KernelInfo>());
  }

  // Get the select kernel build info.
  auto kernel_info = static_cast<device::KernelInfo *>(backend_node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->GetMutableSelectKernelBuildInfo();
  MS_EXCEPTION_IF_NULL(build_info);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, front_node.get());

  // Create device tensor.
  std::string output_format = AnfAlgo::GetOutputFormat(backend_node, 0);
  device::DeviceAddressPtr address =
    device_context->CreateDeviceAddress(nullptr, tensor_size, output_format, output_type_id);
  MS_EXCEPTION_IF_NULL(address);
  MS_LOG(DEBUG) << "Create addr for node:" << AnfAlgo::GetNodeDebugString(front_node) << " addr:" << address;
  AnfAlgo::SetOutputAddr(address, front_node_with_index.second, front_node.get());
}

// Create a device tensor for front node.
// When the condition input of the switch and switchlayer or the output of a subgraph is a parameter or value node,
// there is no corresponding backend node for this parameter, so a device tensor needs to be created for it.
void CreateDeviceTensorForFrontNode(const KernelWithIndex &front_node_with_index, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(device_context);
  const auto &node = front_node_with_index.first;
  MS_EXCEPTION_IF_NULL(device_context);

  TypeId type_id = AnfAlgo::GetOutputInferDataType(node, 0);
  if (node->kernel_info() == nullptr) {
    auto kernel_info = std::make_shared<device::KernelInfo>();
    std::shared_ptr<KernelBuildInfoBuilder> builder = std::make_shared<KernelBuildInfoBuilder>();
    builder->SetOutputsFormat({kOpFormat_DEFAULT});
    builder->SetOutputsDeviceType({type_id});
    kernel_info->set_select_kernel_build_info(builder->Build());
    node->set_kernel_info(kernel_info);
  }
  size_t size = AnfAlgo::GetOutputTensorMemSize(node, 0);

  // Create device tensor.
  device::DeviceAddressPtr address = device_context->CreateDeviceAddress(nullptr, size, kOpFormat_DEFAULT, type_id);
  MS_EXCEPTION_IF_NULL(address);
  MS_LOG(DEBUG) << "Create addr for node:" << AnfAlgo::GetNodeDebugString(node) << " addr:" << address;
  AnfAlgo::SetOutputAddr(address, front_node_with_index.second, node.get());
}
}  // namespace

bool HasAbstractRef(const AnfNodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  auto &abs = node->abstract();
  return (abs != nullptr) && abs->isa<abstract::AbstractRef>();
}

KernelWithIndex GetFrontNodeByKernelGraph(const AnfNodePtr &backend_node, const KernelGraphPtr &graph) {
  const auto &front_node = graph->GetFrontAnfByBackendAnf(backend_node);
  if (front_node != nullptr) {
    return {front_node, 0};
  }
  const auto &front_node_with_index = graph->GetFrontNodeByInternalParameter(backend_node);
  if (front_node_with_index.first == nullptr) {
    MS_LOG(EXCEPTION) << "Cannot find front node for backend node:" << backend_node->DebugString()
                      << " in graph:" << graph->ToString();
  }
  return front_node_with_index;
}

void ControlNodeParser::Parse(const std::vector<AnfNodePtr> &control_nodes, const std::vector<KernelGraphPtr> &graphs,
                              const std::vector<DeviceContext *> &device_contexts, const FuncGraphPtr &root_graph,
                              const FuncGraphToKernelGraph &func_graph_to_kernel_graphs) {
  if (graphs.size() != device_contexts.size()) {
    MS_LOG(EXCEPTION) << "Graph num is not equal to device context, graph:" << graphs.size()
                      << " device context num:" << device_contexts.size();
  }

  if (control_nodes.size() <= 1 || device_contexts.empty()) {
    return;
  }

  is_inited_ = true;

  root_func_graph_ = root_graph;

  root_graph_parameters_ = root_graph->parameters();

  func_graph_to_kernel_graphs_ = func_graph_to_kernel_graphs;

  CreateBranchIDForCallNode(control_nodes);

  ParseCallNodeToFuncGraph(control_nodes);

  FetchFrontNodeToKernelGraph(graphs);

  ParseFormalToRealParameter(control_nodes);

  ParseFrontToBackendParameter(graphs, device_contexts);

  FetchHostParameterToWeight();

  FetchCallInputKernelGraph(graphs, device_contexts);

  FetchFrontValueNode(device_contexts[0]);

  FetchFrontToBackendKernel(graphs, device_contexts);

  ParseDeviceContext(control_nodes, graphs, device_contexts, func_graph_to_kernel_graphs);

  FetchControlNodeParameter(control_nodes);

  FetchAutoMonadNode(control_nodes);

  ParseFirstControlNodeForFuncGraph(control_nodes);
}

bool ControlNodeParser::IsControlFlowDataArrow(const KernelGraphPtr &graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(graph);
  if ((!IsInited()) || (!node->isa<Parameter>())) {
    return false;
  }

  // If the graph has a call input, all of its inputs in the graph should be Linked to its stack actor.
  if (IsCallInputKernelGraph(graph)) {
    return true;
  }

  // Parameter input should be Linked to its entrance actor.
  const auto &front_node = graph->GetFrontAnfByBackendAnf(node);
  return front_node != nullptr && front_node->isa<Parameter>() &&
         (!AnfAlgo::IsParameterWeight(front_node->cast<ParameterPtr>()));
}

void ControlNodeParser::ParseDeviceContext(const std::vector<AnfNodePtr> &control_nodes,
                                           const std::vector<KernelGraphPtr> &kernel_graphs,
                                           const std::vector<DeviceContext *> &device_contexts,
                                           const FuncGraphToKernelGraph &func_graph_to_kernel_graphs) {
  if (device_contexts.empty()) {
    MS_LOG(EXCEPTION) << "Invalid device contexts.";
  }

  ParseDeviceContextForFuncGraph(control_nodes, kernel_graphs, device_contexts, func_graph_to_kernel_graphs);
  ParseDeviceContextForControlNode(device_contexts[0]);
}

void ControlNodeParser::ParseDeviceContextForFuncGraph(const std::vector<AnfNodePtr> &control_nodes,
                                                       const std::vector<KernelGraphPtr> &kernel_graphs,
                                                       const std::vector<DeviceContext *> &device_contexts,
                                                       const FuncGraphToKernelGraph &func_graph_to_kernel_graphs) {
  std::unordered_map<KernelGraphPtr, DeviceContext *> kernel_graph_to_device_context;
  for (size_t i = 0; i < kernel_graphs.size(); ++i) {
    kernel_graph_to_device_context[kernel_graphs[i]] = device_contexts[i];
  }
  const auto &default_context = device_contexts[0];

  // Collect the device context type of the parameter in the kernel graph as the type of the real parameters.
  for (const auto &func_graph_to_kernel_graph : func_graph_to_kernel_graphs) {
    const auto &func_graph = func_graph_to_kernel_graph.first;
    const auto &front_parameters = func_graph->parameters();
    std::vector<const DeviceContext *> parameter_device_contexts(front_parameters.size(), nullptr);
    std::unordered_map<AnfNodePtr, DeviceContext *> front_parameter_to_device_context;

    for (const auto &kernel_graph : func_graph_to_kernel_graph.second) {
      const auto &backend_parameters = kernel_graph->parameters();

      for (const auto &backend_parameter : backend_parameters) {
        const auto &front_parameter = kernel_graph->GetBackendAnfByFrontAnf(backend_parameter);
        if (front_parameter != nullptr && front_parameter->isa<Parameter>()) {
          front_parameter_to_device_context[front_parameter] = kernel_graph_to_device_context[kernel_graph];
        }
      }
    }

    for (size_t i = 0; i < front_parameters.size(); ++i) {
      const auto &front_parameter = front_parameters[i];
      const auto &iter = front_parameter_to_device_context.find(front_parameter);
      if (iter != front_parameter_to_device_context.end()) {
        parameter_device_contexts[i] = iter->second;
      }
    }
    func_graph_to_device_contexts_[func_graph] = parameter_device_contexts;
  }

  // If there is no kernel in funcgraph, the parameter uses the default device context type.
  FuncGraphSet sub_graphs = root_func_graph_->manager()->func_graphs();
  for (auto sub_graph : sub_graphs) {
    if (func_graph_to_device_contexts_.find(sub_graph) == func_graph_to_device_contexts_.end()) {
      func_graph_to_device_contexts_[sub_graph] =
        std::vector<const DeviceContext *>(sub_graph->parameters().size(), default_context);
    }
  }
}

void ControlNodeParser::ParseDeviceContextForControlNode(const DeviceContext *default_context) {
  // Collect the call realationship between funcgraphs.
  FuncGraphCallRelation func_graph_call_relation;
  for (const auto &call_node_to_func_graphs : call_node_to_func_graphs_) {
    const auto &call_node = call_node_to_func_graphs.first;
    MS_EXCEPTION_IF_NULL(call_node);
    const auto &func_graph = call_node->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    func_graph_call_relation[func_graph].emplace_back(call_node_to_func_graphs.second);
  }

  // Topologically sort all funcgraphs according to the function call relationship.
  const auto &topo_sort_func_graphs = TopoSortForFuncGraph(root_func_graph_, &func_graph_call_relation);

  // Deduces the device context type of funcgraph outputs according to the topological order.
  for (const auto &func_graph : topo_sort_func_graphs) {
    MS_EXCEPTION_IF_NULL(func_graph);
    const auto &return_node = func_graph->return_node();
    MS_EXCEPTION_IF_NULL(return_node);
    const auto &cnode = return_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    const auto &inputs = cnode->inputs();
    const auto output_nodes = FetchAllOutputWithIndex(inputs[kReturnInputPos]);
    std::vector<const DeviceContext *> return_device_contexts;

    for (const auto &output_node : output_nodes) {
      if (output_node.first->isa<Parameter>()) {
        // If the output is parameter, get the device context type from the formal parameter.
        const auto &iter = find(func_graph->parameters().begin(), func_graph->parameters().end(), output_node.first);
        if (iter == func_graph->parameters().end()) {
          MS_LOG(EXCEPTION) << "Invalid parameter:" << output_node.first->DebugString()
                            << " for func_graph:" << func_graph->ToString();
        }
        const auto &func_graph_iter = func_graph_to_device_contexts_.find(func_graph);
        if (func_graph_iter == func_graph_to_device_contexts_.end()) {
          MS_LOG(EXCEPTION) << "Cannot find device context for funcgraph:" << func_graph->ToString();
        }
        return_device_contexts.emplace_back(func_graph_iter->second[iter - func_graph->parameters().begin()]);
      } else if (output_node.first->isa<ValueNode>()) {
        // If the output is parameter, used the default context type.
        return_device_contexts.emplace_back(default_context);
      } else if (AnfAlgo::IsCallNode(output_node.first)) {
        // If the output is call node, get the device context type by the output of funcgraph.
        const auto &func_graphs = call_node_to_func_graphs_[output_node.first];
        std::vector<const DeviceContext *> call_device_contexts;
        for (const auto &graph : func_graphs) {
          MS_EXCEPTION_IF_NULL(graph);
          const auto &node = graph->return_node();
          MS_EXCEPTION_IF_NULL(node);
          const auto &iter = control_node_to_device_contexts_.find(node);
          if (iter != control_node_to_device_contexts_.end()) {
            call_device_contexts = iter->second;
            break;
          }
        }
        // Since funcgraph has been topo-sorted according to the calling relationship, when there is a call node in
        // the output, the output type of the funcgraph called by it should have been determined, if not, an exception
        // will be thrown.
        if (call_device_contexts.empty() || call_device_contexts.size() <= output_node.second) {
          MS_LOG(EXCEPTION) << "Cannot find device context for call node:" << output_node.first->DebugString()
                            << " device contexts size:" << call_device_contexts.size()
                            << " index:" << output_node.second;
        }
        return_device_contexts.emplace_back(call_device_contexts[output_node.second]);
      } else if (output_node.first->isa<CNode>()) {
        // If the output is a cnode, get the device context type by the kernel.
        const auto &iter = front_to_backend_kernels_.find(output_node);
        if (iter == front_to_backend_kernels_.end()) {
          MS_LOG(EXCEPTION) << "Cannot find backend kernel for cnode:" << output_node.first->DebugString();
        }
        return_device_contexts.emplace_back(iter->second.second);
      } else {
        MS_LOG(EXCEPTION) << "Invalid node for return:" << output_node.first->DebugString();
      }
    }
    control_node_to_device_contexts_[return_node] = return_device_contexts;
  }
}

void ControlNodeParser::FetchFrontNodeToKernelGraph(const std::vector<KernelGraphPtr> &graphs) {
  for (const auto &graph : graphs) {
    const auto &graph_outputs = graph->graph_output_map();
    for (const auto &backend_to_front : graph_outputs) {
      front_node_to_kernel_graph_[backend_to_front.second.first] = graph;
    }
  }
}

int ControlNodeParser::FetchBranchIDByCallNode(const AnfNodePtr &call_node) {
  MS_EXCEPTION_IF_NULL(call_node);

  if (call_node_to_branch_id_.find(call_node) == call_node_to_branch_id_.end()) {
    MS_LOG(EXCEPTION) << "Invalid branch id for call_node:" << call_node->DebugString();
  }
  return call_node_to_branch_id_[call_node];
}

FuncGraphPtr ControlNodeParser::FetchKernelGraphByFrontNode(const AnfNodePtr &kernel) {
  const auto &iter = front_node_to_kernel_graph_.find(kernel);
  if (iter == front_node_to_kernel_graph_.end()) {
    return nullptr;
  }
  return iter->second;
}

bool ControlNodeParser::IsCallInputKernelGraph(const KernelGraphPtr &graph) {
  if (call_input_kernel_graphs_.find(graph) == call_input_kernel_graphs_.end()) {
    return false;
  }
  return true;
}

KernelWithIndex ControlNodeParser::FetchBackendNodeByFrontNode(const KernelWithIndex &node_with_index) {
  const auto &iter = front_to_backend_kernels_.find(node_with_index);
  if (iter != front_to_backend_kernels_.end()) {
    return iter->second.first;
  }
  return {};
}

void ControlNodeParser::FetchFrontValueNode(DeviceContext *default_context) {
  MS_EXCEPTION_IF_NULL(default_context);

  for (const auto &formal_to_real_parameter : formal_to_real_parameters_) {
    for (const auto &real_parameter_with_index : formal_to_real_parameter.second) {
      const auto &real_parameter = real_parameter_with_index.first;
      if (!real_parameter->isa<ValueNode>()) {
        continue;
      }

      const auto &iter = front_to_backend_parameters_.find(real_parameter_with_index);
      if (iter != front_to_backend_parameters_.end() && (!iter->second.empty())) {
        front_value_nodes_.emplace(real_parameter_with_index, iter->second.begin()->second);
        CreateDeviceTensorForValueNode(real_parameter_with_index, iter->second.begin()->first,
                                       iter->second.begin()->second);
      } else {
        front_value_nodes_.emplace(real_parameter_with_index, default_context);
        CreateDeviceTensorForFrontNode(real_parameter_with_index, default_context);
      }
    }
  }

  // If the output of funcgraph is a value node, it will eventually be sent to the kernel as a real parameter.
  // These the value nodes also need to create a device address.
  for (const auto &front_to_backend_parameters : front_to_backend_parameters_) {
    const auto &front_node = front_to_backend_parameters.first.first;
    MS_EXCEPTION_IF_NULL(front_node);
    if (front_node->isa<ValueNode>() && (!front_to_backend_parameters.second.empty())) {
      const auto &backend_parameter = front_to_backend_parameters.second.begin()->first;
      const auto &device_context = front_to_backend_parameters.second.begin()->second;
      CreateDeviceTensorForValueNode(front_to_backend_parameters.first, backend_parameter, device_context);
      front_value_nodes_.emplace(front_to_backend_parameters.first, device_context);
    }
  }
}

void ControlNodeParser::ParseFormalToRealParameter(const std::vector<AnfNodePtr> &control_nodes) {
  std::unordered_map<AnfNodePtr, std::set<KernelWithIndex>> formal_to_real_parameters;

  // The actual parameters of the function are divided into two parts:
  // 1. Input of partial node.
  // 2. Input of call node.
  for (const auto &node : control_nodes) {
    if (AnfAlgo::IsCallNode(node)) {
      const auto &cnode = node->cast<CNodePtr>();
      const auto &inputs = cnode->inputs();
      const auto &func_graphs = FetchFuncGraphbyCallNode(node);
      for (const auto func_graph : func_graphs) {
        const auto &parameters = func_graph->parameters();
        for (size_t i = inputs.size() - 1, j = parameters.size() - 1; i >= kCallInputStartPos && j >= 0; --i, --j) {
          std::set<KernelWithIndex> real_parameters;
          std::set<KernelWithIndex> invalid_call_nodes;
          MS_EXCEPTION_IF_NULL(inputs[i]);
          MS_EXCEPTION_IF_NULL(parameters[j]);
          FetchRealParameterByNode({inputs[i], 0}, &real_parameters, &invalid_call_nodes);
          if (real_parameters.empty()) {
            MS_LOG(EXCEPTION) << "Failed to find real parameter for formal parameter:" << inputs[i]->DebugString();
          }
          formal_to_real_parameters[parameters[j]].insert(real_parameters.begin(), real_parameters.end());
        }
      }
    } else if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimPartial)) {
      const auto &cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      const auto &inputs = cnode->inputs();
      if (inputs.size() <= kPartialFuncGraphPos || (!inputs[kPartialFuncGraphPos]->isa<ValueNode>()) ||
          (!IsValueNode<FuncGraph>(inputs[kPartialFuncGraphPos]))) {
        MS_LOG(EXCEPTION) << "Invalid partial node:" << node->DebugString();
      }
      const auto &func_graph = GetValueNode<FuncGraphPtr>(inputs[kPartialFuncGraphPos]);
      MS_EXCEPTION_IF_NULL(func_graph);
      const auto &parameters = func_graph->parameters();
      if (inputs.size() - kPartialInputStartPos > parameters.size()) {
        MS_LOG(EXCEPTION) << "Invalid partial input size:" << inputs.size()
                          << " formal parameter size:" << parameters.size();
      }
      for (size_t i = kPartialInputStartPos; i < inputs.size(); ++i) {
        std::set<KernelWithIndex> real_parameters;
        std::set<KernelWithIndex> invalid_call_nodes;
        MS_EXCEPTION_IF_NULL(inputs[i]);
        MS_EXCEPTION_IF_NULL(parameters[i - kPartialInputStartPos]);
        FetchRealParameterByNode({inputs[i], 0}, &real_parameters, &invalid_call_nodes);
        if (real_parameters.empty()) {
          MS_LOG(EXCEPTION) << "Failed to find real parameter for formal parameter:" << inputs[i]->DebugString();
        }
        formal_to_real_parameters[parameters[i - kPartialInputStartPos]].insert(real_parameters.begin(),
                                                                                real_parameters.end());
      }
    }
  }

  // When the real parameter is also a parameter, the corresponding actual parameter needs to be obtained recursively.
  for (const auto &formal_to_real_parameter : formal_to_real_parameters) {
    const auto &formal_parameter = formal_to_real_parameter.first;
    const auto &real_parameters = formal_to_real_parameter.second;
    std::set<KernelWithIndex> total_real_parameters = real_parameters;
    for (const auto &real_parameter : real_parameters) {
      if (real_parameter.first->isa<Parameter>()) {
        std::set<AnfNodePtr> invalid_real_parameter{formal_parameter};
        ParseAllRealParameterByFormalParameter(real_parameter.first, formal_to_real_parameters, &total_real_parameters,
                                               &invalid_real_parameter);
        real_to_formal_parameters_[real_parameter.first].emplace(formal_parameter);
      } else {
        total_real_parameters.emplace(real_parameter);
      }
    }
    std::swap(formal_to_real_parameters_[formal_parameter], total_real_parameters);
  }
}

void ControlNodeParser::ParseAllRealParameterByFormalParameter(const AnfNodePtr &formal_parameter,
                                                               const FormalToRealParameter &formal_to_real_parameters,
                                                               std::set<KernelWithIndex> *total_real_parameters,
                                                               std::set<AnfNodePtr> *invalid_real_parameter) {
  if (invalid_real_parameter->find(formal_parameter) != invalid_real_parameter->end()) {
    return;
  }
  invalid_real_parameter->emplace(formal_parameter);

  // Get all the actual parameters corresponding to parameter recursively.
  const auto &dst_iter = formal_to_real_parameters_.find(formal_parameter);
  if (dst_iter != formal_to_real_parameters_.end()) {
    total_real_parameters->insert(dst_iter->second.begin(), dst_iter->second.end());
    return;
  }
  const auto &src_iter = formal_to_real_parameters.find(formal_parameter);
  if (src_iter == formal_to_real_parameters.end()) {
    const auto &func_graph = formal_parameter->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    if (func_graph == root_func_graph_) {
      return;
    }
    MS_LOG(EXCEPTION) << "Invalid formal parameter:" << formal_parameter->DebugString();
  }
  const auto &real_parameters = src_iter->second;
  for (const auto &real_parameter : real_parameters) {
    MS_EXCEPTION_IF_NULL(real_parameter.first);
    if (real_parameter.first->isa<Parameter>()) {
      ParseAllRealParameterByFormalParameter(real_parameter.first, formal_to_real_parameters, total_real_parameters,
                                             invalid_real_parameter);
    } else {
      total_real_parameters->emplace(real_parameter);
    }
  }
}

void ControlNodeParser::FetchControlNodeParameter(const std::vector<AnfNodePtr> &control_nodes) {
  for (const auto &control_node : control_nodes) {
    CNodePtr cnode = control_node->cast<CNodePtr>();
    const auto &inputs = cnode->inputs();
    if (AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimReturn)) {
      break;
    } else if (AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimPartial)) {
      for (size_t i = kPartialInputStartPos; i < inputs.size(); ++i) {
        if (inputs[i]->isa<Parameter>()) {
          (void)control_node_parameters_.emplace_back(inputs[i]);
        }
      }
    } else if (cnode->input(0)->isa<CNode>() || IsValueNode<FuncGraph>(cnode->input(0))) {
      for (size_t i = kCallInputStartPos; i < inputs.size(); ++i) {
        if (inputs[i]->isa<Parameter>()) {
          (void)control_node_parameters_.emplace_back(inputs[i]);
        }
      }
    } else if (AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimSwitch)) {
      if (inputs.size() != kSwitchInputNum) {
        MS_LOG(EXCEPTION) << "Invalid switch node:" << AnfAlgo::GetNodeDebugString(control_node);
      }
      if (inputs[kSwitchCondPos]->isa<Parameter>()) {
        (void)control_node_parameters_.emplace_back(inputs[kSwitchCondPos]);
      }
    } else if (AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimSwitchLayer)) {
      if (inputs.size() != kSwitchLayerInputNum) {
        MS_LOG(EXCEPTION) << "Invalid switch node:" << AnfAlgo::GetNodeDebugString(control_node);
      }
      if (inputs[kSwitchLayerCondPos]->isa<Parameter>()) {
        (void)control_node_parameters_.emplace_back(inputs[kSwitchLayerCondPos]);
      }
    }
  }
}

void ControlNodeParser::FetchCallInputKernelGraph(const std::vector<KernelGraphPtr> &graphs,
                                                  const std::vector<DeviceContext *> &device_contexts) {
  for (size_t i = 0; i < graphs.size(); ++i) {
    const auto &graph = graphs[i];
    const auto &device_context = device_contexts[i];

    const auto inputs = graph->input_nodes();
    for (const auto &input : inputs) {
      const auto &internal_parameter_with_index = graph->GetFrontNodeByInternalParameter(input);
      if (internal_parameter_with_index.first != nullptr && AnfAlgo::IsCallNode(internal_parameter_with_index.first)) {
        call_input_kernel_graphs_[graph] = device_context;
      }
    }
  }
}

void ControlNodeParser::CreateBranchIDForCallNode(const std::vector<AnfNodePtr> &control_nodes) {
  int branch_id = kMainBranchID;

  for (const auto &control_node : control_nodes) {
    // Root funcgraph does not need to create a gather actor.
    if (AnfAlgo::IsCallNode(control_node)) {
      call_node_to_branch_id_[control_node] = ++branch_id;
    }
  }
}

void ControlNodeParser::ParseFrontToBackendParameter(const std::vector<KernelGraphPtr> &graphs,
                                                     const std::vector<DeviceContext *> &device_contexts) {
  if (graphs.size() != device_contexts.size()) {
    MS_LOG(EXCEPTION) << "Graph num is not equal to device context num.";
  }

  // Fetch the mapping relationship between front parameters and backend parameters in the kernel graphs.
  for (size_t i = 0; i < graphs.size(); ++i) {
    const auto &graph = graphs[i];
    auto device_context = device_contexts[i];
    for (const auto &parameter : graph->input_nodes()) {
      const auto &front_node = graph->GetFrontAnfByBackendAnf(parameter);
      const auto &front_node_with_index = graph->GetFrontNodeByInternalParameter(parameter);
      if (front_node == nullptr && front_node_with_index.first == nullptr) {
        MS_LOG(EXCEPTION) << "Invalid backend parameter:" << parameter->DebugString()
                          << " for kernel graph:" << graph->ToString();
      }

      if (front_node_with_index.first != nullptr) {
        std::set<KernelWithIndex> real_parameters;
        std::set<KernelWithIndex> invalid_call_nodes;
        FetchRealParameterByNode(front_node_with_index, &real_parameters, &invalid_call_nodes);
        for (const auto real_parameter : real_parameters) {
          if (real_parameter.first->isa<Parameter>() || real_parameter.first->isa<ValueNode>()) {
            front_to_backend_parameters_[real_parameter].emplace(parameter, device_context);
          }
        }
      } else {
        front_to_backend_parameters_[{front_node, 0}].emplace(parameter, device_context);
      }
    }
  }

  // Get the corresponding backend node for the real parameter according to the relationship between real
  // parameter and formal parameter.
  for (const auto &front_to_backend_parameters : front_to_backend_parameters_) {
    const auto &front_parameter = front_to_backend_parameters.first;
    const auto &backend_parameters = front_to_backend_parameters.second;
    const auto &iter = formal_to_real_parameters_.find(front_parameter.first);
    if (iter != formal_to_real_parameters_.end()) {
      for (const auto &real_parameter_with_index : iter->second) {
        const auto &real_parameter = real_parameter_with_index.first;
        if (real_parameter->isa<Parameter>()) {
          front_to_backend_parameters_[real_parameter_with_index].insert(backend_parameters.begin(),
                                                                         backend_parameters.end());
        }
      }
    }
  }
}

void ControlNodeParser::ParseCallNodeToFuncGraph(const std::vector<AnfNodePtr> &control_nodes) {
  for (const auto &control_node : control_nodes) {
    MS_EXCEPTION_IF_NULL(control_node);

    if (AnfAlgo::IsCallNode(control_node)) {
      call_node_to_func_graphs_[control_node] = AnfAlgo::GetFuncGraphbyCallNode(control_node);
    }
  }
}

const std::set<FuncGraphPtr> &ControlNodeParser::FetchFuncGraphbyCallNode(const AnfNodePtr &control_node) {
  const auto &iter = call_node_to_func_graphs_.find(control_node);
  if (iter == call_node_to_func_graphs_.end()) {
    MS_LOG(EXCEPTION) << "Invalid call node:" << control_node->DebugString();
  }
  return iter->second;
}

void ControlNodeParser::FetchHostParameterToWeight() {
  for (const auto &pair : real_to_formal_parameters_) {
    std::set<AnfNodePtr> dest_nodes;
    FetchWeightbyHostParameter(pair.first, &dest_nodes, real_to_formal_parameters_);
    host_parameter_to_weights_[pair.first] = dest_nodes;

    if (std::find(root_graph_parameters_.begin(), root_graph_parameters_.end(), pair.first) !=
        root_graph_parameters_.end()) {
      for (auto &sub_front_node : dest_nodes) {
        sub_front_node_to_root_front_node_[sub_front_node] = pair.first;
      }
    }
  }
}

void ControlNodeParser::FetchFrontToBackendKernel(const std::vector<KernelGraphPtr> &graphs,
                                                  const std::vector<DeviceContext *> &device_contexts) {
  for (size_t i = 0; i < graphs.size(); ++i) {
    const auto &graph = graphs[i];
    const auto &device_context = device_contexts[i];
    MS_EXCEPTION_IF_NULL(graph);
    auto execution_order = graph->execution_order();
    for (auto &kernel : execution_order) {
      auto front_node = graph->GetFrontAnfByBackendAnf(kernel);
      if (front_node != nullptr) {
        for (size_t j = 0; j < AnfAlgo::GetOutputTensorNum(kernel); ++j) {
          front_to_backend_kernels_[{front_node, j}] = {{kernel, j}, device_context};
          MS_LOG(DEBUG) << "Add front to backend kernel, front:" << AnfAlgo::GetNodeDebugString(front_node)
                        << "index:" << j << " addr:" << front_node << " second:" << AnfAlgo::GetNodeDebugString(kernel)
                        << "index:" << j << " addr:" << kernel;
        }
      }
    }

    const auto graph_output_map = graph->graph_output_map();
    for (const auto &output_pair : graph_output_map) {
      front_to_backend_kernels_[output_pair.second] = {output_pair.first, device_context};
    }
  }
}

void ControlNodeParser::FetchAutoMonadNode(const std::vector<AnfNodePtr> &control_nodes) {
  for (const auto &control_node : control_nodes) {
    const auto &cnode = control_node->cast<CNodePtr>();
    const auto &inputs = cnode->inputs();
    if (inputs.empty()) {
      MS_LOG(EXCEPTION) << "Invalid control node:" << AnfAlgo::GetNodeDebugString(control_node);
    }

    if (inputs[0]->isa<ValueNode>() && IsValueNode<FuncGraph>(inputs[0])) {
      for (size_t i = kCallInputStartPos; i < inputs.size(); ++i) {
        if (AnfAlgo::CheckPrimitiveType(inputs[i], prim::kPrimUpdateState)) {
          const auto &node = FetchSourceNodeByAutoMonad(inputs[i]);
          const auto &iter = front_to_backend_kernels_.find(AnfAlgo::VisitKernelWithReturnType(node, 0));
          if (iter != front_to_backend_kernels_.end()) {
            kernel_to_call_nodes_[iter->second.first.first] = control_node;
          }
        }
      }
    }
  }
}

AnfNodePtr ControlNodeParser::FetchRootGraphFrontNodeBySubFrontNode(const AnfNodePtr &sub_front_node) {
  if (sub_front_node_to_root_front_node_.count(sub_front_node) == 0) {
    return sub_front_node;
  }
  return sub_front_node_to_root_front_node_[sub_front_node];
}

bool IsFirstControlNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return true;
  }

  const auto &cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &inputs = cnode->inputs();
  for (const auto &input : inputs) {
    MS_EXCEPTION_IF_NULL(input);
    if (AnfAlgo::IsCallNode(input) || (!IsFirstControlNode(input))) {
      return false;
    }
  }
  return true;
}

void ControlNodeParser::ParseFirstControlNodeForFuncGraph(const std::vector<AnfNodePtr> &control_nodes) {
  for (const auto &control_node : control_nodes) {
    if ((AnfAlgo::IsCallNode(control_node) || AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimReturn)) &&
        IsFirstControlNode(control_node)) {
      const auto &func_graph = control_node->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      func_graph_to_first_control_nodes_[func_graph].emplace(control_node);
    }
  }
}
}  // namespace runtime
}  // namespace mindspore
