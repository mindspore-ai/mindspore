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
#include "runtime/framework/actor/switch_actor.h"
#include "runtime/framework/actor/gather_actor.h"
#include "ir/tensor.h"

namespace mindspore {
namespace runtime {

namespace {
// Fetch all the weight parameters related to node. It runs like this:
// if we have a map like {{a, {b, c}}, {b, {d, e}}}, final we will get {{a, {b, c, d, e}}, {b, {c, d}}}.
void FetchWeightbyHostParameter(const AnfNodePtr &node, std::vector<AnfNodePtr> *dest_nodes,
                                const std::unordered_map<AnfNodePtr, std::vector<AnfNodePtr>> &front_to_front_weight) {
  if (find((*dest_nodes).begin(), (*dest_nodes).end(), node) != (*dest_nodes).end()) {
    return;
  }
  (*dest_nodes).emplace_back(node);
  if (front_to_front_weight.find(node) == front_to_front_weight.end()) {
    return;
  }

  const auto weight_nodes = front_to_front_weight.at(node);
  for (const auto weight_node : weight_nodes) {
    FetchWeightbyHostParameter(weight_node, dest_nodes, front_to_front_weight);
  }
}

// Check whether the input is a valid parameter.
bool CheckValidFuncGraphInput(const AnfNodePtr &node) {
  if (HasAbstractMonad(node)) {
    return false;
  } else if (node->isa<Parameter>()) {
    return !HasAbstractRef(node);
  }
  return true;
}

// Get the funcgraph in partial node.
FuncGraphPtr GetFuncGraphFromPartial(const AnfNodePtr &node) {
  const auto &partial_inputs = node->cast<CNodePtr>()->inputs();
  return GetValueNode<FuncGraphPtr>(partial_inputs[1]);
}

// Get the relationship between funcgraph and parameters in the switch node.
void FetchParameterBySwitchNode(const AnfNodePtr &switch_node, FuncGraphToParameter *graph_to_real_parameters) {
  const auto &switch_cnode = switch_node->cast<CNodePtr>();
  const auto &switch_inputs = switch_cnode->inputs();
  if (switch_inputs.size() != kSwitchInputNum) {
    MS_LOG(EXCEPTION) << "Invalid control node:" << AnfAlgo::GetNodeDebugString(switch_node);
  }

  for (size_t i = kSwitchTrueBranchPos; i < kSwitchInputNum; ++i) {
    const auto &partial_node = switch_inputs[i];
    const auto &func_graph = GetFuncGraphFromPartial(partial_node);
    std::vector<AnfNodePtr> parameters;
    const auto &partial_inputs = partial_node->cast<CNodePtr>()->inputs();
    for (size_t j = kPartialInputStartPos; j < partial_inputs.size(); ++j) {
      if (CheckValidFuncGraphInput(partial_inputs[j])) {
        parameters.emplace_back(partial_inputs[j]);
      }
    }
    (*graph_to_real_parameters)[func_graph].emplace_back(parameters);
  }
}

// Get the corresponding relationship between funcgraph and parameters in the switch layer node.
void FetchParameterBySwitchLayerNode(const AnfNodePtr &switch_layer_node, const std::vector<AnfNodePtr> &call_inputs,
                                     FuncGraphToParameter *graph_to_real_parameters) {
  const auto &switch_layer_cnode = switch_layer_node->cast<CNodePtr>();
  const auto &switch_layer_inputs = switch_layer_cnode->inputs();

  if (switch_layer_inputs.size() != kSwitchLayerInputNum) {
    MS_LOG(EXCEPTION) << "Invalid control node:" << AnfAlgo::GetNodeDebugString(switch_layer_node);
  }

  auto tuple_inputs = switch_layer_inputs[kSwitchLayerBranchPos]->cast<CNodePtr>()->inputs();

  // Get the parameter corresponding to each funcgraph in make tuple.
  for (size_t i = kMakeTupleInputStartPos; i < tuple_inputs.size(); ++i) {
    if (AnfAlgo::CheckPrimitiveType(tuple_inputs[i], prim::kPrimPartial)) {
      // Tuple branch is a partial node.
      const auto &func_graph = GetFuncGraphFromPartial(tuple_inputs[i]);
      std::vector<AnfNodePtr> parameters;
      const auto &partial_inputs = tuple_inputs[i]->cast<CNodePtr>()->inputs();

      // Get inputs in partial node.
      for (size_t j = kPartialInputStartPos; j < partial_inputs.size(); ++j) {
        if (CheckValidFuncGraphInput(partial_inputs[j])) {
          parameters.emplace_back(partial_inputs[j]);
        }
      }

      // Get inputs in call node.
      for (size_t j = kCallInputStartPos; j < call_inputs.size(); ++j) {
        if (CheckValidFuncGraphInput(call_inputs[j])) {
          parameters.emplace_back(call_inputs[j]);
        }
      }
      (*graph_to_real_parameters)[func_graph].emplace_back(parameters);
    } else if (tuple_inputs[i]->isa<ValueNode>() && IsValueNode<FuncGraph>(tuple_inputs[i])) {
      // Tuple branch is a call node.
      const auto &func_graph = GetValueNode<FuncGraphPtr>(tuple_inputs[i]);
      std::vector<AnfNodePtr> parameters;

      // Get inputs in call node.
      for (size_t j = kCallInputStartPos; j < call_inputs.size(); ++j) {
        if (CheckValidFuncGraphInput(call_inputs[j])) {
          parameters.emplace_back(call_inputs[j]);
        }
      }

      (*graph_to_real_parameters)[func_graph].emplace_back(parameters);
    }
  }
}

// Create a device tensor for the front node.
// Get the output format and select kernel build info from the backend node corresponding to the front node to
// create the device address.
void CreateDeviceTensorForValueNode(const AnfNodePtr &front_node, const AnfNodePtr &backend_node,
                                    const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(device_context);

  const auto &node_value = front_node->cast<ValueNodePtr>()->value();
  if (!node_value->isa<tensor::Tensor>()) {
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
  AnfAlgo::SetOutputAddr(address, 0, front_node.get());
}

// Find the corresponding backend parameter for the front_node. If the front_node does not have the corresponding
// backend parameter, then recursively find the backend parameters of other front parameters corresponding to the
// front_node.
std::pair<AnfNodePtr, DeviceContext *> FetchBackendNodeByFrontNode(
  const AnfNodePtr &front_node,
  const std::unordered_map<AnfNodePtr, std::vector<AnfNodePtr>> &real_to_formal_front_parameters,
  const std::unordered_map<AnfNodePtr, std::vector<AnfNodePtr>> &formal_to_real_front_parameters,
  const std::unordered_map<AnfNodePtr, std::pair<AnfNodePtr, DeviceContext *>> &front_to_backend_parameter,
  std::set<AnfNodePtr> *invalid_node) {
  // Check whether the front_node has been looked for.
  if ((*invalid_node).find(front_node) != (*invalid_node).end()) {
    return std::pair<AnfNodePtr, DeviceContext *>();
  }
  (*invalid_node).insert(front_node);

  const auto front_to_backend_iter = front_to_backend_parameter.find(front_node);
  if (front_to_backend_iter != front_to_backend_parameter.end()) {
    return front_to_backend_iter->second;
  }

  const auto &real_to_formal_iter = real_to_formal_front_parameters.find(front_node);
  if (real_to_formal_iter == real_to_formal_front_parameters.end()) {
    return std::pair<AnfNodePtr, DeviceContext *>();
  }
  for (const auto &next_node : real_to_formal_iter->second) {
    auto banckend_node =
      FetchBackendNodeByFrontNode(next_node, real_to_formal_front_parameters, formal_to_real_front_parameters,
                                  front_to_backend_parameter, invalid_node);
    if (banckend_node.first != nullptr) {
      return banckend_node;
    }
  }

  const auto &formal_to_real_iter = formal_to_real_front_parameters.find(front_node);
  if (formal_to_real_iter == formal_to_real_front_parameters.end()) {
    return std::pair<AnfNodePtr, DeviceContext *>();
  }
  for (const auto &next_node : formal_to_real_iter->second) {
    auto banckend_node =
      FetchBackendNodeByFrontNode(next_node, real_to_formal_front_parameters, formal_to_real_front_parameters,
                                  front_to_backend_parameter, invalid_node);
    if (banckend_node.first != nullptr) {
      return banckend_node;
    }
  }
  return std::pair<AnfNodePtr, DeviceContext *>();
}

// Fetch all backend input nodes by parameter for gather actor.
std::vector<AnfNodePtr> FetchInputNodeByParameter(const AnfNodePtr &parameter,
                                                  const std::vector<AnfNodePtr> &host_ds_parameters,
                                                  std::set<AnfNodePtr> *invalid_inputs,
                                                  const FuncGraphToParameter &graph_to_real_parameters) {
  std::vector<AnfNodePtr> input_nodes;

  // If the node has been collected, skip it.
  if (find((*invalid_inputs).begin(), (*invalid_inputs).end(), parameter) != (*invalid_inputs).end()) {
    return input_nodes;
  }

  // Record the node which has been collected.
  (*invalid_inputs).insert(parameter);

  // If the parameter node is a parameter of host data source actor, return it.
  if (find(host_ds_parameters.begin(), host_ds_parameters.end(), parameter) != host_ds_parameters.end()) {
    input_nodes.emplace_back(parameter);
    return input_nodes;
  }

  // Check the parameter which send to its funcgraph.
  const auto &func_graph = parameter->func_graph();
  if (graph_to_real_parameters.find(func_graph) == graph_to_real_parameters.end()) {
    return input_nodes;
  }

  std::vector<AnfNodePtr> self_inputs;
  for (const auto &input : func_graph->get_inputs()) {
    // Monad input need not send to funcgraph.
    if (HasAbstractMonad(input) || HasAbstractRef(input)) {
      continue;
    }
    self_inputs.emplace_back(input);
  }

  const auto iter = find(self_inputs.begin(), self_inputs.end(), parameter);
  if (iter == self_inputs.end()) {
    MS_LOG(EXCEPTION) << "Cannot find parameter node:" << AnfAlgo::GetNodeDebugString(parameter);
  }
  size_t pos = iter - self_inputs.begin();

  for (const auto parameters : graph_to_real_parameters.at(func_graph)) {
    if (parameters.size() != self_inputs.size()) {
      MS_LOG(EXCEPTION) << "Invalid input num:" << parameters.size() << " and:" << self_inputs.size()
                        << " for func_graph:" << func_graph->ToString();
    }
    const auto input = parameters[pos];
    if (input->isa<CNode>()) {
      input_nodes.emplace_back(input);
    } else if (input->isa<Parameter>()) {
      // If input is a parameter, you need to find its input recursively.
      auto inputs = FetchInputNodeByParameter(input, host_ds_parameters, invalid_inputs, graph_to_real_parameters);
      input_nodes.insert(input_nodes.end(), inputs.begin(), inputs.end());
    }
  }
  return input_nodes;
}

// Find the output of the funcgraph, if the output is a call node, return the output of the funcgraph
// called by the call node.
std::vector<AnfNodePtr> FetchFuncGraphOutput(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *call_nodes) {
  std::vector<AnfNodePtr> outputs;
  const auto &output = func_graph->output();
  const auto &real_output = AnfAlgo::VisitKernelWithReturnType(output, 0, false, {prim::kPrimTupleGetItem});
  if (find((*call_nodes).begin(), (*call_nodes).end(), real_output.first) != (*call_nodes).end()) {
    return outputs;
  }
  if (!IsCallNode(real_output.first)) {
    outputs.push_back(real_output.first);
    return outputs;
  }

  (*call_nodes).push_back(real_output.first);
  std::vector<FuncGraphPtr> func_graphs = FetchFuncGraphbyCallNode(real_output.first);
  for (const auto &graph : func_graphs) {
    auto single_outputs = FetchFuncGraphOutput(graph, call_nodes);
    outputs.insert(outputs.end(), single_outputs.begin(), single_outputs.end());
  }
  return outputs;
}
std::vector<AnfNodePtr> FetchOutputBySwitchNode(const AnfNodePtr &switch_node, std::set<AnfNodePtr> *call_nodes,
                                                std::set<AnfNodePtr> *switch_nodes);

// Recursive interface, get all possible output nodes of call node.
std::vector<AnfNodePtr> FetchOutputByCallNode(const AnfNodePtr &call_node, std::set<AnfNodePtr> *call_nodes,
                                              std::set<AnfNodePtr> *switch_nodes) {
  std::vector<AnfNodePtr> outputs;
  if ((*call_nodes).find(call_node) != (*call_nodes).end()) {
    return outputs;
  }
  (*call_nodes).insert(call_node);

  const auto func_graphs = FetchFuncGraphbyCallNode(call_node);

  for (const auto func_graph : func_graphs) {
    if (func_graph->output()->isa<ValueNode>()) {
      outputs.push_back(func_graph->output());
    } else {
      std::vector<AnfNodePtr> sub_call_nodes;
      const std::vector<AnfNodePtr> graph_outputs = FetchFuncGraphOutput(func_graph, &sub_call_nodes);
      for (const auto &graph_output : graph_outputs) {
        if (graph_output->isa<Parameter>()) {
          outputs.push_back(graph_output);
        } else if (AnfAlgo::CheckPrimitiveType(graph_output, prim::kPrimSwitch)) {
          const auto &switch_outputs = FetchOutputBySwitchNode(graph_output, call_nodes, switch_nodes);
          outputs.insert(outputs.end(), switch_outputs.begin(), switch_outputs.end());
        } else if (IsCallNode(graph_output)) {
          const auto &call_outputs = FetchOutputByCallNode(graph_output, call_nodes, switch_nodes);
          outputs.insert(outputs.end(), call_outputs.begin(), call_outputs.end());
        } else if (graph_output->isa<CNode>()) {
          outputs.emplace_back(graph_output);
        } else {
          MS_LOG(EXCEPTION) << "Invalid front output:" << AnfAlgo::GetNodeDebugString(graph_output);
        }
      }
    }
  }

  return outputs;
}

// Recursive interface, get all possible output nodes of switch node.
std::vector<AnfNodePtr> FetchOutputBySwitchNode(const AnfNodePtr &switch_node, std::set<AnfNodePtr> *call_nodes,
                                                std::set<AnfNodePtr> *switch_nodes) {
  std::vector<AnfNodePtr> outputs;
  if ((*switch_nodes).find(switch_node) != (*switch_nodes).end()) {
    return outputs;
  }
  (*switch_nodes).insert(switch_node);

  if (!switch_node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Invalid switch node:" << AnfAlgo::GetNodeDebugString(switch_node);
  }
  const auto &inputs = switch_node->cast<CNodePtr>()->inputs();
  if (inputs.size() != kSwitchInputNum) {
    MS_LOG(EXCEPTION) << "Invalid switch node:" << AnfAlgo::GetNodeDebugString(switch_node);
  }

  for (size_t i = kSwitchTrueBranchPos; i < kSwitchInputNum; ++i) {
    if (AnfAlgo::CheckPrimitiveType(inputs[i], prim::kPrimPartial)) {
      continue;
    } else if (AnfAlgo::CheckPrimitiveType(inputs[i], prim::kPrimSwitch)) {
      const auto &switch_outputs = FetchOutputBySwitchNode(inputs[i], call_nodes, switch_nodes);
      outputs.insert(outputs.end(), switch_outputs.begin(), switch_outputs.end());
    } else if (IsCallNode(inputs[i])) {
      const auto &call_outputs = FetchOutputByCallNode(inputs[i], call_nodes, switch_nodes);
      outputs.insert(outputs.end(), call_outputs.begin(), call_outputs.end());
    } else {
      outputs.emplace_back(inputs[i]);
    }
  }

  return outputs;
}
}  // namespace

// Return true if the node has Ref abstract.
bool HasAbstractRef(const AnfNodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  auto &abs = node->abstract();
  return (abs != nullptr) && abs->isa<abstract::AbstractRef>();
}

bool IsCallNode(const AnfNodePtr &node) {
  if (!node->isa<CNode>()) {
    return false;
  }
  const auto &cnode = node->cast<CNodePtr>();
  const auto &inputs = cnode->inputs();
  return inputs[0]->isa<CNode>() || (inputs[0]->isa<ValueNode>() && IsValueNode<FuncGraph>(inputs[0]));
}

std::vector<AnfNodePtr> FetchInputsByMakeTuple(const AnfNodePtr &node) {
  std::vector<AnfNodePtr> parameters;
  if (!AnfAlgo::CheckPrimitiveType(node, prim::kPrimMakeTuple)) {
    const auto &parameter = AnfAlgo::VisitKernelWithReturnType(node, 0, false, {prim::kPrimTupleGetItem}).first;
    parameters.emplace_back(parameter);
    return parameters;
  }

  const auto &inputs = node->cast<CNodePtr>()->inputs();
  for (size_t i = kMakeTupleInputStartPos; i < inputs.size(); ++i) {
    const auto &sub_parameters = FetchInputsByMakeTuple(inputs[i]);
    parameters.insert(parameters.end(), sub_parameters.begin(), sub_parameters.end());
  }

  return parameters;
}

std::vector<FuncGraphPtr> FetchFuncGraphbyCallNode(const AnfNodePtr &node) {
  std::vector<FuncGraphPtr> func_graphs;
  if (!node->isa<CNode>()) {
    return func_graphs;
  }

  const auto &call_inputs = node->cast<CNodePtr>()->inputs();
  if (call_inputs[0]->isa<CNode>()) {
    const auto &cnode = call_inputs[0]->cast<CNodePtr>();
    const auto &cnode_inputs = cnode->inputs();
    if (AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimSwitch)) {
      for (size_t i = kSwitchTrueBranchPos; i < cnode_inputs.size(); ++i) {
        if (IsPrimitiveCNode(cnode_inputs[i], prim::kPrimPartial)) {
          func_graphs.emplace_back(GetFuncGraphFromPartial(cnode_inputs[i]));
        }
      }
    } else if (AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimSwitchLayer) &&
               AnfAlgo::CheckPrimitiveType(cnode_inputs[kSwitchLayerBranchPos], prim::kPrimMakeTuple)) {
      const auto &tuple_inputs = cnode_inputs[kSwitchLayerBranchPos]->cast<CNodePtr>()->inputs();

      for (size_t i = kMakeTupleInputStartPos; i < tuple_inputs.size(); ++i) {
        if (AnfAlgo::CheckPrimitiveType(tuple_inputs[i], prim::kPrimPartial)) {
          func_graphs.emplace_back(GetFuncGraphFromPartial(tuple_inputs[i]));
        } else if (IsValueNode<FuncGraph>(tuple_inputs[i])) {
          func_graphs.emplace_back(GetValueNode<FuncGraphPtr>(tuple_inputs[i]));
        }
      }
    } else {
      MS_LOG(EXCEPTION) << "Unable to identify call node" << node->DebugString();
    }
  } else if (call_inputs[0]->isa<ValueNode>() && IsValueNode<FuncGraph>(call_inputs[0])) {
    func_graphs.emplace_back(GetValueNode<FuncGraphPtr>(call_inputs[0]));
  } else {
    MS_LOG(EXCEPTION) << "Unable to identify call node" << node->DebugString();
  }
  return func_graphs;
}

size_t FetchOutputSizebyCallNode(const AnfNodePtr &node, std::vector<AnfNodePtr> *call_nodes) {
  if (!IsCallNode(node)) {
    MS_LOG(EXCEPTION) << "Invalid call node:" << AnfAlgo::GetNodeDebugString(node);
  }
  if (find((*call_nodes).begin(), (*call_nodes).end(), node) != (*call_nodes).end()) {
    return 0;
  }
  (*call_nodes).emplace_back(node);

  const auto &func_graphs = FetchFuncGraphbyCallNode(node);
  for (const auto &func_graph : func_graphs) {
    const auto &output = func_graph->output();
    const auto &real_output = AnfAlgo::VisitKernelWithReturnType(output, 0);

    if (IsCallNode(real_output.first)) {
      size_t output_num = FetchOutputSizebyCallNode(real_output.first, call_nodes);
      if (output_num > 0) {
        return output_num;
      }
    } else if (AnfAlgo::CheckPrimitiveType(real_output.first, prim::kPrimMakeTuple)) {
      size_t total_num = 0;
      const auto &tuple_cnode = real_output.first->cast<CNodePtr>();
      const auto &inputs = tuple_cnode->inputs();
      size_t i = 1;
      for (; i < inputs.size(); ++i) {
        if (IsCallNode(inputs[i])) {
          size_t call_output_num = FetchOutputSizebyCallNode(inputs[i], call_nodes);
          if (call_output_num == 0) {
            break;
          }
          total_num += call_output_num;
        } else {
          ++total_num;
        }
      }
      if (i == inputs.size()) {
        return total_num;
      }
    } else {
      return 1;
    }
  }
  return 0;
}

FuncGraphPtr FetchFuncGraphByNode(const AnfNodePtr &node) {
  auto front_node = GetFrontNodeByBackendNode(node);

  // If the front node is nullptr, we can check its inputs.
  if (front_node == nullptr) {
    if (node->isa<CNode>()) {
      const auto &cnode = node->cast<CNodePtr>();
      const auto &inputs = cnode->inputs();

      for (size_t i = kCallInputStartPos; i < inputs.size(); ++i) {
        const auto &func_graph = FetchFuncGraphByNode(inputs[i]);
        if (func_graph != nullptr) {
          return func_graph;
        }
      }
    } else {
      return nullptr;
    }
  }

  const auto &func_graph = front_node->func_graph();
  return func_graph;
}

AnfNodePtr GetFrontNodeByBackendNode(const AnfNodePtr &backend_node) {
  if (backend_node->func_graph() == nullptr) {
    return nullptr;
  }
  auto kernel_graph = dynamic_cast<KernelGraph *>(backend_node->func_graph().get());
  if (kernel_graph == nullptr) {
    return nullptr;
  }
  return kernel_graph->GetFrontAnfByBackendAnf(backend_node);
}

AnfNodePtr GetFrontNodeByKernelGraph(const AnfNodePtr &backend_node, const KernelGraphPtr &graph) {
  const auto &front_node = graph->GetFrontAnfByBackendAnf(backend_node);
  if (front_node != nullptr) {
    return front_node;
  }
  const auto &front_node_with_index = graph->GetFrontNodeByInternalParameter(backend_node);
  if (front_node_with_index.first == nullptr) {
    MS_LOG(EXCEPTION) << "Invalid parameter of kernel graph, parameter:" << AnfAlgo::GetNodeDebugString(backend_node);
  }
  return front_node_with_index.first;
}

FuncGraphPtr GetFuncgraphByBackendNode(const AnfNodePtr &backend_node) {
  auto front_node = GetFrontNodeByBackendNode(backend_node);
  if (front_node == nullptr) {
    return nullptr;
  }
  return front_node->func_graph();
}

void ControlNodeParser::Parse(const std::vector<AnfNodePtr> &control_nodes, const std::vector<KernelGraphPtr> &graphs,
                              const std::vector<DeviceContext *> &device_contexts, const FuncGraphPtr &root_graph) {
  root_graph_parameters_ = root_graph->parameters();

  CreateBranchIDForFuncGraph(control_nodes);

  FetchFrontToBackendParameter(graphs, device_contexts, control_nodes);

  FetchFuncGraphToParameter(control_nodes);

  FetchHostParameterToWeight(control_nodes);

  FetchFrontValueNode(control_nodes, graphs, device_contexts);

  FetchFrontToBackendKernel(graphs, device_contexts);

  control_node_parameters_ = FetchControlNodeParameter(control_nodes);

  FetchFuncGraphCallNum(control_nodes);

  FetchCallInputKernelGraph(graphs, device_contexts);

  FetchBackendInputNode();

  front_output_nodes_ = FetchAllBranchOutputs(root_graph);
}

std::vector<KernelWithIndex> ControlNodeParser::GetBackendInputByParameter(const AnfNodePtr &parameter) {
  return formal_to_real_parameters_[parameter];
}

std::vector<KernelWithIndex> ControlNodeParser::FetchBackendInputNodeByFrontNode(const AnfNodePtr &front_output) {
  std::set<AnfNodePtr> call_nodes;
  std::set<AnfNodePtr> switch_nodes;
  return FetchBackendOutputByFrontOutput(front_output, &call_nodes, &switch_nodes);
}

int ControlNodeParser::GetBranchIDByFuncGraph(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);

  if (func_graph_to_branch_id_.find(func_graph) == func_graph_to_branch_id_.end()) {
    MS_LOG(EXCEPTION) << "Invalid branch id for funcgraph:" << func_graph->ToString();
  }
  return func_graph_to_branch_id_[func_graph];
}

bool ControlNodeParser::IsCallInputKernelGraph(const KernelGraphPtr &graph) {
  if (call_input_kernel_graphs_.find(graph) == call_input_kernel_graphs_.end()) {
    return false;
  }
  return true;
}

size_t ControlNodeParser::GetCallNumByFuncGraph(const FuncGraphPtr &func_graph) {
  if (func_graph_to_call_num_.find(func_graph) == func_graph_to_call_num_.end()) {
    MS_LOG(EXCEPTION) << "Invalid funcgraph:" << func_graph->ToString();
  }

  return func_graph_to_call_num_[func_graph];
}

std::vector<AnfNodePtr> ControlNodeParser::FetchAllBranchOutputs(const FuncGraphPtr &func_graph) {
  std::vector<AnfNodePtr> call_nodes;
  return FetchFuncGraphOutput(func_graph, &call_nodes);
}

DeviceContext *ControlNodeParser::GetFrontValueNodeDeviceContext(const AnfNodePtr &value_node) {
  auto iter = std::find_if(
    front_value_nodes_.begin(), front_value_nodes_.end(),
    [value_node](const auto &front_node_with_context) { return front_node_with_context.first == value_node; });

  if (iter != front_value_nodes_.end()) {
    return iter->second;
  }
  return nullptr;
}

void ControlNodeParser::FetchValueNodeBySwitchNode(const AnfNodePtr &switch_node,
                                                   std::vector<AnfNodePtr> *value_nodes) {
  const auto &cnode = switch_node->cast<CNodePtr>();
  const auto &inputs = cnode->inputs();
  if (inputs.size() != kSwitchInputNum) {
    MS_LOG(EXCEPTION) << "Invalid switch node input num:" << inputs.size();
  }

  for (const auto &input : inputs) {
    if (input->isa<ValueNode>()) {
      const auto &node_value = input->cast<ValueNodePtr>()->value();
      if (node_value->isa<tensor::Tensor>()) {
        (*value_nodes).emplace_back(input);
      }
    } else if (IsCallNode(input)) {
      // If input is a call not, should check the switch node in its input.
      const auto &call_node = input->cast<CNodePtr>();
      const auto &call_inputs = call_node->inputs();
      if (call_inputs.empty() || (!AnfAlgo::CheckPrimitiveType(call_inputs[0], prim::kPrimSwitch))) {
        continue;
      }
      FetchValueNodeBySwitchNode(call_inputs[0], value_nodes);
    } else if (AnfAlgo::CheckPrimitiveType(input, prim::kPrimPartial)) {
      const auto &partial_node = input->cast<CNodePtr>();
      const auto &partial_inputs = partial_node->inputs();
      if (partial_inputs.size() <= kPartialFuncGraphPos) {
        MS_LOG(EXCEPTION) << "Invalid partial node input num:" << partial_inputs.size();
      }

      // if input is a partial node, get the value node in its funcgraph.
      const auto &func_graph = GetValueNode<FuncGraphPtr>(partial_inputs[kPartialFuncGraphPos]);
      if (func_graph->output()->isa<ValueNode>()) {
        (*value_nodes).emplace_back(func_graph->output());
      }
    }
  }
}

void ControlNodeParser::FetchFrontValueNode(const std::vector<AnfNodePtr> &control_nodes,
                                            const std::vector<KernelGraphPtr> &graphs,
                                            const std::vector<DeviceContext *> &device_contexts) {
  for (const auto &control_node : control_nodes) {
    CNodePtr cnode = control_node->cast<CNodePtr>();
    auto inputs = cnode->inputs();
    if (inputs[0]->isa<ValueNode>() && IsValueNode<FuncGraph>(inputs[0])) {
      auto func_graph = GetValueNode<FuncGraphPtr>(inputs[0]);
      const auto parameters = func_graph->parameters();
      if (parameters.size() != inputs.size() - kCallInputStartPos) {
        MS_LOG(EXCEPTION) << "Invalid parameters num, need:" << parameters.size()
                          << " has:" << inputs.size() - kCallInputStartPos;
      }
      for (size_t i = kCallInputStartPos; i < inputs.size(); ++i) {
        if (inputs[i]->isa<ValueNode>()) {
          const auto &node_value = inputs[i]->cast<ValueNodePtr>()->value();
          if (!node_value->isa<tensor::Tensor>()) {
            continue;
          }
          if (front_to_backend_parameters_.find(parameters[i - kCallInputStartPos]) ==
              front_to_backend_parameters_.end()) {
            MS_LOG(EXCEPTION) << "Cannot find backend parameter for front parameter:"
                              << AnfAlgo::GetNodeDebugString(parameters[i - kCallInputStartPos]);
          }

          const auto &backend_node = front_to_backend_parameters_[parameters[i - kCallInputStartPos]].first;
          const auto &device_context = front_to_backend_parameters_[parameters[i - kCallInputStartPos]].second;
          CreateDeviceTensorForValueNode(inputs[i], backend_node, device_context);
          front_value_nodes_.push_back({inputs[i], device_context});
        }
      }
    }
  }

  for (size_t index = 0; index < graphs.size(); ++index) {
    const auto &graph = graphs[index];
    MS_EXCEPTION_IF_NULL(graph);
    auto execution_order = graph->execution_order();

    for (const auto &parameter : graph->input_nodes()) {
      const auto &front_node = graph->GetFrontAnfByBackendAnf(parameter);
      const auto &internal_node = graph->GetFrontNodeByInternalParameter(parameter);

      MS_EXCEPTION_IF_NULL(parameter);
      if (IsInternalParameter(parameter, graph)) {
        auto front_node_with_index = graph->GetFrontNodeByInternalParameter(parameter);
        MS_EXCEPTION_IF_NULL(front_node_with_index.first);
        const auto &front_output_with_index =
          AnfAlgo::VisitKernelWithReturnType(front_node_with_index.first, front_node_with_index.second, false);
        auto front_output_node = front_output_with_index.first;
        MS_EXCEPTION_IF_NULL(front_output_node);
        if (AnfAlgo::CheckPrimitiveType(front_output_node, prim::kPrimSwitch)) {
          std::vector<AnfNodePtr> value_nodes;
          FetchValueNodeBySwitchNode(front_output_node, &value_nodes);
          for (const auto value_node : value_nodes) {
            CreateDeviceTensorForValueNode(value_node, parameter, device_contexts[index]);
            front_value_nodes_.push_back({value_node, device_contexts[index]});
          }
        }
      }
    }
  }
}

void ControlNodeParser::FetchFrontToFrontParameterMap(
  const std::vector<AnfNodePtr> &control_nodes,
  std::unordered_map<AnfNodePtr, std::vector<AnfNodePtr>> *front_to_front_parameter) {
  // Function used to collect the input of call node.
  const auto &call_input_parse = [front_to_front_parameter](const std::vector<AnfNodePtr> &parameters,
                                                            const std::vector<AnfNodePtr> &call_inputs,
                                                            const size_t call_input_start_pos) {
    for (size_t i = 0; i < call_inputs.size(); ++i) {
      if (call_inputs[i]->isa<Parameter>()) {
        (*front_to_front_parameter)[call_inputs[i]].push_back(parameters[i + call_input_start_pos]);
      }
    }
  };

  // Function used to collect the input of partial node.
  const auto &partial_input_parse = [call_input_parse, front_to_front_parameter](
                                      const AnfNodePtr &partial_node, const std::vector<AnfNodePtr> &call_inputs) {
    const auto &cnode = partial_node->cast<CNodePtr>();
    const auto &inputs = cnode->inputs();
    const auto &func_graph = GetValueNode<FuncGraphPtr>(inputs[kPartialFuncGraphPos]);
    const auto &parameters = func_graph->parameters();
    for (size_t i = kPartialInputStartPos; i < inputs.size(); ++i) {
      if (inputs[i]->isa<Parameter>()) {
        (*front_to_front_parameter)[inputs[i]].push_back(parameters[i - kPartialInputStartPos]);
      }
    }
    call_input_parse(parameters, call_inputs, inputs.size() - kPartialInputStartPos);
  };

  // Function used to collect the input of switch node.
  const auto &switch_input_parse = [&](const AnfNodePtr &switch_node, const std::vector<AnfNodePtr> &call_inputs) {
    CNodePtr cnode = switch_node->cast<CNodePtr>();
    const auto &switch_inputs = cnode->inputs();
    if (AnfAlgo::CheckPrimitiveType(switch_node, prim::kPrimSwitch)) {
      // Parse the switch node. The switch node has two partial node inputs.
      if (AnfAlgo::CheckPrimitiveType(switch_inputs[kSwitchTrueBranchPos], prim::kPrimPartial)) {
        partial_input_parse(switch_inputs[kSwitchTrueBranchPos], call_inputs);
        partial_input_parse(switch_inputs[kSwitchFalseBranchPos], call_inputs);
      }
    } else {
      // Parse the switchlayer node. The switchlayer node has a maketuple node input, which is a tuple of funcgraphs.
      // call_inputs will be the input of these funcgraphs.
      const auto &tuple_node = switch_inputs[kSwitchLayerBranchPos]->cast<CNodePtr>();
      const auto &tuple_inputs = tuple_node->inputs();
      for (size_t i = kMakeTupleInputStartPos; i < tuple_inputs.size(); ++i) {
        const auto &input = tuple_inputs[i];
        if (AnfAlgo::CheckPrimitiveType(input, prim::kPrimPartial)) {
          partial_input_parse(input, call_inputs);
        } else {
          auto func_graph = GetValueNode<FuncGraphPtr>(input);
          call_input_parse(func_graph->parameters(), call_inputs, 0);
        }
      }
    }
  };

  for (const auto &node : control_nodes) {
    CNodePtr cnode = node->cast<CNodePtr>();
    const auto &inputs = cnode->inputs();
    if (inputs[0]->isa<ValueNode>() && IsValueNode<FuncGraph>(inputs[0])) {
      // Call node which the first input node is a valuenode of funcgraph.
      const auto &func_graph = GetValueNode<FuncGraphPtr>(inputs[0]);
      const auto &parameters = func_graph->parameters();
      for (size_t i = kCallInputStartPos; i < inputs.size(); ++i) {
        if (inputs[i]->isa<Parameter>()) {
          (*front_to_front_parameter)[inputs[i]].push_back(parameters[i - kCallInputStartPos]);
        }
      }
    } else if (inputs[0]->isa<CNode>()) {
      // Call node which the first input node is a switch or switchlayer node.
      if ((!AnfAlgo::CheckPrimitiveType(inputs[0], prim::kPrimSwitch)) &&
          (!AnfAlgo::CheckPrimitiveType(inputs[0], prim::kPrimSwitchLayer))) {
        MS_LOG(EXCEPTION) << "First input node of call node is not switch, node:"
                          << AnfAlgo::GetNodeDebugString(inputs[0]);
      }
      std::vector<AnfNodePtr> call_inputs;
      call_inputs.assign(inputs.begin() + kCallInputStartPos, inputs.end());
      switch_input_parse(inputs[0], call_inputs);
    }
  }
}

std::vector<AnfNodePtr> ControlNodeParser::FetchControlNodeParameter(const std::vector<AnfNodePtr> &control_nodes) {
  std::vector<AnfNodePtr> parameters;

  for (const auto &control_node : control_nodes) {
    CNodePtr cnode = control_node->cast<CNodePtr>();
    const auto &inputs = cnode->inputs();
    if (AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimReturn)) {
      break;
    } else if (AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimPartial)) {
      for (size_t i = kPartialInputStartPos; i < inputs.size(); ++i) {
        if (inputs[i]->isa<Parameter>()) {
          parameters.emplace_back(inputs[i]);
        }
      }
    } else if (cnode->input(0)->isa<CNode>() || IsValueNode<FuncGraph>(cnode->input(0))) {
      for (size_t i = kCallInputStartPos; i < inputs.size(); ++i) {
        if (inputs[i]->isa<Parameter>()) {
          parameters.emplace_back(inputs[i]);
        }
      }
    }
  }

  return parameters;
}

void ControlNodeParser::FetchFuncGraphCallNum(const std::vector<AnfNodePtr> &control_nodes) {
  for (const auto &control_node : control_nodes) {
    if (IsCallNode(control_node)) {
      const auto &func_graphs = FetchFuncGraphbyCallNode(control_node);
      for (const auto &func_graph : func_graphs) {
        MS_EXCEPTION_IF_NULL(func_graph);
        if (func_graph->output()->isa<ValueNode>()) {
          continue;
        }

        if (func_graph_to_call_num_.find(func_graph) == func_graph_to_call_num_.end()) {
          func_graph_to_call_num_[func_graph] = 1;
        } else {
          func_graph_to_call_num_[func_graph]++;
        }
      }
    }
  }
}

void ControlNodeParser::FetchCallInputKernelGraph(const std::vector<KernelGraphPtr> &graphs,
                                                  const std::vector<DeviceContext *> &device_contexts) {
  for (size_t i = 0; i < graphs.size(); ++i) {
    const auto &graph = graphs[i];
    const auto &device_context = device_contexts[i];

    const auto inputs = graph->parameters();
    for (const auto &input : inputs) {
      const auto &internal_parameter_with_index = graph->GetFrontNodeByInternalParameter(input);
      if (internal_parameter_with_index.first != nullptr && IsCallNode(internal_parameter_with_index.first)) {
        call_input_kernel_graphs_[graph] = device_context;
        break;
      }
    }
  }
}

void ControlNodeParser::CreateBranchIDForFuncGraph(const std::vector<AnfNodePtr> &control_nodes) {
  int branch_id = 0;

  for (const auto &control_node : control_nodes) {
    // Root funcgraph does not need to create a gather actor.
    if (AnfAlgo::CheckPrimitiveType(control_node, prim::kPrimReturn)) {
      const auto &cnode = control_node->cast<CNodePtr>();
      const auto &inputs = cnode->inputs();
      // If the output of funcgraph is a value node, no need to create gather actor.
      if (inputs[kReturnInputPos]->isa<ValueNode>()) {
        continue;
      }

      auto func_graph = control_node->func_graph();
      func_graph_to_branch_id_[func_graph] = branch_id++;
    }
  }
}

std::vector<AnfNodePtr> FetchInputParameterbyControlNode(const AnfNodePtr &node, std::set<AnfNodePtr> *switch_nodes,
                                                         std::set<AnfNodePtr> *call_nodes) {
  std::vector<AnfNodePtr> parameters;

  if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimSwitch)) {
    if ((*switch_nodes).find(node) != (*switch_nodes).end()) {
      return parameters;
    }
    (*switch_nodes).insert(node);

    const auto &cnode = node->cast<CNodePtr>();
    const auto &inputs = cnode->inputs();
    if (inputs.size() != kSwitchInputNum) {
      MS_LOG(EXCEPTION) << "Invalid switch node:" << AnfAlgo::GetNodeDebugString(node);
    }

    for (size_t i = kSwitchTrueBranchPos; i < kSwitchInputNum; ++i) {
      if (inputs[i]->isa<Parameter>()) {
        parameters.emplace_back(inputs[i]);
      } else if (IsCallNode(inputs[i]) || AnfAlgo::CheckPrimitiveType(inputs[i], prim::kPrimSwitch)) {
        const auto &sub_parameters = FetchInputParameterbyControlNode(inputs[i], switch_nodes, call_nodes);
        parameters.insert(parameters.end(), sub_parameters.begin(), sub_parameters.end());
      }
    }
  } else if (IsCallNode(node)) {
    if ((*call_nodes).find(node) != (*call_nodes).end()) {
      return parameters;
    }
    (*call_nodes).insert(node);

    const auto &func_graphs = FetchFuncGraphbyCallNode(node);
    for (const auto &func_graph : func_graphs) {
      if (func_graph->output()->isa<Parameter>()) {
        parameters.emplace_back(func_graph->output());
      }
    }
  }
  return parameters;
}

void ControlNodeParser::FetchFrontToBackendParameter(const std::vector<KernelGraphPtr> &graphs,
                                                     const std::vector<DeviceContext *> &device_contexts,
                                                     const std::vector<AnfNodePtr> &control_nodes) {
  if (graphs.size() != device_contexts.size()) {
    MS_LOG(EXCEPTION) << "Graph num is not equal to device context num.";
  }

  // Fetch the mapping relationship between front parameters and backend parameters in the kernel graphs.
  for (size_t i = 0; i < graphs.size(); ++i) {
    const auto &graph = graphs[i];
    auto device_context = device_contexts[i];
    for (const auto &parameter : graph->parameters()) {
      auto front_node = graph->GetFrontAnfByBackendAnf(parameter);

      if (front_node != nullptr && front_node->isa<Parameter>() &&
          front_to_backend_parameters_.find(front_node) == front_to_backend_parameters_.end()) {
        front_to_backend_parameters_[front_node] = {parameter, device_context};
      }
    }
  }

  // This for loop cannot be combined with the for loop above, because the relationship between front
  // and backend needs to be consistent with HostDataSource.
  for (size_t i = 0; i < graphs.size(); ++i) {
    const auto &graph = graphs[i];
    auto device_context = device_contexts[i];
    for (const auto &parameter : graph->parameters()) {
      const auto &internal_front_node = graph->GetFrontNodeByInternalParameter(parameter);

      if (internal_front_node.first != nullptr) {
        std::set<AnfNodePtr> call_nodes;
        std::set<AnfNodePtr> switch_nodes;
        const auto &front_paramters =
          FetchInputParameterbyControlNode(internal_front_node.first, &switch_nodes, &call_nodes);
        for (const auto &front_paramter : front_paramters) {
          if (front_to_backend_parameters_.find(front_paramter) == front_to_backend_parameters_.end()) {
            front_to_backend_parameters_[front_paramter] = {parameter, device_context};
          }
        }
      }
    }
  }

  // Fetch the mapping relationship between front parameters and backend parameters in the control nodes. First
  // fetch the mapping relationship of the front parameter. When the input of the call node or the partial node
  // is a parameter node, it means that the parameter is directly transmitted. If a parameter does not have a
  // corresponding backend node, then recursively find whether the front parameter corresponding to the parameter
  // has one.
  std::unordered_map<AnfNodePtr, std::vector<AnfNodePtr>> real_to_formal_front_parameters;
  FetchFrontToFrontParameterMap(control_nodes, &real_to_formal_front_parameters);

  std::unordered_map<AnfNodePtr, std::vector<AnfNodePtr>> formal_to_real_front_parameters;
  for (const auto real_to_formal_front_parameter : real_to_formal_front_parameters) {
    for (const auto formal_parameter : real_to_formal_front_parameter.second) {
      formal_to_real_front_parameters[formal_parameter].emplace_back(real_to_formal_front_parameter.first);
    }
  }

  for (const auto &front_pair : real_to_formal_front_parameters) {
    std::set<AnfNodePtr> invalid_node;
    const auto &backend_node =
      FetchBackendNodeByFrontNode(front_pair.first, real_to_formal_front_parameters, formal_to_real_front_parameters,
                                  front_to_backend_parameters_, &invalid_node);
    if (backend_node.first != nullptr) {
      if (front_to_backend_parameters_.find(front_pair.first) == front_to_backend_parameters_.end()) {
        front_to_backend_parameters_[front_pair.first] = backend_node;
      }
    }
  }
}

void ControlNodeParser::FetchHostParameterToWeight(const std::vector<AnfNodePtr> &control_nodes) {
  std::unordered_map<AnfNodePtr, std::vector<AnfNodePtr>> front_to_front_parameter;

  FetchFrontToFrontParameterMap(control_nodes, &front_to_front_parameter);

  for (const auto &pair : front_to_front_parameter) {
    std::vector<AnfNodePtr> dest_nodes;
    FetchWeightbyHostParameter(pair.first, &dest_nodes, front_to_front_parameter);
    host_parameter_to_weights_[pair.first] = dest_nodes;
  }
}

void ControlNodeParser::FetchFuncGraphToParameter(const std::vector<AnfNodePtr> &control_nodes) {
  for (const auto &control_node : control_nodes) {
    const auto &cnode = control_node->cast<CNodePtr>();
    const auto &inputs = cnode->inputs();
    if (inputs.empty()) {
      MS_LOG(EXCEPTION) << "Invalid control node:" << AnfAlgo::GetNodeDebugString(control_node);
    }

    // Call node which the first input is a cnode.
    if (inputs[0]->isa<CNode>()) {
      const auto &switch_cnode = inputs[0]->cast<CNodePtr>();

      if (AnfAlgo::CheckPrimitiveType(switch_cnode, prim::kPrimSwitch)) {
        // Switch node.
        FetchParameterBySwitchNode(inputs[0], &func_graph_to_parameters_);
      } else if (AnfAlgo::CheckPrimitiveType(inputs[0], prim::kPrimSwitchLayer)) {
        // Switchlayer node.
        FetchParameterBySwitchLayerNode(inputs[0], inputs, &func_graph_to_parameters_);
      } else {
        MS_LOG(EXCEPTION) << "Unable to identify call node" << switch_cnode->DebugString();
      }
    } else if (inputs[0]->isa<ValueNode>() && IsValueNode<FuncGraph>(inputs[0])) {
      // Call node which the first input is a value node of funcgraph.
      const auto &func_graph = GetValueNode<FuncGraphPtr>(inputs[0]);
      std::vector<AnfNodePtr> parameters;
      for (size_t i = kCallInputStartPos; i < inputs.size(); ++i) {
        if (CheckValidFuncGraphInput(inputs[i])) {
          parameters.emplace_back(inputs[i]);
        }
      }
      func_graph_to_parameters_[func_graph].emplace_back(parameters);
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
      if (IsKernelActor(kernel) && (!IsSkippedKernelActor(kernel))) {
        auto front_node = graph->GetFrontAnfByBackendAnf(kernel);
        if (front_node != nullptr) {
          front_to_backend_kernels_[front_node] = {kernel, device_context};
        }
      }
    }
  }
}

std::vector<KernelWithIndex> ControlNodeParser::FetchBackendOutputByFrontOutput(const AnfNodePtr &front_output,
                                                                                std::set<AnfNodePtr> *call_nodes,
                                                                                std::set<AnfNodePtr> *switch_nodes) {
  std::vector<KernelWithIndex> backend_outputs;
  if (front_output->isa<ValueNode>()) {
    backend_outputs.push_back({front_output, 0});
  } else if (front_output->isa<Parameter>()) {
    // Output is a parameter.
    const auto iter = formal_to_real_parameters_.find(front_output);

    if (iter != formal_to_real_parameters_.end()) {
      backend_outputs.insert(backend_outputs.end(), iter->second.begin(), iter->second.end());
    } else {
      MS_LOG(EXCEPTION) << "Cannot find backend node for front parameter:" << AnfAlgo::GetNodeDebugString(front_output);
    }
  } else if (AnfAlgo::CheckPrimitiveType(front_output, prim::kPrimSwitch)) {
    // Output is a switch.
    const auto &switch_outputs = FetchOutputBySwitchNode(front_output, call_nodes, switch_nodes);

    for (const auto &switch_output : switch_outputs) {
      const auto outputs = FetchBackendOutputByFrontOutput(switch_output, call_nodes, switch_nodes);
      backend_outputs.insert(backend_outputs.end(), outputs.begin(), outputs.end());
    }
  } else if (IsCallNode(front_output)) {
    // Output is a call.
    const auto &call_outputs = FetchOutputByCallNode(front_output, call_nodes, switch_nodes);

    for (const auto &call_output : call_outputs) {
      const auto outputs = FetchBackendOutputByFrontOutput(call_output, call_nodes, switch_nodes);
      backend_outputs.insert(backend_outputs.end(), outputs.begin(), outputs.end());
    }
  } else if (AnfAlgo::CheckPrimitiveType(front_output, prim::kPrimMakeTuple)) {
    // Output is a make tuple.
    const auto &cnode = front_output->cast<CNodePtr>();
    const auto &inputs = cnode->inputs();

    for (size_t i = kMakeTupleInputStartPos; i < inputs.size(); ++i) {
      const auto outputs = FetchBackendOutputByFrontOutput(inputs[i], call_nodes, switch_nodes);
      backend_outputs.insert(backend_outputs.end(), outputs.begin(), outputs.end());
    }
  } else if (front_output->isa<CNode>()) {
    // Output is a kernel.
    const auto iter = front_to_backend_kernels_.find(front_output);

    if (iter != front_to_backend_kernels_.end()) {
      const auto &output_with_index = AnfAlgo::VisitKernelWithReturnType(iter->second.first, 0);
      backend_outputs.emplace_back(output_with_index);
    } else {
      MS_LOG(EXCEPTION) << "Cannot find backend node for front kernel:" << AnfAlgo::GetNodeDebugString(front_output);
    }
  } else {
    MS_LOG(EXCEPTION) << "Invalid front node:" << AnfAlgo::GetNodeDebugString(front_output);
  }

  return backend_outputs;
}

void ControlNodeParser::FetchBackendInputNodebyFrontNode(const AnfNodePtr &real_parameter,
                                                         const AnfNodePtr &formal_parameter) {
  if (real_parameter->isa<Parameter>()) {
    // Input node is a parameter from host data source actor.
    std::set<AnfNodePtr> invalid_inputs;
    std::vector<AnfNodePtr> front_inputs =
      FetchInputNodeByParameter(real_parameter, root_graph_parameters_, &invalid_inputs, func_graph_to_parameters_);

    for (const auto &front_input : front_inputs) {
      const auto node_with_index = AnfAlgo::VisitKernelWithReturnType(front_input, 0);

      if (node_with_index.first->isa<Parameter>()) {
        const auto &iter = front_to_backend_parameters_.find(real_parameter);
        if (iter == front_to_backend_parameters_.end()) {
          MS_LOG(WARNING) << "Cannot find backend node of node:" << AnfAlgo::GetNodeDebugString(node_with_index.first);
          continue;
        }
        formal_to_real_parameters_[formal_parameter].push_back({iter->second.first, 0});
      } else {
        const auto iter = front_to_backend_kernels_.find(node_with_index.first);
        if (iter == front_to_backend_kernels_.end()) {
          MS_LOG(EXCEPTION) << "Cannot find actor of front node:" << AnfAlgo::GetNodeDebugString(node_with_index.first);
        }
        formal_to_real_parameters_[formal_parameter].push_back({iter->second.first, node_with_index.second});
      }
    }
  } else if (real_parameter->isa<ValueNode>()) {
    formal_to_real_parameters_[formal_parameter].push_back({real_parameter, 0});
  } else if (IsCallNode(real_parameter)) {
    const auto func_graphs = FetchFuncGraphbyCallNode(real_parameter);
    for (const auto func_graph : func_graphs) {
      FetchBackendInputNodebyFrontNode(func_graph->output(), formal_parameter);
    }
  } else {
    // Input node is a cnode.
    const auto node_with_index = AnfAlgo::VisitKernelWithReturnType(real_parameter, 0);
    const auto iter = front_to_backend_kernels_.find(node_with_index.first);
    if (iter == front_to_backend_kernels_.end()) {
      MS_LOG(EXCEPTION) << "Cannot find backend node of node:" << AnfAlgo::GetNodeDebugString(node_with_index.first);
    }
    formal_to_real_parameters_[formal_parameter].push_back({iter->second.first, node_with_index.second});
  }
}

void ControlNodeParser::FetchBackendInputNode() {
  for (const auto &func_graph_to_parameters : func_graph_to_parameters_) {
    const auto &func_graph = func_graph_to_parameters.first;
    std::vector<AnfNodePtr> graph_inputs;
    for (const auto &input : func_graph->get_inputs()) {
      // Monad input would not send to gather actor.
      if (HasAbstractMonad(input) || (input->isa<Parameter>() && HasAbstractRef(input))) {
        continue;
      }
      graph_inputs.emplace_back(input);
    }

    // Collect all backend input node to gather, There are two situations:
    // 1. The parameter from the host data source.
    // 2. Output the kernel actor.
    for (const auto parameters : func_graph_to_parameters.second) {
      if (parameters.size() != graph_inputs.size()) {
        MS_LOG(EXCEPTION) << "Parameters num is invalid, current:" << parameters.size()
                          << " need:" << graph_inputs.size() << " func_graph:" << func_graph->ToString();
      }

      for (size_t i = 0; i < parameters.size(); ++i) {
        FetchBackendInputNodebyFrontNode(parameters[i], graph_inputs[i]);
      }
    }
  }

  for (const auto front_to_backend_parameters : front_to_backend_parameters_) {
    formal_to_real_parameters_[front_to_backend_parameters.first].push_back(
      {front_to_backend_parameters.second.first, 0});
  }
}
}  // namespace runtime
}  // namespace mindspore
