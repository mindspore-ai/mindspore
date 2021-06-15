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

bool CheckValidFuncGraphInput(const AnfNodePtr &node) {
  return (!IsPersistentDeviceTensor(node)) && (!HasAbstractMonad(node));
}

// Get the funcgraph in partial node.
FuncGraphPtr GetFuncGraphFromPartial(const AnfNodePtr &node) {
  const auto &partial_inputs = node->cast<CNodePtr>()->inputs();
  return GetValueNode<FuncGraphPtr>(partial_inputs[1]);
}

// Get the corresponding relationship between funcgraph and parameters in the switch node.
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
  for (size_t i = kMakeTupleInputStartPos; i < tuple_inputs.size(); ++i) {
    if (AnfAlgo::CheckPrimitiveType(tuple_inputs[i], prim::kPrimPartial)) {
      const auto &func_graph = GetFuncGraphFromPartial(tuple_inputs[i]);
      std::vector<AnfNodePtr> parameters;
      const auto &partial_inputs = tuple_inputs[i]->cast<CNodePtr>()->inputs();
      for (size_t j = kPartialInputStartPos; j < partial_inputs.size(); ++j) {
        if (CheckValidFuncGraphInput(partial_inputs[j])) {
          parameters.emplace_back(partial_inputs[j]);
        }
      }
      for (size_t j = kCallInputStartPos; j < call_inputs.size(); ++j) {
        if (CheckValidFuncGraphInput(call_inputs[j])) {
          parameters.emplace_back(call_inputs[j]);
        }
      }
      (*graph_to_real_parameters)[func_graph].emplace_back(parameters);
    } else if (tuple_inputs[i]->isa<ValueNode>() && IsValueNode<FuncGraph>(tuple_inputs[i])) {
      const auto &func_graph = GetValueNode<FuncGraphPtr>(tuple_inputs[i]);
      std::vector<AnfNodePtr> parameters;
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
}  // namespace

bool IsCallNode(const AnfNodePtr &node) {
  if (!node->isa<CNode>()) {
    return false;
  }
  const auto &cnode = node->cast<CNodePtr>();
  const auto &inputs = cnode->inputs();
  return inputs[0]->isa<CNode>() || (inputs[0]->isa<ValueNode>() && IsValueNode<FuncGraph>(inputs[0]));
}

std::vector<FuncGraphPtr> FetchFuncGraphbyCallNode(const CNodePtr &node) {
  std::vector<FuncGraphPtr> func_graphs;
  const auto &call_inputs = node->inputs();

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

      for (size_t i = 1; i < tuple_inputs.size(); ++i) {
        if (AnfAlgo::CheckPrimitiveType(tuple_inputs[i], prim::kPrimPartial)) {
          func_graphs.emplace_back(GetFuncGraphFromPartial(cnode_inputs[i]));
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

FuncGraphPtr FetchFuncGraphByNode(const AnfNodePtr &node) {
  auto front_node = GetFrontNodeByBackendNode(node);

  // If the front node is nullptr, we can check its inputs.
  if (front_node == nullptr) {
    if (node->isa<CNode>()) {
      const auto &cnode = node->cast<CNodePtr>();
      const auto &inputs = cnode->inputs();
      for (size_t i = 1; i < inputs.size(); ++i) {
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

FuncGraphPtr GetFuncgraphByBackendNode(const AnfNodePtr &backend_node) {
  auto front_node = GetFrontNodeByBackendNode(backend_node);
  if (front_node == nullptr) {
    return nullptr;
  }
  return front_node->func_graph();
}

std::vector<AnfNodePtr> FetchInputNodeByParameter(const AnfNodePtr &parameter,
                                                  const std::vector<AnfNodePtr> &host_ds_parameters,
                                                  std::vector<AnfNodePtr> *invalid_inputs,
                                                  const FuncGraphToParameter &graph_to_real_parameters) {
  std::vector<AnfNodePtr> input_nodes;

  // If the node has been collected, skip it.
  if (find((*invalid_inputs).begin(), (*invalid_inputs).end(), parameter) != (*invalid_inputs).end()) {
    return input_nodes;
  }

  // Record the node which has been collected.
  (*invalid_inputs).emplace_back(parameter);

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
    if (!HasAbstractMonad(input)) {
      self_inputs.emplace_back(input);
    }
  }

  size_t pos = find(self_inputs.begin(), self_inputs.end(), parameter) - self_inputs.begin();
  for (const auto parameters : graph_to_real_parameters.at(func_graph)) {
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

void ControlNodeParser::Parse(const std::vector<AnfNodePtr> &control_nodes, const std::vector<KernelGraphPtr> &graphs,
                              const std::vector<DeviceContext *> &device_contexts, const FuncGraphPtr &root_graph) {
  FetchFrontToBackendParameterMap(graphs, device_contexts, control_nodes);

  FetchFuncGraphToParameterMap(control_nodes);

  FetchHostParameterToWeightMap(control_nodes);

  FetchFrontValueNode(graphs, device_contexts);

  // Get inputs of control node which come from the host actor.
  control_node_parameters_ = FetchControlNodeParameter(control_nodes);

  front_output_nodes_ = FetchAllBranchOutputs(root_graph);
}

void ControlNodeParser::FetchValueNodeInSwitchNode(const AnfNodePtr &switch_node,
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
      FetchValueNodeInSwitchNode(call_inputs[0], value_nodes);
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

void ControlNodeParser::FetchFrontValueNode(const std::vector<KernelGraphPtr> &graphs,
                                            const std::vector<DeviceContext *> &device_contexts) {
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
          FetchValueNodeInSwitchNode(front_output_node, &value_nodes);
          for (const auto value_node : value_nodes) {
            CreateDeviceTensorForValueNode(value_node, parameter, device_contexts[index]);
            front_value_nodes_.push_back({value_node, device_contexts[index]});
          }
        }
      }
    }
  }
}

AnfNodePtr ControlNodeParser::GetCallNodeInputByPos(const AnfNodePtr &call_node, const FuncGraphPtr &func_graph,
                                                    const size_t pos) {
  const auto &cnode = call_node->cast<CNodePtr>();
  const auto &inputs = cnode->inputs();
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "Input of call node is empty, node:" << AnfAlgo::GetNodeDebugString(call_node);
  }

  if (inputs[0]->isa<CNode>()) {
    // The first input of call node is a switch node.
    if (AnfAlgo::CheckPrimitiveType(inputs[0], prim::kPrimSwitch)) {
      if (inputs.size() != kSwitchInputNum) {
        MS_LOG(EXCEPTION) << "Invalid input num of switch node, num:" << inputs.size();
      }

      auto switch_cnode = inputs[0]->cast<CNodePtr>();
      auto switch_inputs = switch_cnode->inputs();
      auto true_partial_cnode = switch_inputs[kSwitchTrueBranchPos]->cast<CNodePtr>();
      auto partial_inputs = true_partial_cnode->inputs();
      if (partial_inputs.size() <= kPartialFuncGraphPos) {
        MS_LOG(EXCEPTION) << "Invalid input num of switch node, num:" << inputs.size();
      }
      auto true_func_graph = GetValueNode<FuncGraphPtr>(partial_inputs[kPartialFuncGraphPos]);

      // If the funcgraph is not in true branch, it must be in false branch.
      if (true_func_graph != func_graph) {
        auto false_partial_cnode = switch_inputs[kSwitchFalseBranchPos]->cast<CNodePtr>();
        partial_inputs = false_partial_cnode->inputs();
      }
      if (pos + kPartialInputStartPos >= partial_inputs.size()) {
        MS_LOG(EXCEPTION) << "Invalid input pos:" << pos << " node:" << AnfAlgo::GetNodeDebugString(call_node);
      }
      return partial_inputs[pos + kPartialInputStartPos];
    } else if (AnfAlgo::CheckPrimitiveType(inputs[0], prim::kPrimSwitchLayer)) {
      if (inputs.size() != kSwitchLayerInputNum) {
        MS_LOG(EXCEPTION) << "Invalid input num of switchlayer node, num:" << inputs.size();
      }

      // The first input of call node is a switchlayer node.
      auto switch_layer_cnode = inputs[0]->cast<CNodePtr>();
      const auto &make_tuple_node = switch_layer_cnode->input(kSwitchLayerBranchPos);
      const auto &make_tuple_cnode = make_tuple_node->cast<CNodePtr>();
      const auto &tuple_inputs = make_tuple_cnode->inputs();

      for (const auto &node : tuple_inputs) {
        // tuple input is a value node of funcgraph.
        if (node->isa<ValueNode>() && IsValueNode<FuncGraph>(node)) {
          auto branch_graph = GetValueNode<FuncGraphPtr>(node);
          if (branch_graph == func_graph) {
            if (pos + kCallInputStartPos >= inputs.size()) {
              MS_LOG(EXCEPTION) << "Invalid input pos:" << pos << " node:" << AnfAlgo::GetNodeDebugString(call_node);
            }
            return inputs[pos + kCallInputStartPos];
          }
        } else if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimPartial)) {
          // tuple input is a partial, it means that part of the input of funcgraph is on partial and part on call.
          auto partial_cnode = node->cast<CNodePtr>();
          auto partial_input = partial_cnode->inputs();
          if (func_graph == GetValueNode<FuncGraphPtr>(partial_input[kPartialFuncGraphPos])) {
            if (pos + kPartialInputStartPos >= partial_input.size()) {
              return inputs[pos + kPartialInputStartPos - partial_input.size()];
            } else {
              return partial_input[pos + kPartialInputStartPos];
            }
          }
        }
      }
    } else {
      MS_LOG(EXCEPTION) << "Invalid input node:" << AnfAlgo::GetNodeDebugString(inputs[0]);
    }
  } else {
    // The first input of funcgraph is a value node of funcgraph.
    if (pos + kCallInputStartPos >= inputs.size()) {
      MS_LOG(EXCEPTION) << "Invalid input pos:" << pos << " node:" << AnfAlgo::GetNodeDebugString(call_node);
    }
    return inputs[pos + kCallInputStartPos];
  }
  return nullptr;
}

std::vector<AnfNodePtr> ControlNodeParser::FetchFuncGraphOutput(const FuncGraphPtr &func_graph,
                                                                std::vector<AnfNodePtr> *call_nodes) {
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
  const auto &call_cnode = real_output.first->cast<CNodePtr>();
  std::vector<FuncGraphPtr> func_graphs = FetchFuncGraphbyCallNode(call_cnode);
  for (const auto &graph : func_graphs) {
    auto single_outputs = FetchFuncGraphOutput(graph, call_nodes);
    outputs.insert(outputs.end(), single_outputs.begin(), single_outputs.end());
  }
  return outputs;
}

std::pair<AnfNodePtr, DeviceContext *> ControlNodeParser::FetchBackendNodeByFrontNode(
  const AnfNodePtr &front_node, const std::unordered_map<AnfNodePtr, std::vector<AnfNodePtr>> &front_to_front_parameter,
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

  const auto &front_to_front_iter = front_to_front_parameter.find(front_node);
  if (front_to_front_iter == front_to_front_parameter.end()) {
    return std::pair<AnfNodePtr, DeviceContext *>();
  }
  for (const auto &next_node : front_to_front_iter->second) {
    auto banckend_node =
      FetchBackendNodeByFrontNode(next_node, front_to_front_parameter, front_to_backend_parameter, invalid_node);
    if (banckend_node.first != nullptr) {
      return banckend_node;
    }
  }
  return std::pair<AnfNodePtr, DeviceContext *>();
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
      for (const auto &input : tuple_inputs) {
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

std::vector<AnfNodePtr> ControlNodeParser::FetchAllBranchOutputs(const FuncGraphPtr &func_graph) {
  std::vector<AnfNodePtr> call_nodes;
  return FetchFuncGraphOutput(func_graph, &call_nodes);
}

void ControlNodeParser::FetchFrontToBackendParameterMap(const std::vector<KernelGraphPtr> &graphs,
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

  // Fetch the mapping relationship between front parameters and backend parameters in the control nodes. First
  // fetch the mapping relationship of the frontparameter. When the input of the call node or the partial node
  // is a parameter node, it means that the parameter is directly transmitted. If a parameter does not have a
  // corresponding backend node, then recursively find whether the front parameter corresponding to the parameter
  // has one.
  std::unordered_map<AnfNodePtr, std::vector<AnfNodePtr>> front_to_front_parameter;
  FetchFrontToFrontParameterMap(control_nodes, &front_to_front_parameter);

  for (const auto &front_pair : front_to_front_parameter) {
    std::set<AnfNodePtr> invalid_node;
    const auto &backend_node = FetchBackendNodeByFrontNode(front_pair.first, front_to_front_parameter,
                                                           front_to_backend_parameters_, &invalid_node);
    if (backend_node.first != nullptr) {
      front_to_backend_parameters_[front_pair.first] = backend_node;
    }
  }
}

void ControlNodeParser::FetchHostParameterToWeightMap(const std::vector<AnfNodePtr> &control_nodes) {
  std::unordered_map<AnfNodePtr, std::vector<AnfNodePtr>> front_to_front_parameter;

  FetchFrontToFrontParameterMap(control_nodes, &front_to_front_parameter);

  for (const auto &pair : front_to_front_parameter) {
    std::vector<AnfNodePtr> dest_nodes;
    FetchWeightbyHostParameter(pair.first, &dest_nodes, front_to_front_parameter);
    host_parameter_to_weights_[pair.first] = dest_nodes;
  }
}

void ControlNodeParser::FetchFuncGraphToParameterMap(const std::vector<AnfNodePtr> &control_nodes) {
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
}  // namespace runtime
}  // namespace mindspore
